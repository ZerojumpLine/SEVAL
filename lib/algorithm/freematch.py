import time
import os
import torch
from yacs.config import CfgNode

from .base_ssl_algorithm import SemiSupervised

import numpy as np
from .base_ssl_algorithm import cal_acc

from lib.utils import Meters, get_last_n_median
import json

# taken from https://github.com/microsoft/Semi-supervised-learning/tree/main/semilearn/algorithms/freematch

class FreeMatch(SemiSupervised):

    def __init__(self, cfg: CfgNode) -> None:
        super().__init__(cfg)

        self.use_quantile = False
        self.clip_thresh = False
        self.lambda_e = 0.001

        self.m = 0.999
        
        self.p_model = torch.ones((self.num_classes)) / self.num_classes
        self.label_hist = torch.ones((self.num_classes)) / self.num_classes
        self.time_p = self.p_model.mean()
        self.record = cfg.PERIODS.RECORD

    def run_step(self) -> None:
        loss_dict = {}
        metrics_dict = {}

        # measure data time
        start = time.perf_counter()
        l_images, labels, _ = next(self._l_iter)

        (ul_weak, ul_strong), UL_LABELS, ul_indices = next(self._ul_iter)
        data_time = time.perf_counter() - start

        # load images and labels onto gpu
        if torch.cuda.is_available():
            l_images = l_images.to(self.device)
            labels = labels.to(self.device).long()
            ul_weak = ul_weak.to(self.device)
            ul_strong = ul_strong.to(self.device)
            UL_LABELS = UL_LABELS.to(self.device)

        num_labels = labels.size(0)

        # input concatenation
        input_concat = torch.cat([l_images, ul_weak, ul_strong], 0)

        # predictions
        logits_concat = self.model(input_concat)

        # loss computation
        l_logits = logits_concat[:num_labels]

        # logit adjustment in train-time.
        if self.with_la:
            l_logits += (self.tau * self.p_data.view(1, -1).log())
        
        cls_loss = self.l_loss(l_logits, labels)
        loss_dict.update({"loss_cls": cls_loss})

        # unlabeled loss
        logits_weak, logits_strong = logits_concat[num_labels:].chunk(2)
        p = logits_weak.detach().softmax(dim=1)  # soft pseudo labels
        if self.with_align:
            p = self.dist_align(p)  # distribution alignment

        with torch.no_grad():
            if self.with_darp:        
                p = self.darp_optimizer.step(p, ul_indices)
            # final pseudo-labels with confidence
            confidence, pred_class = torch.max(p, dim=1)

        # loss_weight = confidence.ge(self.conf_thres).float()
        # logits_weak -> loss_weight
        if not self.p_model.is_cuda:
            self.p_model = self.p_model.to(self.device)
        if not self.label_hist.is_cuda:
            self.label_hist = self.label_hist.to(self.device)
        if not self.time_p.is_cuda:
            self.time_p = self.time_p.to(self.device)

        self.update(p)

        max_probs, max_idx = p.max(dim=-1)
        mod = self.p_model / torch.max(self.p_model, dim=-1)[0]
        loss_weight = max_probs.ge(self.time_p * mod[max_idx]).to(max_probs.dtype)

        cons_loss = self.ul_loss(
            logits_strong, pred_class, weight=loss_weight, avg_factor=ul_weak.size(0)
        )
        loss_dict.update({"loss_cons": cons_loss})

        if loss_weight.sum() > 0:
            ent_loss, _ = entropy_loss(loss_weight, logits_strong, self.p_model, self.label_hist)
        else:
            ent_loss = 0.0
        loss_dict.update({"ent_loss": self.lambda_e * ent_loss})

        losses = sum(loss_dict.values())

        # compute batch-wise accuracy and update metrics_dict
        top1, top5 = self.accuracy(l_logits, labels)
        metrics_dict.update(loss_dict)
        metrics_dict.update({"top1": top1, "top5": top5})

        # update params and schedule learning rates
        self.optimizer.zero_grad()
        losses.backward()
        self.optimizer.step()
        if self.apply_scheduler:
            self.scheduler.step()

        current_lr = self.optimizer.param_groups[0]["lr"]
        ema_decay = self.ema_model.update(self.model, step=self.iter, current_lr=current_lr)

        # measure iter time
        iter_time = time.perf_counter() - start

        # logging
        self.iter_timer.update(iter_time, n=l_images.size(0))
        self.meters.put_scalar(
            "misc/iter_time", self.iter_timer.avg, n=l_images.size(0), show_avg=False
        )
        self.meters.put_scalar("train/ema_decay", ema_decay, show_avg=False)
        self.meters.put_scalar("misc/data_time", data_time, n=l_images.size(0))
        self.meters.put_scalar("misc/lr", current_lr, show_avg=False)

        # make a log for accuracy and losses
        self._write_metrics(metrics_dict, n=l_images.size(0), prefix="train")
    
    def update(self, probs_x_ulb):
        max_probs, max_idx = torch.max(probs_x_ulb, dim=-1,keepdim=True)

        if self.use_quantile:
            self.time_p = self.time_p * self.m + (1 - self.m) * torch.quantile(max_probs,0.8) #* max_probs.mean()
        else:
            self.time_p = self.time_p * self.m + (1 - self.m) * max_probs.mean()
        
        if self.clip_thresh:
            self.time_p = torch.clip(self.time_p, 0.0, 0.95)

        self.p_model = self.p_model * self.m + (1 - self.m) * probs_x_ulb.mean(dim=0)
        hist = torch.bincount(max_idx.reshape(-1), minlength=self.p_model.shape[0]).to(self.p_model.dtype) 
        self.label_hist = self.label_hist * self.m + (1 - self.m) * (hist / hist.sum())
    
    def train(self):
        self.logger.info(f"Starting training from iteration {self.start_iter}")
        self.model.train()

        for self.iter in range(self.start_iter, self.max_iter):
            if (
                self.cfg.MODEL.LOSS.WITH_LABELED_COST_SENSITIVE
                and (self.iter + 1) >= self.cfg.MODEL.LOSS.WARMUP_ITERS and not self.is_warmed
            ):
                # warmup, LDAM-DRW (deferred reweight)
                self.is_warmed = True
                self.l_loss = self.build_labeled_loss(self.cfg, warmed_up=True)

            # one step of forward path and backprop
            self.run_step()

            # increase the meter's iteration
            self.meters.step()

            # eval period
            if ((self.iter + 1) % self.cfg.PERIODS.EVAL == 0):
                self.evaluate(self.model)
                self.dist_logger.write()

                if self.record:
                    # calculate correctness and gain.
                    acc_test = self.eval_history["test/top1"][-1]
                    ul_preds, ul_labels, _ = self.eval_ul_dataset()
                    # ul_preds: N x D
                    # ul_labels: N
                
                    conf, pred_class = torch.max(ul_preds.detach().softmax(dim=1), dim=1)
                    ul_preds = np.array(ul_preds)
                    ul_labels = np.array(ul_labels)
                    class_weight = 1 / np.array(self.p_data.cpu())

                    conf = np.array(conf)
                    threshold_pos = []  

                    mod = self.p_model / torch.max(self.p_model, dim=-1)[0]
                    
                    for kbatch in range(len(conf)):
                        if conf[kbatch] > self.time_p * mod[pred_class[kbatch]]:
                            threshold_pos.append(kbatch)
                    threshold_pos = np.array(threshold_pos)
                    
                    true_class = ul_labels[threshold_pos]
                    target_class = np.argmax(ul_preds, axis = 1)[threshold_pos]

                    acc_numerator = 0
                    correct_y1_pos = np.where(true_class == target_class)[0]
                    for cor_pos in correct_y1_pos:
                        acc_numerator = acc_numerator + class_weight[true_class[cor_pos]]
                    acc_denominator_1 = 1e-6
                    for all_pos in range(len(ul_labels)):
                        acc_denominator_1 = acc_denominator_1 + class_weight[ul_labels[all_pos]]
                    acc_denominator_2 = 1e-6
                    for all_pos in range(len(threshold_pos)):
                        acc_denominator_2 = acc_denominator_2 + class_weight[true_class[all_pos]]

                    correctness = acc_numerator * acc_numerator / acc_denominator_1 / acc_denominator_2

                    self.correctness_iters.append(correctness)
                    self.gain_iters.append(0)
                    self.acc_iters.append(acc_test)

                    np.savetxt(self.coorectness_savename, np.array(self.correctness_iters), delimiter = ',')
                    np.savetxt(self.gain_savename, np.array(self.gain_iters), delimiter = ',')
                    np.savetxt(self.acc_savename, np.array(self.acc_iters), delimiter = ',')

            # periodically save checkpoints
            if (
                self.cfg.PERIODS.CHECKPOINT > 0
                and (self.iter + 1) % self.cfg.PERIODS.CHECKPOINT == 0
            ):
                save_ema_model = self.with_ul
                if self.cfg.ALGORITHM.NAME == "DARP_ESTIM":
                    save_ema_model = False
                self.save_checkpoint(save_ema_model=save_ema_model)

            # print logs
            if (((self.iter + 1) % self.cfg.PERIODS.LOG == 0 or (self.iter + 1) == self.max_iter)):
                assert self.cfg.PERIODS.EVAL == self.cfg.PERIODS.LOG
                for writer in self.writers:
                    writer.write(self.meters)
                self.meters.reset()

            # start new generation after evaluation!
            if (self.iter + 1) % self.gen_period_steps == 0:
                crest_names = ["ReMixMatchCReST", "FixMatchCReST"]
                with_crest = self.cfg.ALGORITHM.NAME in crest_names
                # new generation except for the last iteration
                if with_crest and (self.iter + 1) < self.max_iter:
                    self.new_generation()
        print()
        print()
        print()

        prefixes = ["valid/top1", "test/top1"]
        self.logger.info("Median 20 Results:")
        self.logger.info(
            ", ".join(
                f"{k}_median (20): {get_last_n_median(v, n=20):.2f}"
                for k, v in self.eval_history.items() if k in prefixes
            )
        )
        print()
        prefixes = ["valid/top1_la", "test/top1_la"]
        self.logger.info("Median 20 Results:")
        self.logger.info(
            ", ".join(
                f"Logit adjusted {k}_median (20): {get_last_n_median(v, n=20):.2f}"
                for k, v in self.eval_history.items() if k in prefixes
            )
        )
        print()

        # final checkpoint
        self.save_checkpoint(save_ema_model=self.with_ul)

        # test top1 and median print
        print()
        save_path = self.cfg.OUTPUT_DIR
        with open(os.path.join(save_path, "results.json"), "w") as f:
            eval_history = {k: v for k, v in self.eval_history.items()}
            f.write(json.dumps(eval_history, indent=4, sort_keys=True))
        self.logger.info(f"final results (results.json) saved on: {save_path}.")

        for writer in self.writers:
            writer.close()


## utils
def replace_inf_to_zero(val):
    val[val == float('inf')] = 0.0
    return val

def entropy_loss(mask, logits_s, prob_model, label_hist):
    mask = mask.bool()

    # select samples
    logits_s = logits_s[mask]

    prob_s = logits_s.softmax(dim=-1)
    _, pred_label_s = torch.max(prob_s, dim=-1)

    hist_s = torch.bincount(pred_label_s, minlength=logits_s.shape[1]).to(logits_s.dtype)
    hist_s = hist_s / hist_s.sum()

    # modulate prob model 
    prob_model = prob_model.reshape(1, -1)
    label_hist = label_hist.reshape(1, -1)
    # prob_model_scaler = torch.nan_to_num(1 / label_hist, nan=0.0, posinf=0.0, neginf=0.0).detach()
    prob_model_scaler = replace_inf_to_zero(1 / label_hist).detach()
    mod_prob_model = prob_model * prob_model_scaler
    mod_prob_model = mod_prob_model / mod_prob_model.sum(dim=-1, keepdim=True)

    # modulate mean prob
    mean_prob_scaler_s = replace_inf_to_zero(1 / hist_s).detach()
    # mean_prob_scaler_s = torch.nan_to_num(1 / hist_s, nan=0.0, posinf=0.0, neginf=0.0).detach()
    mod_mean_prob_s = prob_s.mean(dim=0, keepdim=True) * mean_prob_scaler_s
    mod_mean_prob_s = mod_mean_prob_s / mod_mean_prob_s.sum(dim=-1, keepdim=True)

    loss = mod_prob_model * torch.log(mod_mean_prob_s + 1e-12)
    loss = loss.sum(dim=1)
    return loss.mean(), hist_s.mean()