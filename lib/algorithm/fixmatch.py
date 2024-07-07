import time
import os
import torch
from yacs.config import CfgNode

from .base_ssl_algorithm import SemiSupervised
import numpy as np
from .base_ssl_algorithm import cal_acc

from lib.utils import Meters, get_last_n_median
import json

class FixMatch(SemiSupervised):

    def __init__(self, cfg: CfgNode) -> None:
        super().__init__(cfg)
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

        loss_weight = confidence.ge(self.conf_thres).float()
        cons_loss = self.ul_loss(
            logits_strong, pred_class, weight=loss_weight, avg_factor=ul_weak.size(0)
        )
        loss_dict.update({"loss_cons": cons_loss})
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
                    ul_preds, ul_labels, ul_inds = self.eval_ul_dataset()
                    # ul_preds: N x D
                    # ul_labels: N

                    # pre-adjustment
                    acc_pre, m_acc_pre = cal_acc(ul_preds, ul_labels)

                    # post-adjustment

                    if self.with_darp:        
                        p = self.darp_optimizer.step(ul_preds.detach().softmax(dim=1), ul_inds)
                        p = p.cpu()
                    else:
                        p = ul_preds.detach().softmax(dim=1)

                    acc_post, m_acc_post = cal_acc(p, ul_labels)
                
                    conf, pred_class = torch.max(p, dim=1)
                    p = np.array(p)
                    ul_labels = np.array(ul_labels)
                    class_weight = 1 / np.array(self.p_data.cpu())

                    conf = np.array(conf)
                    threshold_pos = []
                    for kbatch in range(len(conf)):
                        if conf[kbatch] > self.conf_thres:
                            threshold_pos.append(kbatch)
                    threshold_pos = np.array(threshold_pos)
                    
                    true_class = ul_labels[threshold_pos]
                    target_class = np.argmax(p, axis = 1)[threshold_pos]

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
                    self.gain_iters.append((m_acc_post-m_acc_pre)*100)
                    self.gain_iters.append((acc_post-acc_pre)*100)
                    self.acc_iters.append(acc_test)

                    print(f"pre adjustment accuracy: {acc_pre}")
                    print(f"post adjustment accuracy: {acc_post}")
                    print(f"pre adjustment balanced accuracy: {m_acc_pre}")
                    print(f"post adjustment balanced accuracy: {m_acc_post}")

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