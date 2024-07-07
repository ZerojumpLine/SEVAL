import time

import torch
from yacs.config import CfgNode
from collections import Counter
from copy import deepcopy

from .base_ssl_algorithm import SemiSupervised

import numpy as np
from .base_ssl_algorithm import cal_acc

from lib.utils import Meters, get_last_n_median
import json
import os

# taken from https://github.com/microsoft/Semi-supervised-learning/tree/main/semilearn/algorithms/adsh

class Adsh(SemiSupervised):

    def __init__(self, cfg: CfgNode) -> None:
        super().__init__(cfg)
        
        self.adsh_tau_1 = 0.95
        self.adsh_s = torch.ones((self.num_classes,)) * self.adsh_tau_1
        
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

        loss_weight = confidence.ge(torch.exp(-self.adsh_s.to(confidence.device)[pred_class])).float()

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
    
    def update(self):
        ul_preds, ul_labels, _ = self.eval_ul_dataset_quite()
        conf_all, pred_all = torch.max(ul_preds.detach().softmax(dim=1), dim=1)

        C = []
        for y in range(self.num_classes):
            C.append(torch.sort(conf_all[pred_all == y], descending=True)[0])  # descending order

        rho = 1.0
        for i in range(len(C[0])):
            if C[0][i] < self.adsh_tau_1:
                break
            rho = i / len(C[0])

        for k in range(self.num_classes):
            if len(C[k]) != 0:
                self.adsh_s[k] = - torch.log(C[k][int(len(C[k]) * rho) - 1])

    def eval_ul_dataset_quite(self):
        ul_dataset = self.ul_loader.dataset
        ul_preds = torch.zeros(len(ul_dataset), self.num_classes)
        ul_labels = torch.zeros(len(ul_dataset)).long()
        ul_inds = torch.zeros(len(ul_dataset)).long()

        model = self.ema_model
        model.eval()
        with torch.no_grad():
            for i, (images, labels, inds) in enumerate(self.ul_test_loader):
                if torch.cuda.is_available():
                    images = images.to(self.device)
                outputs = model(images, is_train=False)
                ul_preds[inds, :] = outputs.detach().data.cpu()
                ul_labels[inds] = labels
                ul_inds[inds] = inds
        model.train()

        return ul_preds, ul_labels, ul_inds

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

                # update adsh parameters
                self.update()

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
                    
                    for kbatch in range(len(conf)):
                        if conf[kbatch] > self.p_cutoff * (self.classwise_acc[pred_class[kbatch]] / (2. - self.classwise_acc[pred_class[kbatch]])):
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