import json
import os

import time
import math
import numpy as np
import torch
import torch.nn as nn
from lib.models import EMAModel
from yacs.config import CfgNode
from lib.models.losses import Logitdiff
from lib.utils import Meters, get_last_n_median
import torch.nn.functional as F

from .base_ssl_algorithm import SemiSupervised


# taken from https://github.com/Gank0078/ACR/blob/main/train.py
class ABCLA(SemiSupervised):

    def __init__(self, cfg: CfgNode) -> None:
        super().__init__(cfg)

        self.tau1 = cfg.ALGORITHM.ACR.TAU1
        self.tau12 = cfg.ALGORITHM.ACR.TAU12
        self.tau2 = cfg.ALGORITHM.ACR.TAU2
        self.warmup_ratio = cfg.ALGORITHM.ACR.WARMUP_RATIO

        self.py_con = self.p_data
        py_uni = torch.ones(self.num_classes) / self.num_classes
        self.py_uni = py_uni.to(self.device)
        self.py_rev = torch.flip(self.py_con, dims=[0])

        self.adjustment_l1 = self.compute_adjustment_by_py(self.py_con, self.tau1)
        self.adjustment_l12 = self.compute_adjustment_by_py(self.py_con, self.tau12)
        self.adjustment_l2 = self.compute_adjustment_by_py(self.py_con, self.tau2)

        self.taumin = 0
        self.taumax = self.tau1

        u_py = torch.ones(self.num_classes) / self.num_classes
        self.u_py = u_py.to(self.device)

        self.ema_u = cfg.ALGORITHM.ACR.EMA_DECAY

    def compute_adjustment_by_py(self, py, tro):
        adjustments = torch.log(py ** tro + 1e-12)
        adjustments = adjustments.to(self.device)
        return adjustments

    def run_step(self) -> None:

        KL_div = nn.KLDivLoss(reduction='sum')

        if self.iter > self.max_iter * self.warmup_ratio:
            self.count_KL = self.count_KL / self.cfg.PERIODS.EVAL
            KL_softmax = (torch.exp(self.count_KL[0])) / (torch.exp(self.count_KL[0])+torch.exp(self.count_KL[1])+torch.exp(self.count_KL[2]))
            tau = self.taumin + (self.taumax - self.taumin) * KL_softmax
            if math.isnan(tau)==False:
                self.adjustment_l1 = self.compute_adjustment_by_py(self.py_con, tau)
        
        self.count_KL = torch.zeros(3).to(self.device)

        loss_dict = {}
        metrics_dict = {}

        # measure data time
        start = time.perf_counter()
        l_images, labels, _ = next(self._l_iter)

        # ACR requires two copies of strong augmented unlabeled samples
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

        ################################ START taken from ACR
        mask_l = (UL_LABELS != -2)
        mask_l = mask_l.cuda()

        # input concatenation
        input_concat = torch.cat([l_images, ul_weak, ul_strong], 0)

        # predictions
        feats_concat = self.model(input_concat, return_features=True)
        logits = self.model.classify(feats_concat)

        # loss computation
        l_logits = logits[:num_labels]
        
        Lx = F.cross_entropy(l_logits, labels, reduction='mean')
        loss_dict.update({"Lx": Lx})

        # unlabeled loss
        logits_u_w, logits_u_s = logits[num_labels:].chunk(2)

        logits_b = self.model.abc_classify(feats_concat)
        logits_x_b = logits_b[:num_labels]
        logits_u_w_b, logits_u_s_b = logits_b[num_labels:].chunk(2)
        Lx_b = F.cross_entropy(logits_x_b + self.adjustment_l2, labels, reduction='mean')
        loss_dict.update({"Lx_b": Lx_b})

        pseudo_label = torch.softmax((logits_u_w.detach() - self.adjustment_l1), dim=-1)
        pseudo_label_h2 = torch.softmax((logits_u_w.detach() - self.adjustment_l12), dim=-1)
        pseudo_label_b = torch.softmax(logits_u_w_b.detach(), dim=-1)
        pseudo_label_t = torch.softmax(logits_u_w.detach(), dim=-1)

        max_probs, targets_u = torch.max(pseudo_label, dim=-1)
        max_probs_h2, targets_u_h2 = torch.max(pseudo_label_h2, dim=-1)
        max_probs_b, targets_u_b = torch.max(pseudo_label_b, dim=-1)
        max_probs_t, targets_u_t = torch.max(pseudo_label_t, dim=-1)

        mask = max_probs.ge(self.conf_thres)
        mask_h2 = max_probs_h2.ge(self.conf_thres)
        mask_b = max_probs_b.ge(self.conf_thres)
        mask_t = max_probs_t.ge(self.conf_thres)

        mask_ss_b_h2 = mask_b + mask_h2
        mask_ss_t = mask + mask_t

        mask = mask.float()
        mask_b = mask_b.float()

        mask_ss_b_h2 = mask_ss_b_h2.float()
        mask_ss_t = mask_ss_t.float()

        now_mask = torch.zeros(self.num_classes)
        now_mask = now_mask.to(self.device)
        UL_LABELS[UL_LABELS==-2] = 0

        Lu = (F.cross_entropy(logits_u_s, targets_u,
                                reduction='none') * mask_ss_t).mean()
        loss_dict.update({"Lu": Lu})
        Lu_b = (F.cross_entropy(logits_u_s_b, targets_u_h2,
                                reduction='none') * mask_ss_b_h2).mean()
        loss_dict.update({"Lu_b": Lu_b})

        losses = sum(loss_dict.values())

        ################################ END

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
