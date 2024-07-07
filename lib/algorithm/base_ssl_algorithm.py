import math

import torch.nn as nn
from yacs.config import CfgNode

from lib.models import EMAModel
from lib.models.dist_align import DistributionAlignment
from lib.models.losses import build_loss

from .base_algorithm import BaseAlgorithm
from .darp_reproduce import DARP

from lib.dataset.base import BaseNumpyDataset
from lib.dataset.loader.build import _build_loader
from lib.dataset.aves import iNatDataset
import numpy as np
import torch

def cal_acc(preacts, targets_all):
    # -> preacts.     N x C
    # -> targets_all. N
    
    # mean of acc
    preacts = np.array(preacts)
    targets_all = np.array(targets_all)
    acc_cls = []
    for kcls in range(preacts.shape[-1]):
        label = kcls
        targets_y1 = np.where(targets_all==label)[0]
        pred_class = np.argmax(preacts, axis = 1)[targets_y1]
        target_class = targets_all[targets_y1]

        acc = np.sum(pred_class == target_class) / len(target_class)

        acc_cls.append(acc)
    
    # return np.mean(acc_cls)

    # sample-wise acc
    preacts = np.array(preacts)
    targets_all = np.array(targets_all)
    acc_numerator = 0
    acc_denominator = 1e-6
    for kcls in range(preacts.shape[-1]):
        label = kcls
        targets_y1 = np.where(targets_all==label)[0]
        pred_class = np.argmax(preacts, axis = 1)[targets_y1]
        target_class = targets_all[targets_y1]

        acc_numerator = acc_numerator + np.sum(pred_class == target_class)
        acc_denominator = acc_denominator + len(target_class)

    acc =  acc_numerator / acc_denominator
    return acc, np.mean(acc_cls)


class SemiSupervised(BaseAlgorithm):

    def __init__(self, cfg: CfgNode) -> None:
        super().__init__(cfg)
        self.resume = None
        self.ema_model = EMAModel(
            self.model,
            cfg.MODEL.EMA_DECAY,
            cfg.MODEL.EMA_WEIGHT_DECAY,
            device=self.device,
            resume=self.resume
        )
        self.ul_loss = self.build_unlabeled_loss(cfg)
        self.apply_scheduler = cfg.SOLVER.APPLY_SCHEDULER

        # confidence threshold for unlabeled predictions in PseudoLabel and FixMatch algorithms
        self.conf_thres = cfg.ALGORITHM.CONFIDENCE_THRESHOLD

        # distribution alignment
        self.with_align = cfg.MODEL.DIST_ALIGN.APPLY
        if self.with_align:
            self.dist_align = DistributionAlignment(cfg, self.p_data)

        # apply darp
        self.with_darp = cfg.ALGORITHM.DARP.APPLY
        if self.with_darp:
            ul_dataset = self.ul_loader.dataset
            self.darp_optimizer = DARP(cfg, ul_dataset)
        
        self.output_dir = cfg.OUTPUT_DIR
        self.correctness_iters = []
        self.gain_iters = []
        self.acc_iters = []
        self.coorectness_savename = self.output_dir + "/correctness.txt"
        self.gain_savename = self.output_dir + "/gain.txt"
        self.acc_savename = self.output_dir + "/acc.txt"

        # if we train with aves / imagenet, we do not load all the things into memory
        self.numpyflag = self.cfg.DATASET.NAME != "aves" and self.cfg.DATASET.NAME != "imagenet"
        ul_dataset = self.ul_loader.dataset
        if self.numpyflag:
        
            # unlabeled dataset configuration
            
            ul_test_dataset = BaseNumpyDataset(
                ul_dataset.select_dataset(),
                transforms=self.test_loader.dataset.transforms,
                is_ul_unknown=ul_dataset.is_ul_unknown
            )
            self.ul_test_loader = _build_loader(
                self.cfg, ul_test_dataset, is_train=False, has_label=False
            )

            # save init stats
            l_dataset = self.l_loader.dataset
            self.init_l_data, self.l_transforms = l_dataset.select_dataset(return_transforms=True)
            self.current_l_dataset = l_dataset
        else:
            ul_data, ul_labels, _ =  ul_dataset.select_dataset(return_transforms=True)
            ul_test_dataset = iNatDataset(ul_data, ul_labels, transform=self.test_loader.dataset.transforms)
            self.ul_test_loader = _build_loader(
                self.cfg, ul_test_dataset, is_train=False, has_label=False
            )
            l_dataset = self.l_loader.dataset
            self.init_l_data, _, self.l_transforms = l_dataset.select_dataset(return_transforms=True)
            self.current_l_dataset = l_dataset

    def build_unlabeled_loss(self, cfg: CfgNode) -> nn.Module:
        loss_type = cfg.MODEL.LOSS.UNLABELED_LOSS
        loss_weight = cfg.MODEL.LOSS.UNLABELED_LOSS_WEIGHT

        ul_loss = build_loss(cfg, loss_type, class_count=None, loss_weight=loss_weight)
        return ul_loss

    def cons_rampup_func(self) -> float:
        max_iter = self.cfg.SOLVER.MAX_ITER
        rampup_schedule = self.cfg.ALGORITHM.CONS_RAMPUP_SCHEDULE
        rampup_ratio = self.cfg.ALGORITHM.CONS_RAMPUP_ITERS_RATIO
        rampup_iter = max_iter * rampup_ratio

        if rampup_schedule == "linear":
            rampup_value = min(float(self.iter) / rampup_iter, 1.0)
        elif rampup_schedule == "exp":
            rampup_value = math.exp(-5 * (1 - min(float(self.iter) / rampup_iter, 1))**2)
        return rampup_value

    def train(self) -> None:
        super().train()

    def evaluate(self, model=None):
        if self.cfg.ALGORITHM.NAME == "cRT":
            eval_model = self.model
        else:
            eval_model = self.ema_model
        return super().evaluate(eval_model)
    
    def eval_ul_dataset(self):
        self.logger.info("evaluating ul data as test set...")
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
