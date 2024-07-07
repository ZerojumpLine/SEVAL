import torch
import torch.nn as nn
from torch import Tensor
import numpy as np
import pandas as pd
import scipy
import math

from lib.utils import Meters
from typing import Tuple, Union

def softmax(x, T, b=0):
    x = x / T + b
    f = np.exp(x - np.max(x, axis = 0))  # shift values
    return f / f.sum(axis = 0)

def get_conf_sel(preacts_trainval, conf_threshold, cls_trainval_inds, num_class):
    '''
    Get the samples that have confidences larger than threshold
    '''
    pd_trainval_cls_sels = []
    for l_index in range(num_class):
        
        cls_trainval_ind = cls_trainval_inds[l_index]
        
        preacts_trainval_cls = preacts_trainval[cls_trainval_ind, :]

        probmax_trainval_cls = np.max(softmax(preacts_trainval[cls_trainval_ind, :].transpose(), T = 1).transpose(), axis = 1)
        trainval_cls_index = np.where(probmax_trainval_cls > conf_threshold[l_index])
        #
        preacts_trainval_cls_sel = preacts_trainval_cls[trainval_cls_index[0]]
    
        pd_trainval_cls_sel = pd.DataFrame({'z0': preacts_trainval_cls_sel[:, 0], 'z1': preacts_trainval_cls_sel[:, 1],
                                   'z2': preacts_trainval_cls_sel[:, 2], 'z3': preacts_trainval_cls_sel[:, 3],
                                   'z4': preacts_trainval_cls_sel[:, 4], 'z5': preacts_trainval_cls_sel[:, 5],
                                   'z6': preacts_trainval_cls_sel[:, 6], 'z7': preacts_trainval_cls_sel[:, 7],
                                   'z8': preacts_trainval_cls_sel[:, 8], 'z9': preacts_trainval_cls_sel[:, 9]})
        
        pd_trainval_cls_sels.append(pd_trainval_cls_sel)
    
    return pd_trainval_cls_sels

def get_conf_sel_correct(preacts_trainval, conf_threshold, cls_trainval_inds, targets_trainval, num_class):
    '''
    Get the correctly labeled samples that have confidences larger than threshold
    '''
    pd_trainval_cls_sels_correct = []
    for l_index in range(num_class):
        
        cls_trainval_ind = cls_trainval_inds[l_index]
    
        preacts_trainval_cls = preacts_trainval[cls_trainval_ind, :]

        preds_argmax_cls = np.argmax(targets_trainval[cls_trainval_ind, :], axis = 1)
        correct_cls_index = np.where(preds_argmax_cls == l_index)

        probmax_trainval_cls_correct = np.max(softmax(preacts_trainval[cls_trainval_ind, :][correct_cls_index[0], :].transpose(), T = 1).transpose(), axis = 1)
        trainval_cls_index_correct = np.where(probmax_trainval_cls_correct > conf_threshold[l_index])
        #
        preacts_trainval_cls_sel_correct = preacts_trainval_cls[correct_cls_index[0]][trainval_cls_index_correct[0]]
        
        pd_trainval_cls_sel_correct = pd.DataFrame({'z0': preacts_trainval_cls_sel_correct[:, 0], 'z1': preacts_trainval_cls_sel_correct[:, 1],
                                   'z2': preacts_trainval_cls_sel_correct[:, 2], 'z3': preacts_trainval_cls_sel_correct[:, 3],
                                   'z4': preacts_trainval_cls_sel_correct[:, 4], 'z5': preacts_trainval_cls_sel_correct[:, 5],
                                   'z6': preacts_trainval_cls_sel_correct[:, 6], 'z7': preacts_trainval_cls_sel_correct[:, 7],
                                   'z8': preacts_trainval_cls_sel_correct[:, 8], 'z9': preacts_trainval_cls_sel_correct[:, 9]})
        
        pd_trainval_cls_sels_correct.append(pd_trainval_cls_sel_correct)
        
        
    return pd_trainval_cls_sels_correct

def cal_diff(preacts_y1, targets_y1, probs_y1, correcnt_portion, p_data, threshold):
    # -> preacts_y1.  N x C
    # -> targets_y1.  N x C
    # -> threshold    float

    threshold_pos = np.where(np.max(probs_y1, axis = 1) > threshold)[0] 

    true_class = np.argmax(targets_y1, axis = 1)[threshold_pos]
    target_class = np.argmax(preacts_y1, axis = 1)[threshold_pos]

    # acc = np.sum(true_class == target_class) / len(target_class)

    # weighted acc, 1 * N1 + 1 * N2 / (N1 + N2)
    class_weight = 1 / p_data
    acc_numerator = 0
    correct_y1_pos = np.where(true_class == target_class)[0]
    for cor_pos in correct_y1_pos:
        acc_numerator = acc_numerator + class_weight[true_class[cor_pos]]
    acc_denominator = 1e-6
    for all_pos in range(len(threshold_pos)):
        acc_denominator = acc_denominator + class_weight[true_class[all_pos]]

    acc = acc_numerator / acc_denominator

    diff_cls = np.abs(acc - correcnt_portion)
    
    return diff_cls

def one_hot_embedding(labels: Tensor, num_classes: int) -> Tensor:
    '''Embedding labels to one-hot form.
    Args:
      labels: (LongTensor) class labels, sized [N,].
      num_classes: (int) number of classes.
    Returns:
      (tensor) encoded labels, sized [N,#classes].
    '''
    y = np.eye(num_classes)  # [D,D]

    return y[labels]  # [N,D]

class OptimizeTH(nn.Module):

    def __init__(self, num_classes: int, th_class: int, threshold: float, p_data: Tensor) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.th_class = th_class
        self.empty = True
        self.preacts = None
        # N x C
        self.targets_all = None
        # N x C
        self.correcnt_portion = threshold
        self.p_data = p_data / np.max(p_data)

    def forward(self, output: Tensor, target: Tensor):

        preacts = np.array(output.cpu())
        targets_all = np.array(one_hot_embedding(target.cpu(), self.num_classes))
        if self.empty:
            self.preacts = preacts
            self.targets_all = targets_all
            self.empty = False
        else:
            self.preacts = np.concatenate((self.preacts, preacts))
            self.targets_all = np.concatenate((self.targets_all, targets_all))

    
    def eval_func(self, x) -> float:
        return np.mean(cal_diff(self.preacts_y1, self.targets_y1, self.probs_y1, self.correcnt_portion, self.p_data, x))

    def optimize(self) -> [Tuple[float], float, float]:
        
        rare_class = []
        threshold_res = []
        self.probs = softmax(self.preacts.transpose(), T = 1).transpose()


        class_weight = 1 / self.p_data
        effectsamples = 0
        true_class_alls = np.argmax(self.targets_all, axis = 1)
        for kn in range(len(self.targets_all)):
            effectsamples = effectsamples + class_weight[true_class_alls[kn]]

        samples_threshold = effectsamples / self.th_class / 10
        # in case some noise

        for kcls in range(self.th_class):
            
            r = math.ceil(self.num_classes / self.th_class)

            targets_y1_pos = []
            clas_sample = 0
            for label in range(r * kcls, r * (kcls + 1)):
                clas_sample = clas_sample + len(np.where(true_class_alls==label)[0])

                if len(targets_y1_pos) == 0:
                    targets_y1_pos = np.where(np.argmax(self.preacts, axis = 1)==label)[0]
                else:
                    targets_y1_pos = np.concatenate((targets_y1_pos, np.where(np.argmax(self.preacts, axis = 1)==label)[0]))
            
            # select the subset of targets_y1 based on the threshold
            self.targets_y1 = self.targets_all[targets_y1_pos, :]
            self.preacts_y1 = self.preacts[targets_y1_pos, :]
            self.probs_y1 = self.probs[targets_y1_pos, :]
            
            true_class_all = np.argmax(self.targets_y1, axis = 1)
            target_class_all = np.argmax(self.preacts_y1, axis = 1)

            # acc_all = np.sum(true_class_all == target_class_all) / len(target_class_all)

            # weighted acc, 1 * N1 + 1 * N2 / (N1 + N2)
            acc_numerator = 0
            correct_y1_pos = np.where(true_class_all == target_class_all)[0]
            for cor_pos in correct_y1_pos:
                acc_numerator = acc_numerator + class_weight[true_class_all[cor_pos]]
            acc_denominator = 1e-6
            for all_pos in range(len(targets_y1_pos)):
                acc_denominator = acc_denominator + class_weight[true_class_all[all_pos]]

            acc_all = acc_numerator / acc_denominator

            if acc_denominator <= samples_threshold or clas_sample < 10:
                for label in range(r * kcls, r * (kcls + 1)):
                    if label < self.num_classes:
                        rare_class.append(label)

            if acc_all < self.correcnt_portion and acc_denominator > samples_threshold and clas_sample >= 10:

                optimization_result = scipy.optimize.minimize(
                                    fun = self.eval_func,
                                    x0 = 0.95,
                                    bounds= [(0.,1.0)] ,
                                    method='Nelder-Mead',
                                    tol=1e-07)

                threshold_res.append(optimization_result.x.item())
            else:
                threshold_res.append(0.)

            # if kcls == self.th_class-2:
            #     print(class_weight)
            #     print(f"predicted classes are {range(r * kcls, r * (kcls + 1))}")
            #     print(f"true classes are {true_class_all}")
            #     print(f"target classes are {target_class_all}")
            #     print(f"sample threshold is {samples_threshold}")
            #     print(f"Predicted class {kcls} samples: {acc_denominator}")
            #     print(f"Balanced accurary of class {kcls} is {acc_all}")
            #     acc_all_og = np.sum(true_class_all == target_class_all) / (len(target_class_all) + 1e-6)
            #     print(f"Unbalanced accurary of class {kcls} is {acc_all_og}")
            #     print(f"Learned threshold of class {kcls} is {threshold_res[kcls]}")

            # if kcls == self.th_class-1:
            #     print(f"predicted classes are {range(r * kcls, r * (kcls + 1))}")
            #     print(f"true classes are {true_class_all}")
            #     print(f"target classes are {target_class_all}")
            #     print(f"Predicted class {kcls} samples: {acc_denominator}")
            #     print(f"Balanced accurary of class {kcls} is {acc_all}")
            #     acc_all_og = np.sum(true_class_all == target_class_all) / (len(target_class_all) + 1e-6)
            #     print(f"Unbalanced accurary of class {kcls} is {acc_all_og}")
            #     print(f"Learned threshold of class {kcls} is {threshold_res[kcls]}")


        labels_trainval = np.argmax(self.preacts, axis = 1)
        cls_trainval_inds = []
        for l_index in range(self.num_classes):
            cls_trainval_ind = np.where(labels_trainval == l_index)[0]
            cls_trainval_inds.append(cls_trainval_ind)
        
        threshold_res_all = []
        for kcls in range(self.th_class):
            for _ in range(r):
                threshold_res_all.append(threshold_res[kcls])

        print(f"optimized thresholds: {threshold_res_all}")

        pd_trainval_cls_sels = get_conf_sel(self.preacts, threshold_res_all, cls_trainval_inds, self.num_classes)
        pd_trainval_cls_sels_correct = get_conf_sel_correct(self.preacts, threshold_res_all, cls_trainval_inds, self.targets_all, self.num_classes)
        
        numl = 1e-6
        numl_correct = 0
        ul_samples = []
        for l_index in range(self.num_classes):
            ul_samples.append(len(pd_trainval_cls_sels[l_index]))
            numl = numl + len(pd_trainval_cls_sels[l_index])
            numl_correct = numl_correct + len(pd_trainval_cls_sels_correct[l_index])

        effective_sample = numl
        correct_rate = numl_correct / numl

        return threshold_res_all, effective_sample, correct_rate, ul_samples, rare_class
