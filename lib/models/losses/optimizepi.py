import torch
import torch.nn as nn
from torch import Tensor
import numpy as np
import scipy

from lib.utils import Meters
from typing import Tuple, Union

def softmax(x, T, b=0):
    x = x / T + b
    f = np.exp(x - np.max(x, axis = 0))  # shift values
    return f / f.sum(axis = 0)

def cal_acc(preacts: Tensor, targets_all: Tensor, alpha: Tuple[float], pi: Tuple[float]) -> Tuple[float]:
    # -> preacts.     N x C
    # -> targets_all. N x C
    # -> alpha        C
    # -> pi           C
    acc_cls = []
    preacts_seval = alpha * (preacts - np.log(pi))
    for kcls in range(len(pi)):
        label = kcls
        targets_y1 = np.where(np.argmax(targets_all, axis = 1)==label)[0]

        if len(targets_y1) > 0:

            pred_class = np.argmax(preacts_seval, axis = 1)[targets_y1]
            target_class = np.argmax(targets_all, axis = 1)[targets_y1]

            acc = np.sum(pred_class == target_class) / len(target_class)

            acc_cls.append(acc)
        else:
            acc_cls.append(0.)
    
    return acc_cls

def cal_ll(preacts: Tensor, targets_all: Tensor, alpha: Tuple[float], pi: Tuple[float]) -> Tuple[float]:
    # -> preacts.     N x C
    # -> targets_all. N x C
    # -> pi           C
    # -> alpha        C
    nll_cls = []
    preacts_seval = alpha * (preacts - np.log(pi))
    probs = softmax(preacts_seval.transpose(), T = 1).transpose()
    labels = np.argmax(targets_all, axis = 1)
    for kcls in range(len(pi)):

        cls_inds = np.where(labels == kcls)[0]

        if len(cls_inds) > 0:
            probs_cls = probs[cls_inds, :]
            targets_cls = targets_all[cls_inds, :]

            res = (np.log(probs_cls)*targets_cls).mean()

            nll_cls.append(res)
        else:
            # assuming the probability is 0.01
            nll_cls.append(-2.)
    
    return nll_cls

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

class OptimizePi(nn.Module):

    def __init__(self, num_classes: int, pi_class: int) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.pi_class = pi_class
        self.empty = True
        self.preacts = None
        # N x C
        self.targets_all = None
        # N x C

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

    def eval_acc(self, x) -> float:
        repeat_times = int(self.num_classes / self.pi_class)
        alpha = np.array(x[:self.pi_class])
        alpha = alpha.repeat(repeat_times)
        pi = np.array(x[self.pi_class:])
        pi = pi.repeat(repeat_times)
        return -np.mean(cal_acc(self.preacts, self.targets_all, alpha, pi))
    
    def eval_nll(self, x) -> float:
        repeat_times = int(self.num_classes / self.pi_class)
        alpha = np.array(x[:self.pi_class])
        alpha = alpha.repeat(repeat_times)
        pi = np.array(x[self.pi_class:])
        pi = pi.repeat(repeat_times)
        return -np.mean(cal_ll(self.preacts, self.targets_all, alpha, pi))

    def optimize(self, metric="nll", bounds=[1., 1., 1., 1.], optimizer='Nelder-Mead') -> [Tuple[float], float, float]:

        if metric == "acc":
            optimization_result = scipy.optimize.minimize(
                        fun=self.eval_acc,
                        x0=np.array([1.0 for x in range(self.pi_class)]
                                  +[1.0 for x in range(self.pi_class)]),
                        bounds=[(bounds[0], bounds[1]) for x in range(self.pi_class)]
                              +[(bounds[2], bounds[3]) for x in range(self.pi_class)],
                        method=optimizer,
                        tol=1e-07)
        elif metric == "nll":
            optimization_result = scipy.optimize.minimize(
                        fun=self.eval_nll,
                        x0=np.array([1.0 for x in range(self.pi_class)]
                                  +[1.0 for x in range(self.pi_class)]),
                        bounds=[(bounds[0], bounds[1]) for x in range(self.pi_class)]
                              +[(bounds[2], bounds[3]) for x in range(self.pi_class)],
                        method=optimizer,
                        tol=1e-07)
        else:
            raise NotImplementedError()

        repeat_times = int(self.num_classes / self.pi_class)
        optimized_alpha = optimization_result.x[:self.pi_class].repeat(repeat_times)
        optimized_pi = optimization_result.x[self.pi_class:].repeat(repeat_times)
        
        print(f"optimized alpha: {optimized_alpha}")
        print(f"optimized pi: {optimized_pi}")


        acc_post = cal_acc(self.preacts, self.targets_all, 
                                  optimized_alpha, optimized_pi)
        acc_pre = cal_acc(self.preacts, self.targets_all, 
                                   np.ones(self.num_classes), 
                                   np.ones(self.num_classes))
        
        print(f"after refinement, accuary raise from {np.mean(acc_pre)} to {np.mean(acc_post)}")
        
        optimization_res = np.concatenate((optimized_alpha, optimized_pi))

        return optimization_res, acc_pre, acc_post
