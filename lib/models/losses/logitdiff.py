import torch
import torch.nn as nn
from torch import Tensor

from lib.utils import Meters
from typing import Tuple, Union


def logitdiff(output: Tensor, target: int) -> int:
    """
    Computes the logit difference between the ground truth logit and the maximum of the rest
    
    output: tensor, N (batch number) * C (class number)
    target: int, the chosen class

    return: res,: int, the logit distance

    """
    with torch.no_grad():

        logit_y = output[:, target]
        cls_list_ey = list(range(output.shape[1]))
        cls_list_ey.remove(target)
        logit_exclude_y = output[:, cls_list_ey]
        max_logit_ey, _ = logit_exclude_y.max(axis = 1)

        res = (logit_y - max_logit_ey).mean().item()
        
        return res

class Logitdiff(nn.Module):

    def __init__(self, num_classes: int):
        super().__init__()
        self.num_classes = num_classes
        self.classwise_diff = Meters()
        self.classwise_meanprob = Meters()

    def forward(self, output: Tensor, target: Tensor, prefix=""):

        # log classwise logitdiff
        for class_idx in range(self.num_classes):
            metric_key = f"{class_idx}"
            if prefix:
                metric_key = f"{prefix}_{metric_key}"
            cls_inds = torch.where(target == class_idx)[0]
            if len(cls_inds):
                cls_logitdiff = logitdiff(output[cls_inds], class_idx)
                self.classwise_diff.put_scalar(metric_key, cls_logitdiff, n=len(cls_inds))
            #
            prob = output.detach().softmax(dim=1)
            pred = torch.max(prob, dim = 1)[1]
            cls_inds_pred = torch.where(pred == class_idx)[0]
            if len(cls_inds_pred):
                
                prob_sel = prob[cls_inds_pred]
                maxprob = prob_sel.max(axis = 1)[0].mean().item()

                self.classwise_meanprob.put_scalar(metric_key, maxprob, n=len(cls_inds_pred))


    @property
    def classwise(self) -> dict:
        return self.classwise_diff.get_latest_scalars_with_avg()
    
    @property
    def classwise_prob(self) -> dict:
        return self.classwise_meanprob.get_latest_scalars_with_avg()
