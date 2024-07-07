import torch.nn as nn
from yacs.config import CfgNode
import torchvision
from torchvision.models import resnet50, ResNet50_Weights

class ResNet50(nn.Module):
    def __init__(self, num_classes: int = 10, pretrain: bool = True) -> None:
        super().__init__()
        self.num_classes = num_classes
        if pretrain:
            weights = ResNet50_Weights.DEFAULT
            resnet = resnet50(weights=weights)
        else:
            resnet = resnet50()
        features = []
        for bottleneck in list(resnet.children()):
            if not isinstance(bottleneck, nn.Linear):
                features.append(bottleneck)
        self.features = nn.ModuleList(features)
        self.out_features = bottleneck.in_features
    
    def forward(self, x):
        out = x
        for module in self.features:
            out = module(out)
        return out[:, :, 0, 0]

def build_resnet(cfg: CfgNode, pretrain = True) -> nn.Module:
    # fmt: off
    num_classes = cfg.MODEL.NUM_CLASSES
    # fmt: on
    return ResNet50(num_classes=num_classes, pretrain=pretrain)