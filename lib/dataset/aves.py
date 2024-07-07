import logging

import os
import numpy as np
import torchvision
from yacs.config import CfgNode
from torchvision.datasets import folder as dataset_parser
from .transform import build_transforms
from torch.utils.data import Dataset

def build_aves_dataset(cfg: CfgNode) -> tuple():
    # fmt: off
    root = cfg.DATASET.ROOT
    algorithm = cfg.ALGORITHM.NAME
    seed = cfg.SEED
    # fmt: on

    # NOTE: it seems that there are some oracle labels for the unlabeled data, but here we do not utilize these
    # the labels should all be -1
    if cfg.DATASET.AVES.UL_SOURCE == "in":
        ul_train_split = "u_train_in"
    else:
        ul_train_split = "u_train"

    logger = logging.getLogger()

    l_trans, ul_trans, eval_trans = build_transforms(cfg, "aves")

    # dataloader is taken from https://github.com/microsoft/Semi-supervised-learning/blob/main/semilearn/datasets/cv_datasets/aves.py
    # NOTE this dataset is inherently imbalanced with unknown distribution
    
    # labeledd data
    l_samples_train, l_num_classes, l_targets_train = make_dataset(root, 'l_train_val')

    aves_valid = None
    # note for aves, we do not leave data for validation, there utilize traning samples as valiation
    # this is just an 'indicator', but not that meaningful
    aves_valid = iNatDataset(l_samples_train, l_targets_train, transform=eval_trans)

    # split the validation data from training for seval estimation
    if cfg.ALGORITHM.SEVAL.ESTIM.APPLY:
        num_holdout_portion = cfg.ALGORITHM.SEVAL.ESTIM.PORTION
        l_samples_train, l_targets_train, l_samples_val, l_targets_val = \
        split_val_from_train_portion(l_num_classes, l_samples_train, l_targets_train, num_holdout_portion, seed)

        aves_valid = iNatDataset(l_samples_val, l_targets_val, transform=eval_trans)

    l_train = iNatDataset(l_samples_train, l_targets_train, transform=l_trans)
    # unlabeled data
    if algorithm == "Supervised":
        ul_train = None
    else:
        ul_samples_train, ul_num_classes, ul_targets_train = make_dataset(root, ul_train_split)
        ul_train = iNatDataset(ul_samples_train, ul_targets_train, transform=ul_trans)

    # test data
    l_samples_test, _, l_targets_test = make_dataset(root, 'test')
    aves_test = iNatDataset(l_samples_test, l_targets_test, transform=eval_trans)

    # mapping the labels so that the class frequency decreases along the idx
    # just make it easier for later use
    class_idx = dict()
    num_samples_per_class = l_train.get_num_per_cls()
    for i in range(len(num_samples_per_class)):
        class_idx[num_samples_per_class[i][0]] = i
    l_train.map_idx(class_idx)
    aves_valid.map_idx(class_idx)
    aves_test.map_idx(class_idx)
    if ul_train != None:
        ul_train.map_idx(class_idx)

    # whether to shuffle the class order
    assert l_num_classes == cfg.MODEL.NUM_CLASSES

    num_samples_per_class = l_train.get_num_per_cls()
    logger.info("class distribution of labeled dataset")
    logger.info(
        ", ".join("idx{}: {}".format(item[0], item[1]) for item in num_samples_per_class)
    )
    logger.info(
        "=> number of labeled data: {}\n".format(
            sum([item[1] for item in num_samples_per_class])
        )
    )
    if ul_train is not None:
        logger.info("Number of unlabeled data: {}\n".format(len(ul_train)))

    return l_train, ul_train, aves_valid, aves_test


class iNatDataset(Dataset):
    def __init__(self, samples, targets, transform=None):

        self.samples = samples
        self.targets = targets
        self.transforms = transform
        self.is_ul_unknown = False
        self.loader = dataset_parser.default_loader

        self.map_cls = False
        self.data = []
        for i in range(len(self.samples)):
            self.data.append(self.samples[i][0])
        
        self.get_num_per_cls()

    def __sample__(self, idx):
        path, target = self.samples[idx]
        
        if self.map_cls:
            if target == -1:
                target_new = target
            else:
                target_new = self.class_idx[target]
        else:
            target_new = target
        img = self.loader(path)
        return img, target_new
    
    def get_num_per_cls(self):
        num_classes = len(np.unique(self.targets))
        classwise_num_samples = dict()
        for i in range(num_classes):
            classwise_num_samples[i] = len(np.where(np.array(self.targets) == i)[0])
        num_data_per_cls = sorted(classwise_num_samples.items(), key=(lambda x: x[1]), reverse=True)

        self.num_samples_per_class = num_data_per_cls

        return num_data_per_cls
    
    def map_idx(self, class_idx):
        targets_new = []
        for target_old in self.targets:
            if target_old == -1:
                targets_new.append(target_old)
            else:
                targets_new.append(class_idx[target_old])
        self.targets = targets_new
        self.class_idx = class_idx
        self.map_cls = True

    def __getitem__(self, idx):
        """
        _labels would be -1 if unlabeled sample is unknown
        """
        img, label = self.__sample__(idx)

        if self.transforms is not None:
            img = self.transforms(img)

        return img, label, idx
    
    def select_dataset(self, indices=None, labels=None, return_transforms=False):

        # resample for crest, a little crumpy but fit the numpy database
        # it will load all the training data in the memory!
        if indices is None:
            indices = list(range(len(self)))
        imgs = []
        for idx in indices:
            sample_cur = self.samples[idx].copy()
            if sample_cur[1] != -1:
                sample_cur[1] = self.class_idx[sample_cur[1]]
            imgs.append(sample_cur)

        if labels is not None:
            # override specified labels (i.e., pseudo-labels)
            for idx in range(len(imgs)):
                imgs[idx][1] = labels[idx]

        classes = [x[1] for x in imgs]
        if return_transforms:
            return imgs, classes, self.transforms    
        return imgs, classes


    def __len__(self):
        return len(self.data)


def make_dataset(dataset_root, split):
    split_file_path = os.path.join(dataset_root, split + '.txt')

    with open(split_file_path, 'r') as f:
        img = f.readlines()

    img = [x.strip('\n').rsplit() for x in img]

    for idx, x in enumerate(img):
        img[idx][0] = os.path.join(dataset_root, x[0])
        img[idx][1] = int(x[1])

    classes = [x[1] for x in img]
    num_classes = len(set(classes))
    print('# images in {}: {}'.format(split, len(img)))
    return img, num_classes, classes

def split_val_from_train_portion(num_classes, l_samples_train_all, l_targets_train_all, num_valid_portion, seed):
    
    # set random state
    rng = np.random.RandomState(seed)

    train_inds = []
    val_inds = []
    for i in range(num_classes):
        cls_inds = np.where(np.array(l_targets_train_all) == i)[0]
        rng.shuffle(cls_inds)
        num_valid_cls = int(len(cls_inds) * num_valid_portion)
        # disjoint
        train_inds.extend(cls_inds[num_valid_cls:])
        val_inds.extend(cls_inds[:num_valid_cls])

    l_samples_train = []
    l_targets_train = []
    l_samples_val = []
    l_targets_val = []
    for train_ind in train_inds:
        l_samples_train.append(l_samples_train_all[train_ind])
        l_targets_train.append(l_targets_train_all[train_ind])

    for val_ind in val_inds:
        l_samples_val.append(l_samples_train_all[val_ind])
        l_targets_val.append(l_targets_train_all[val_ind])

    return l_samples_train, l_targets_train, l_samples_val, l_targets_val
    