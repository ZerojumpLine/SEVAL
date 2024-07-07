import logging

import torchvision
from yacs.config import CfgNode
from PIL import Image
import os
import pickle
import sys
import numpy as np
from .transform import build_transforms
from torch.utils.data import Dataset

def build_imagenet_dataset(cfg: CfgNode) -> tuple():
    # fmt: off
    root = cfg.DATASET.ROOT
    algorithm = cfg.ALGORITHM.NAME

    num_classes = cfg.MODEL.NUM_CLASSES
    seed = cfg.SEED
    # fmt: on

    logger = logging.getLogger()

    l_trans, ul_trans, eval_trans = build_transforms(cfg, "cifar10")

    # split the validation data from training for seval estimation
    train_list = ['train_data_batch_1']
    if cfg.ALGORITHM.SEVAL.ESTIM.APPLY:
        num_holdout_portion = cfg.ALGORITHM.SEVAL.ESTIM.PORTION

        l_train = SmallImageNet(file_path = root, imgsize = 32, train = True, train_list = train_list, trainval = 'train', transforms=l_trans, seed=seed)
        imagenet_valid = SmallImageNet(file_path = root, imgsize = 32, train = True, train_list = train_list, trainval = 'val', transforms=eval_trans, seed=seed)

    else:
        l_train = SmallImageNet(file_path = root, imgsize = 32, train = True, train_list = train_list, transforms=l_trans)
        # note for imagenet, we do not leave data for validation, there utilize traning samples as valiation
        # this is just an 'indicator', but not that meaningful
        imagenet_valid = SmallImageNet(file_path = root, imgsize = 32, train = True, train_list = train_list, transforms=eval_trans)

    # unlabeled data
    if algorithm == "Supervised":
        ul_train = None
    else:
        ul_list = ['train_data_batch_2', 'train_data_batch_3', 'train_data_batch_4', 'train_data_batch_5',
                  'train_data_batch_6', 'train_data_batch_7', 'train_data_batch_8', 'train_data_batch_9', 'train_data_batch_10']
        ul_train = SmallImageNet(file_path = root, imgsize = 32, train = True, train_list = ul_list, transforms=ul_trans)

    imagenet_test = SmallImageNet(file_path = root, imgsize = 32, train = False, train_list = None, transforms=eval_trans)


    # mapping the labels so that the class frequency decreases along the idx
    # just make it easier for later use
    class_idx = dict()
    num_samples_per_class = l_train.get_num_per_cls()
    for i in range(len(num_samples_per_class)):
        class_idx[num_samples_per_class[i][0]] = i
    l_train.map_idx(class_idx)
    imagenet_valid.map_idx(class_idx)
    imagenet_test.map_idx(class_idx)
    if ul_train != None:
        ul_train.map_idx(class_idx)

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
        logger.info("class distribution of unlabeled dataset")

    return l_train, ul_train, imagenet_valid, imagenet_test


class SmallImageNet(Dataset):
    test_list = ['val_data']

    def __init__(self, file_path, imgsize, train, train_list, trainval=None, transforms=None, seed=None):
        # assert imgsize == 32 or imgsize == 64, 'imgsize should only be 32 or 64'
        self.imgsize = imgsize
        self.train = train
        self.transforms = transforms
        self.is_ul_unknown = False
        self.data = []
        self.targets = []
        self.map_cls = False
        if self.train:
            downloaded_list = train_list
        else:
            downloaded_list = self.test_list

        # now load the picked numpy arrays
        for filename in downloaded_list:
            file = os.path.join(file_path, filename)
            with open(file, 'rb') as f:
                if sys.version_info[0] == 2:
                    entry = pickle.load(f)
                else:
                    entry = pickle.load(f, encoding='latin1')
                self.data.append(entry['data'])
                self.targets.extend(entry['labels'])  # Labels are indexed from 1
        self.targets = [i - 1 for i in self.targets]
        self.data = np.vstack(self.data).reshape((len(self.targets), 3, self.imgsize, self.imgsize))
        self.data = self.data.transpose((0, 2, 3, 1))  # convert to HWC

        # split the portion
        rng = np.random.RandomState(seed)
        
        sellist = list(range(len(self.targets)))
        rng.shuffle(sellist)
        dataset = self.data[sellist, :, :, :]
        targets = [self.targets[i] for i in sellist]

        trainlist = []
        vallist = []
        # return valist such that 10 samples per class.
        for kcls in range(127):
            kcnt = 0
            for index, value in enumerate(targets):
                if value == kcls:
                    if kcnt < 10:
                        vallist.append(index)
                        kcnt = kcnt + 1
                    else:
                        trainlist.append(index)

        if trainval == 'train':
            self.targets = [targets[i] for i in trainlist]
            self.data = dataset[trainlist, :, :, :]
        elif trainval == 'val':
            self.targets = [targets[i] for i in vallist]
            self.data = dataset[vallist, :, :, :]
    
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
    
    def get_num_per_cls(self):
        num_classes = len(np.unique(self.targets))
        classwise_num_samples = dict()
        for i in range(num_classes):
            classwise_num_samples[i] = len(np.where(np.array(self.targets) == i)[0])
        num_data_per_cls = sorted(classwise_num_samples.items(), key=(lambda x: x[1]), reverse=True)

        self.num_samples_per_class = num_data_per_cls

        return num_data_per_cls

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transforms is not None:
            img = self.transforms(img)

        if self.map_cls:
            if target == -1:
                target_new = target
            else:
                target_new = self.class_idx[target]
        else:
            target_new = target

        return img, target_new, index
    
    def select_dataset(self, indices=None, labels=None, return_transforms=False):

        # resample for crest, a little crumpy but fit the numpy database
        # it will load all the training data in the memory!
        if indices is None:
            indices = list(range(len(self)))
        imgs = []
        for idx in indices:
            sample_cur = [self.data[idx].copy(), self.targets[idx]]
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
    
