from yacs.config import CfgNode as CN

_C = CN()

_C.MISC = CN()
_C.MISC.LOG_CLASSWISE = True

# Model
_C.MODEL = CN()
_C.MODEL.NAME = "WRN"
_C.MODEL.WIDTH = 2
_C.MODEL.NUM_CLASSES = 10
_C.MODEL.EMA_DECAY = 0.999
_C.MODEL.EMA_WEIGHT_DECAY = 0.0
_C.MODEL.WITH_ROTATION_HEAD = False

# Distribution Alignment
_C.MODEL.DIST_ALIGN = CN()
_C.MODEL.DIST_ALIGN.APPLY = False
_C.MODEL.DIST_ALIGN.TEMPERATURE = 1.0  # default temperature for scaling the target distribution

# Feature Queue for DASO and USADTM
_C.MODEL.QUEUE = CN()
_C.MODEL.QUEUE.MAX_SIZE = 256
_C.MODEL.QUEUE.FEAT_DIM = 128


# Losses
_C.MODEL.LOSS = CN()
_C.MODEL.LOSS.LABELED_LOSS = "CrossEntropyLoss"
_C.MODEL.LOSS.WITH_LABELED_COST_SENSITIVE = False

_C.MODEL.LOSS.UNLABELED_LOSS = "MSELoss"
_C.MODEL.LOSS.UNLABELED_LOSS_WEIGHT = 1.0  #
_C.MODEL.LOSS.WITH_SUPPRESSED_CONSISTENCY = False
_C.MODEL.LOSS.WARMUP_ITERS = 200000

_C.MODEL.LOSS.COST_SENSITIVE = CN()
_C.MODEL.LOSS.COST_SENSITIVE.LOSS_OVERRIDE = ""  # default: balanced CE loss (CB, LDAM)
_C.MODEL.LOSS.COST_SENSITIVE.BETA = 0.999

# Cross Entropy
_C.MODEL.LOSS.CROSSENTROPY = CN()
_C.MODEL.LOSS.CROSSENTROPY.USE_SIGMOID = False


# Algorithm
_C.ALGORITHM = CN()
_C.ALGORITHM.NAME = "Supervised"
_C.ALGORITHM.CONFIDENCE_THRESHOLD = 0.95
_C.ALGORITHM.DOUBLE_AUG = False
_C.ALGORITHM.CONS_RAMPUP_SCHEDULE = "exp"  # "exp" or "linear"
_C.ALGORITHM.CONS_RAMPUP_ITERS_RATIO = 0.4
_C.ALGORITHM.EXP = "" # experiment name
_C.ALGORITHM.LABELED_STRONG_AUG = False

# PseudoLabel
_C.ALGORITHM.PSEUDO_LABEL = CN()

# Mean Teacher
_C.ALGORITHM.MEANTEACHER = CN()
_C.ALGORITHM.MEANTEACHER.APPLY_DASO = False

# MixMatch
_C.ALGORITHM.MIXMATCH = CN()
_C.ALGORITHM.MIXMATCH.NUM_AUG = 2
_C.ALGORITHM.MIXMATCH.TEMPERATURE = 0.5
_C.ALGORITHM.MIXMATCH.MIXUP_ALPHA = 0.75
_C.ALGORITHM.MIXMATCH.APPLY_DASO = False

# ReMixMatch
_C.ALGORITHM.REMIXMATCH = CN()
_C.ALGORITHM.REMIXMATCH.NUM_AUG = 2
_C.ALGORITHM.REMIXMATCH.TEMPERATURE = 0.5
_C.ALGORITHM.REMIXMATCH.MIXUP_ALPHA = 0.75
_C.ALGORITHM.REMIXMATCH.WEIGHT_KL = 1.0
_C.ALGORITHM.REMIXMATCH.WEIGHT_ROT = 1.0
_C.ALGORITHM.REMIXMATCH.WITH_DISTRIBUTION_MATCHING = True
_C.ALGORITHM.REMIXMATCH.LABELED_STRONG_AUG = False

# FixMatch
_C.ALGORITHM.FIXMATCH = CN()

# DASO
_C.ALGORITHM.DASO = CN()
_C.ALGORITHM.DASO.APPLY = False
_C.ALGORITHM.DASO.PRETRAIN_STEPS = 5000
_C.ALGORITHM.DASO.PROTO_TEMP = 0.05
_C.ALGORITHM.DASO.PL_DIST_UPDATE_PERIOD = 100

# pseudo-label mixup
_C.ALGORITHM.DASO.WITH_DIST_AWARE = True
_C.ALGORITHM.DASO.DIST_TEMP = 1.0
_C.ALGORITHM.DASO.INTERP_ALPHA = 0.5

# prototype option
_C.ALGORITHM.DASO.QUEUE_SIZE = 256

# Semantic Alignment loss
_C.ALGORITHM.DASO.PSA_LOSS_WEIGHT = 1.0

# CReST
_C.ALGORITHM.CREST = CN()
_C.ALGORITHM.CREST.GEN_PERIOD_STEPS = 50000  # 5gens x 50000 = total 250k steps
_C.ALGORITHM.CREST.ALPHA = 3.0
_C.ALGORITHM.CREST.TMIN = 0.5
_C.ALGORITHM.CREST.PROGRESSIVE_ALIGN = False

# ACR
_C.ALGORITHM.ACR = CN()
# hyperparameter taken from https://github.com/Gank0078/ACR/blob/main/train.py
_C.ALGORITHM.ACR.WARMUP_RATIO = 0.01
_C.ALGORITHM.ACR.TAU1 = 2
_C.ALGORITHM.ACR.TAU12 = 2
_C.ALGORITHM.ACR.TAU2 = 2
_C.ALGORITHM.ACR.EMA_DECAY = 0.9

# ABC
_C.ALGORITHM.ABC = CN()
_C.ALGORITHM.ABC.APPLY = False
_C.ALGORITHM.ABC.DASO_PSEUDO_LABEL = True

# USADTM
# Unsupervised Semantic Aggregation and Deformable Template Matching for Semi-Supervised Learning  # noqa
_C.ALGORITHM.USADTM = CN()
_C.ALGORITHM.USADTM.PRETRAIN_STEPS = 500        # for pseudo-labeling
_C.ALGORITHM.USADTM.WARMUP_CLUSTER_LOSS = 5000  # for clustering objective
_C.ALGORITHM.USADTM.DTM_THRES = 0.85
_C.ALGORITHM.USADTM.UC_LOSS_WEIGHT = 0.1


# cRT
_C.ALGORITHM.CRT = CN()
_C.ALGORITHM.CRT.TARGET_DIR = ""

# Logit Adjustment
_C.ALGORITHM.LOGIT_ADJUST = CN()
_C.ALGORITHM.LOGIT_ADJUST.APPLY = False
_C.ALGORITHM.LOGIT_ADJUST.TAU = 1.0

# DARP
_C.ALGORITHM.DARP = CN()
_C.ALGORITHM.DARP.APPLY = False
_C.ALGORITHM.DARP.WARMUP_RATIO = 0.4
_C.ALGORITHM.DARP.PER_ITERS = 10
_C.ALGORITHM.DARP.EST = "darp_estim"
_C.ALGORITHM.DARP.ALPHA = 2.0
_C.ALGORITHM.DARP.NUM_DARP_ITERS = 10

_C.ALGORITHM.DARP_ESTIM = CN()
_C.ALGORITHM.DARP_ESTIM.PER_CLASS_VALID_SAMPLES = 10
_C.ALGORITHM.DARP_ESTIM.THRESH_COND = 100

# SEVAL
_C.ALGORITHM.SEVAL = CN()
_C.ALGORITHM.SEVAL.PI_EMA_DECAY = 0.999
_C.ALGORITHM.SEVAL.TH_EMA_DECAY = 0.999
_C.ALGORITHM.SEVAL.OPT_FUNC = "nll" # nll or acc
_C.ALGORITHM.SEVAL.TH_START = 0.9
_C.ALGORITHM.SEVAL.TH_END = 0.9
_C.ALGORITHM.SEVAL.OPT_PI = True
_C.ALGORITHM.SEVAL.OPT_TH = True
_C.ALGORITHM.SEVAL.PI_CLS = -1
_C.ALGORITHM.SEVAL.TH_CLS = -1
_C.ALGORITHM.SEVAL.PI_LOW = 1e-3
_C.ALGORITHM.SEVAL.PI_UP = 3.
_C.ALGORITHM.SEVAL.ALPHA_LOW = 1.
_C.ALGORITHM.SEVAL.ALPHA_UP = 1.
_C.ALGORITHM.SEVAL.PI_WARMUP_RATIO = 0.
_C.ALGORITHM.SEVAL.PI_OPTIMITZER = "Nelder-Mead"  # Nelder-Mead (quick) or Powell (better) etc.

_C.ALGORITHM.SEVAL.EST = "seval_estim" # save folder
_C.ALGORITHM.SEVAL.LOAD_PARAM = False # load paramters from folders

_C.ALGORITHM.SEVAL.ESTIM = CN() # estimate the paramters from training split
_C.ALGORITHM.SEVAL.ESTIM.APPLY = False
_C.ALGORITHM.SEVAL.ESTIM.PORTION = 0.5 # validation portion

# dataset
_C.DATASET = CN()
_C.DATASET.BUILDER = "build_cifar10_dataset"
_C.DATASET.NAME = "cifar10"
_C.DATASET.ROOT = "./data"
_C.DATASET.SAMPLER_NAME = "RandomSampler"
_C.DATASET.SAMPLER_BETA = 0.999
_C.DATASET.RESOLUTION = 32

_C.DATASET.NUM_VALID = 5000
_C.DATASET.NUM_WORKERS = 8
_C.DATASET.REVERSE_UL_DISTRIBUTION = False

_C.DATASET.CIFAR10 = CN()
_C.DATASET.CIFAR10.NUM_LABELED_HEAD = 1500
_C.DATASET.CIFAR10.IMB_FACTOR_L = 100
_C.DATASET.CIFAR10.NUM_UNLABELED_HEAD = 3000
_C.DATASET.CIFAR10.IMB_FACTOR_UL = 100

_C.DATASET.CIFAR100 = CN()
_C.DATASET.CIFAR100.NUM_LABELED_HEAD = 150
_C.DATASET.CIFAR100.IMB_FACTOR_L = 10
_C.DATASET.CIFAR100.NUM_UNLABELED_HEAD = 300
_C.DATASET.CIFAR100.IMB_FACTOR_UL = 10

_C.DATASET.STL10 = CN()
_C.DATASET.STL10.NUM_LABELED_HEAD = 450
_C.DATASET.STL10.IMB_FACTOR_L = 10
_C.DATASET.STL10.NUM_UNLABELED_HEAD = -1  # unknown unlabeled data
_C.DATASET.STL10.IMB_FACTOR_UL = -1  # # unknown unlabeled data

_C.DATASET.AVES = CN()
_C.DATASET.AVES.NUM_LABELED_HEAD = -1 # do not need to specify
_C.DATASET.AVES.IMB_FACTOR_L = -1
_C.DATASET.AVES.NUM_UNLABELED_HEAD = -1
_C.DATASET.AVES.IMB_FACTOR_UL = -1 
_C.DATASET.AVES.UL_SOURCE = "in" # "in" or "all"

_C.DATASET.IMAGENET = CN()
_C.DATASET.IMAGENET.NUM_LABELED_HEAD = -1 # do not need to specify
_C.DATASET.IMAGENET.IMB_FACTOR_L = -1
_C.DATASET.IMAGENET.NUM_UNLABELED_HEAD = -1
_C.DATASET.IMAGENET.IMB_FACTOR_UL = -1 

# transform parameters
_C.DATASET.TRANSFORM = CN()
_C.DATASET.TRANSFORM.STRONG_AUG = False

# solver
_C.SOLVER = CN()
_C.SOLVER.IMS_PER_BATCH = 64           # Batch size for labeled data
_C.SOLVER.UNLABELED_BATCH_RATIO = 1    # =unlabeled_batch_size / labeled_batch_size
_C.SOLVER.MAX_ITER = 250000
_C.SOLVER.BASE_LR = 0.03
_C.SOLVER.OPTIM_NAME = "SGD"  # "SGD" or "Adam"
_C.SOLVER.APPLY_SCHEDULER = False

# SGD params
_C.SOLVER.SGD = CN()
_C.SOLVER.SGD.MOMENTUM = 0.9
_C.SOLVER.SGD.WEIGHT_DECAY = 0.0005
_C.SOLVER.SGD.NESTEROV = True

# Adam params
_C.SOLVER.ADAM = CN()
_C.SOLVER.ADAM.BETA1 = 0.9
_C.SOLVER.ADAM.BETA2 = 0.999
_C.SOLVER.ADAM.EPS = 1e-08
_C.SOLVER.ADAM.WEIGHT_DECAY = 0.0

# scheduler setup
_C.SOLVER.LR_SCHEDULER_NAME = "WarmupCosineLRFixMatch"  # or WarmupMultiStepLR
_C.SOLVER.COS_LR_RATIO = 7  # for WarmupCosineLRFixMatch, default: 7
_C.SOLVER.STEPS = (400000,)
_C.SOLVER.GAMMA = 0.2   # lr decay factor
_C.SOLVER.RAMPDOWN_ITERS = 0
_C.SOLVER.WARMUP_FACTOR = 0
_C.SOLVER.WARMUP_ITERS = 0
_C.SOLVER.WARMUP_METHOD = "linear"

# Periodical params
_C.PERIODS = CN()
_C.PERIODS.EVAL = 500
_C.PERIODS.CHECKPOINT = 5000
_C.PERIODS.LOG = 500
_C.PERIODS.RECORD = False


_C.OUTPUT_DIR = "outputs"
_C.RESUME = ""
_C.EVAL_ON_TEST_SET = True
_C.GPU_ID = 0
_C.MEMO = ""

# Reproducability
_C.SEED = -1
_C.CUDNN_DETERMINISTIC = True
_C.CUDNN_BENCHMARK = False
