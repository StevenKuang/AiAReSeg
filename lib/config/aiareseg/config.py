import yaml
from easydict import EasyDict as edict
import os
# Add default config

cfg = edict()

# MODEL
cfg.MODEL = edict()
cfg.MODEL.HEAD_TYPE = 'CORNER'
cfg.MODEL.HIDDEN_DIM = 256
cfg.MODEL.NUM_OBJECT_QUERIES = 1
cfg.MODEL.POSITION_EMBEDDING = 'sine'  # Sine or learned
cfg.MODEL.PREDICT_MASK = False
# MODEL.BACKBONE
cfg.MODEL.BACKBONE = edict()
cfg.MODEL.BACKBONE.TYPE = 'resnet50'  # ResNet50, ResNeXt101_32x8d
cfg.MODEL.BACKBONE.OUTPUT_LAYERS = ['layer3']
cfg.MODEL.BACKBONE.DILATION = False
# MODEL.TRANSFORMER
cfg.MODEL.TRANSFORMER = edict()
cfg.MODEL.TRANSFORMER.NHEADS = 8
cfg.MODEL.TRANSFORMER.DROPOUT = 0.1
cfg.MODEL.TRANSFORMER.DIM_FEEDFORWARD = 2048
cfg.MODEL.TRANSFORMER.ENC_LAYERS = 6
cfg.MODEL.TRANSFORMER.DEC_LAYERS = 6
cfg.MODEL.TRANSFORMER.PRE_NORM = False
cfg.MODEL.TRANSFORMER.DIVIDE_NORM = False
# MODEL.AIA
cfg.MODEL.AIA = edict()
cfg.MODEL.AIA.USE_AIA = True
cfg.MODEL.AIA.MATCH_DIM = 64
cfg.MODEL.AIA.FEAT_SIZE = 400

# TRAIN
cfg.TRAIN = edict()
cfg.TRAIN.RESUME = False
cfg.TRAIN.LR = 0.0001           # 0.0001
cfg.TRAIN.WEIGHT_DECAY = 0.0001
cfg.TRAIN.EPOCH = 500
cfg.TRAIN.LR_DROP_EPOCH = 60 # 400
cfg.TRAIN.BATCH_SIZE = 20
cfg.TRAIN.NUM_WORKER = 0
cfg.TRAIN.OPTIMIZER = 'ADAMW'
cfg.TRAIN.BACKBONE_MULTIPLIER = 0.1     # if > 0 then backbones are trained
cfg.TRAIN.GIOU_WEIGHT = 2.0
cfg.TRAIN.L1_WEIGHT = 5.0
cfg.TRAIN.IOU_WEIGHT = 2.0
cfg.TRAIN.IOU_MASK_WEIGHT = 5.0
cfg.TRAIN.BCE_MASK_WEIGHT = 2.0
cfg.TRAIN.MSE_MASK_WEIGHT = 2.0
cfg.TRAIN.USE_RECONSTRUCTION = False
cfg.TRAIN.L1_RECONSTRUCTION_WEIGHT = 0.6
cfg.TRAIN.MSE_RECONSTRUCTION_WEIGHT = 0.4
cfg.TRAIN.RECONSTRUCTION_WEIGHT = 10.0
cfg.TRAIN.DEEP_SUPERVISION = False
cfg.TRAIN.FREEZE_BACKBONE_BN = True
cfg.TRAIN.FREEZE_LAYERS = ['conv1', 'layer1']
cfg.TRAIN.PRINT_INTERVAL = 5    # 50
cfg.TRAIN.VAL_EPOCH_INTERVAL = 20
cfg.TRAIN.GRAD_CLIP_NORM = 0.1
# TRAIN.SCHEDULER
cfg.TRAIN.SCHEDULER = edict()
cfg.TRAIN.SCHEDULER.TYPE = 'step'
cfg.TRAIN.SCHEDULER.DECAY_RATE = 0.1
# TRAIN.SCHEDULER.COSINE
cfg.TRAIN.SCHEDULER.MIN_LR = 0.00001
cfg.TRAIN.SCHEDULER.T_MULT = 2
cfg.TRAIN.SCHEDULER.T0_EPOCH = 5


# DATA
cfg.DATA = edict()
cfg.DATA.MEAN = [0.485, 0.456, 0.406]
cfg.DATA.STD = [0.229, 0.224, 0.225]
cfg.DATA.MAX_SAMPLE_INTERVAL = 200
# DATA.TRAIN
cfg.DATA.TRAIN = edict()
#cfg.DATA.TRAIN.DATASETS_NAME = ['LASOT', 'GOT10K_vot_train']
cfg.DATA.TRAIN.DATASETS_NAME = ['LASOT']
cfg.DATA.TRAIN.DATASETS_RATIO = [1]
cfg.DATA.TRAIN.SAMPLE_PER_EPOCH = 5000 # Original 60000
# DATA.SEARCH
cfg.DATA.SEARCH = edict()
cfg.DATA.SEARCH.SIZE = 320
cfg.DATA.SEARCH.FACTOR = 5.0
cfg.DATA.SEARCH.CENTER_JITTER = 4.5
cfg.DATA.SEARCH.SCALE_JITTER = 0.5
# DATA.TEMPLATE
cfg.DATA.TEMPLATE = edict()
cfg.DATA.TEMPLATE.CENTER_JITTER = 0
cfg.DATA.TEMPLATE.SCALE_JITTER = 0

# TEST
cfg.TEST = edict()
cfg.TEST.SEARCH_FACTOR = 5.0
cfg.TEST.SEARCH_SIZE = 320
cfg.TEST.EPOCH = 600
cfg.TEST.HYPER = edict()
cfg.TEST.HYPER.DEFAULT = [100, 3, 0.7]
cfg.TEST.HYPER.LASOT = [100, 4, 0.8]
cfg.TEST.HYPER.LASOT_EXT = [100, 6, 0.8]
cfg.TEST.HYPER.TRACKINGNET = [100, 6, 0.7]
# cfg.TEST.HYPER.GOT10K_TEST = [80, 4, 0.7]
cfg.TEST.HYPER.NFS = [80, 3, 0.6]
cfg.TEST.HYPER.OTB = [100, 3, 0.7]
cfg.TEST.HYPER.UAV = [100, 3, 0.7]


def _edict2dict(dest_dict, src_edict):
    if isinstance(dest_dict, dict) and isinstance(src_edict, dict):
        for k, v in src_edict.items():
            if not isinstance(v, edict):
                dest_dict[k] = v
            else:
                dest_dict[k] = {}
                _edict2dict(dest_dict[k], v)
    else:
        return


def gen_config(config_file):
    cfg_dict = {}
    _edict2dict(cfg_dict, cfg)
    with open(config_file, 'w') as f:
        yaml.dump(cfg_dict, f, default_flow_style=False)


def _update_config(base_cfg, exp_cfg):
    if isinstance(base_cfg, dict) and isinstance(exp_cfg, edict):
        for k, v in exp_cfg.items():
            if k in base_cfg:
                if not isinstance(v, dict):
                    base_cfg[k] = v
                else:
                    _update_config(base_cfg[k], v)
            else:
                raise ValueError('ERROR: {} not exist in config.py'.format(k))
    else:
        return


def update_config_from_file(filename):
    exp_config = None
    with open(filename) as f:
        exp_config = edict(yaml.safe_load(f))
        _update_config(cfg, exp_config)


# def add_gwm_config(cfg):
#     cfg.GWM = edict()
#     cfg.GWM.MODEL = "AIARESEG"
#     cfg.GWM.RESOLUTION = (320, 320)
#     # cfg.GWM.RESOLUTION = (300, 400)
#     cfg.GWM.FLOW_RES = (320, 320)
#     cfg.GWM.SAMPLE_KEYS = ["rgb"]
#     cfg.GWM.ADD_POS_EMB = False
#     cfg.GWM.CRITERION = "L2"
#     cfg.GWM.L1_OPTIMIZE = False
#     cfg.GWM.HOMOGRAPHY = 'quad'  # False
#     cfg.GWM.HOMOGRAPHY_SUBSAMPLE = 8
#     cfg.GWM.HOMOGRAPHY_SKIP = 0.4
#     cfg.GWM.DATASET = 'DAVIS'
#     cfg.GWM.DATA_ROOT = None
#     cfg.GWM.FLOW2RGB = False
#     cfg.GWM.SIMPLE_REC = False
#     cfg.GWM.DAVIS_SINGLE_VID = None
#     cfg.GWM.USE_MULT_FLOW = False
#     cfg.GWM.FLOW_COLORSPACE_REC = None
#
#     cfg.GWM.DATA_ROOT = '/media/liming/Data/IDP/dataset'
#
#     cfg.GWM.FLOW_CLIP_U_LOW = float('-inf')
#     cfg.GWM.FLOW_CLIP_U_HIGH = float('inf')
#     cfg.GWM.FLOW_CLIP_V_LOW = float('-inf')
#     cfg.GWM.FLOW_CLIP_V_HIGH = float('inf')
#
#     cfg.GWM.FLOW_CLIP = float('inf')
#     cfg.GWM.FLOW_NORM = False
#
#     cfg.GWM.LOSS_MULT = edict()
#     cfg.GWM.LOSS_MULT.REC = 1.0
#     cfg.GWM.LOSS_MULT.HEIR_W = [0.1, 0.3, 0.6]
#
#
#     cfg.GWM.TTA = 100  # Test-time-adaptation
#     cfg.GWM.TTA_AS_TRAIN = False  # Use train-like data logic for test-time-adaptation
#
#     cfg.GWM.LOSS = 'OG'
#
#     cfg.FLAGS = edict()
#     cfg.FLAGS.MAKE_VIS_VIDEOS = False  # Making videos is kinda slow
#     cfg.FLAGS.EXTENDED_FLOW_RECON_VIS = False  # Does not cost much
#     cfg.FLAGS.COMP_NLL_FOR_GT = False  # Should we log loss against ground truth?
#     cfg.FLAGS.DEV_DATA = False
#     cfg.FLAGS.KEEP_ALL = True  # Keep all checkoints
#     cfg.FLAGS.ORACLE_CHECK = False  # Use oracle check to estimate max performance when grouping multiple components
#
#     cfg.FLAGS.INF_TPS = False
#
#     # cfg.FLAGS.UNFREEZE_AT = [(1, 10000), (0, 20000), (-1, 30000)]
#     # cfg.FLAGS.UNFREEZE_AT = [(4, 0), (2, 500), (1, 1000), (-1, 10000)]
#
#     cfg.FLAGS.IGNORE_SIZE_DIV = False
#
#     cfg.FLAGS.IGNORE_TMP = True
#
#     cfg.WANDB = edict()
#     cfg.WANDB.ENABLE = True
#     cfg.WANDB.BASEDIR = '../'
#
#     cfg.DEBUG = False
#
#     cfg.LOG_ID = 'exp'
#     cfg.LOG_FREQ = 250
#     cfg.OUTPUT_BASEDIR = '../outputs'
#     cfg.SLURM = False
#     cfg.SKIP_TB = False
#     cfg.TOTAL_ITER = 20000
#     cfg.CONFIG_FILE = None
#
#     if os.environ.get('SLURM_JOB_ID', None):
#         cfg.LOG_ID = os.environ.get('SLURM_JOB_NAME', cfg.LOG_ID)
#         # logger.info(f"Setting name {cfg.LOG_ID} based on SLURM job name")
#
# add_gwm_config(cfg)