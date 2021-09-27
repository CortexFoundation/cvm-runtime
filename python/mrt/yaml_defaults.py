from yacs.config import CfgNode as CN
from . import conf
from mrt import mrt_entry as mentry

MRT_CFG = CN()

MRT_CFG.COMMON = CN()
MRT_CFG.COMMON.MODEL_DIR = conf.MRT_MODEL_ROOT
MRT_CFG.COMMON.MODEL_NAME = conf.MRT_MODEL_ROOT
MRT_CFG.COMMON.VERBOSITY = "debug"
MRT_CFG.COMMON.START_AFTER = None
MRT_CFG.COMMON.DEVICE_TYPE = mentry.default_device_type
MRT_CFG.COMMON.DEVICE_IDS = mentry.default_device_ids
MRT_CFG.COMMON.BATCH = mentry.default_batch
MRT_CFG.COMMON.RUN_EVALUATE = True
MRT_CFG.COMMON.RUN_COMPILE = True

MRT_CFG.PREPARE = CN()
MRT_CFG.PREPARE.DEVICE_TYPE = mentry.default_device_type
MRT_CFG.PREPARE.DEVICE_IDS = mentry.default_device_ids
MRT_CFG.PREPARE.INPUT_SHAPE = [-1, 3, 224, 224]
MRT_CFG.PREPARE.SPLIT_KEYS = ""

MRT_CFG.CALIBRATE = CN()
MRT_CFG.CALIBRATE.BATCH = mentry.default_batch
MRT_CFG.CALIBRATE.NUM_CALIB = 1
MRT_CFG.CALIBRATE.LAMBD = None
MRT_CFG.CALIBRATE.DATASET_NAME = "imagenet"
MRT_CFG.CALIBRATE.DATASET_DIR = conf.MRT_DATASET_ROOT
MRT_CFG.CALIBRATE.DEVICE_TYPE = mentry.default_device_type
MRT_CFG.CALIBRATE.DEVICE_IDS = mentry.default_device_ids

MRT_CFG.QUANTIZE = CN()
MRT_CFG.QUANTIZE.RESTORE_NAMES = []
MRT_CFG.QUANTIZE.INPUT_PRECISION = None
MRT_CFG.QUANTIZE.OUTPUT_PRECISION = None
MRT_CFG.QUANTIZE.DEVICE_TYPE = mentry.default_device_type
MRT_CFG.QUANTIZE.DEVICE_IDS = mentry.default_device_ids
MRT_CFG.QUANTIZE.SOFTMAX_LAMBD = None
MRT_CFG.QUANTIZE.SHIFT_BITS = None
MRT_CFG.QUANTIZE.THRESHOLDS = None
MRT_CFG.QUANTIZE.ATTRIBUTE_DEPS = None
MRT_CFG.QUANTIZE.OSCALE_MAPS = None

MRT_CFG.EVALUATE = CN()
MRT_CFG.EVALUATE.BATCH = mentry.default_batch
MRT_CFG.EVALUATE.DEVICE_TYPE = mentry.default_device_type
MRT_CFG.EVALUATE.DEVICE_IDS = mentry.default_device_ids
MRT_CFG.EVALUATE.ITER_NUM = 10

MRT_CFG.COMPILE = CN()
MRT_CFG.COMPILE.BATCH = 1
MRT_CFG.COMPILE.DUMP_DIR = "/data1/tmp"
MRT_CFG.COMPILE.DEVICE_TYPE = mentry.default_device_type
MRT_CFG.COMPILE.DEVICE_IDS = mentry.default_device_ids

def get_cfg_defaults():
    """Get a yacs CfgNode object with default values for mrt."""
    # Return a clone so that the defaults will not be altered
    # This is for the "local variable" use pattern
    return MRT_CFG.clone()
