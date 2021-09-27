import sys
from os import path

from mrt.defaults import get_cfg_defaults
from mrt.conf import YAML_ROOT
from mrt import mrt_entry as mentry

def yaml_prepare(CM, CN):
    mentry.mrt_prepare(
        CM.MODEL_DIR, CM.MODEL_NAME, CM.VERBOSITY, CN.DEVICE_TYPE,
        CN.DEVICE_IDS, CN.INPUT_SHAPE, CN.SPLIT_KEYS)

def yaml_calibrate(CM, CN):
    mentry.mrt_calibrate(
        CM.MODEL_DIR, CM.MODEL_NAME, CM.VERBOSITY, CN.DATASET_NAME,
        CN.DATASET_DIR, CN.DEVICE_TYPE, CN.DEVICE_IDS, CN.NUM_CALIBRATE,
        CN.LAMBD, batch=CN. BATCH)

def yaml_quantize(CM, CN):
    mentry.mrt_quantize(
        CM.MODEL_DIR, CM.MODEL_NAME, CM.VERBOSITY, CN.RESTORE_NAMES,
        CN.INPUT_PRECISION, CN.OUTPUT_PRECISION, CN.DEVICE_TYPE, CN.DEVICE_IDS,
        CN.SOFTMAX_LAMBD, CN.SHIFT_BITS, CN.THRESHOLDS, CN.ATTRIBUTE_DEPS,
        CN.OSCALE_MAPS)

def yaml_evaluate(CM, CN):
    mentry.mrt_evaluate(
        CM.MODEL_DIR, CM.MODEL_NAME, CM.VERBOSITY, CN.DEVICE_TYPE, CN.DEVICE_IDS,
        CN.ITER_NUM, batch=CN.BATCH)

def yaml_compile(CM, CN):
    mentry.mrt_compile(
        CM.MODEL_DIR, CM.MODEL_NAME, CM.VERBOSITY, CN.DUMP_DIR,
        device_type=CN.DEVICE_TYPE, device_ids=CN.DEVICE_IDS, batch=CN.BATCH)

def yaml_main(cfg):
    for prefix in ["BATCH", "DEVICE_TYPE", "DEVICE_IDS"]:
        for subcfg in [
            cfg.PREPARE, cfg.CALIBRATE, cfg.QUANTIZE, cfg.EVALUATE,
            cfg.COMPILE]:
            for attr in dir(subcfg):
                if attr == prefix and getattr(subcfg, prefix) is None:
                    setattr(subcfg, prefix, getattr(cfg.COMMON, prefix)
    start_pos = 0
    start_pos_map = {'prepare': 1, 'calibrate': 2, 'quantize': 3}
    if cfg.COMMON.START_AFTER in start_pos_map:
        start_pos = start_pos_map[cfg.COMMON.START_AFTER]
    if start_pos < 1:
        yaml_prepare(cfg.COMMON, cfg.PREPARE)
    if start_pos < 2:
        yaml_calibrate(cfg.COMMON, cfg.CALIBRATE)
    if start_pos < 3:
        yaml_quantize(cfg.COMMON, cfg.QUANTIZE)
    if cfg.COMMON.RUN_EVALUATE:
        yaml_evaluate(cfg.COMMON, cfg.EVALUATE)
    if cfg.COMMON.RUN_COMPILE:
        yaml_compile(cfg.COMMON, cfg.COMPILE)

if __name__ == "__main__":
    assert len(sys.argv) >= 2, len(sys.argv)
    model_name = sys.argv[1]
    yaml_file = path.join(YAML_ROOT, model_name+".yaml")
    cfg = get_cfg_defaults()
    cfg.merge_from_file(yaml_file)
    cfg.freeze()
    if len(sys.argv) == 3:
        entry_name = sys.argv[2]
        yaml_func_name = "mrt_{}".format(entry_name)
        if not hasattr(mentry, yaml_func_name):
            raise RuntimeError(
                "invalid entry_name: {}, yaml_func_name: {}".format(
                    entry_name, yaml_func_name))
        yaml_func = getattr(mentry, yaml_func_name)
        cfg_node_name = entry_name.upper()
        if not hasattr(cfg, cfg_node_name):
            raise RuntimeError(
                "invalid entry_name: {}, cfg_node_name: {}".format(
                    entry_name, cfg_node_name))
        cfg_node = getattr(cfg, cfg_node_name)
        yaml_func(cfg.COMMON, cfg_node)
    else:
        yaml_main(cfg)
