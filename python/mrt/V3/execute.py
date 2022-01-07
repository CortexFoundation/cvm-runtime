import sys

from mrt.V3.prepare import prepare
from mrt.V3.calibrate import calibrate
from mrt.V3.quantize import quantize
from mrt.V3.evaluate import evaluate
from mrt.V3.mrt_compile import mrt_compile

thismodule = sys.modules[__name__]

def yaml_main(cfg, logger=None):
    if cfg.is_frozen():
        cfg.defrost()
    for prefix in ["BATCH", "DEVICE_TYPE", "DEVICE_IDS"]:
        for subcfg in [cfg.PREPARE, cfg.CALIBRATE, cfg.QUANTIZE,
            cfg.EVALUATE, cfg.COMPILE]:
            for attr in dir(subcfg):
                if attr == prefix and getattr(subcfg, prefix) is None:
                    setattr(subcfg, prefix, getattr(cfg.COMMON, prefix))
    if not cfg.is_frozen():
        cfg.freeze()
    start_pos = 0
    start_pos_map = {
        'initial': 0, 'prepare': 1, 'calibrate': 2, 'quantize': 3}
    start_after = cfg.COMMON.START_AFTER
    assert start_after in start_pos_map, \
        "start_after: {}, start_pos_map: {}".format(
            start_after, start_pos_map)
    start_pos = start_pos_map[start_after]
    if start_pos < 1:
        prepare(cfg.COMMON, cfg.PREPARE, logger=logger)
    if start_pos < 2:
        calibrate(cfg.COMMON, cfg.CALIBRATE, logger=logger)
    if start_pos < 3:
        quantize(cfg.COMMON, cfg.QUANTIZE, logger=logger)
    if cfg.COMMON.RUN_EVALUATE:
        evaluate(cfg.COMMON, cfg.EVALUATE, logger=logger)
    if cfg.COMMON.RUN_COMPILE:
        mrt_compile(cfg.COMMON, cfg.COMPILE, logger=logger)

def run(cfg, logger=None):
    pass_name = cfg.COMMON.PASS_NAME
    if pass_name == "all":
        yaml_main(cfg, logger=logger)
    else:
        if pass_name == "compile":
            pass_name = "mrt_compile"
        if not hasattr(thismodule, pass_name):
            raise RuntimeError("invalid pass_name: {}".format(pass_name))
        yaml_func = getattr(thismodule, pass_name)
        cm_cfg = cfg.COMMON
        if pass_name == "mrt_compile":
            cfg_name = "COMPILE"
        else:
            cfg_name = pass_name.upper()
        pass_cfg = getattr(cfg, cfg_name)
        yaml_func(cm_cfg, pass_cfg, logger=logger)
