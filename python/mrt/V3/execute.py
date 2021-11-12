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
    start_pos_map = {'prepare': 1, 'calibrate': 2, 'quantize': 3}
    if cfg.COMMON.START_AFTER in start_pos_map:
        start_pos = start_pos_map[cfg.COMMON.START_AFTER]
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

def run(cfg, pass_name, logger=None):
    if pass_name is not None:
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
    else:
        yaml_main(cfg, logger=logger)
