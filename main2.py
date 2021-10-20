import sys
from os import path

from mrt.V3.utils import get_cfg_defaults
from mrt.V3.prepare import prepare
from mrt.V3.calibrate import calibrate
from mrt.V3.quantize import quantize
from mrt.V3.evaluate import evaluate
from mrt.V3.mrt_compile import mrt_compile

thismodule = sys.modules[__name__]

def yaml_main(cfg):
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
        prepare(cfg.COMMON, cfg.PREPARE)
    if start_pos < 2:
        calibrate(cfg.COMMON, cfg.CALIBRATE)
    if start_pos < 3:
        quantize(cfg.COMMON, cfg.QUANTIZE)
    if cfg.COMMON.RUN_EVALUATE:
        evaluate(cfg.COMMON, cfg.EVALUATE)
    if cfg.COMMON.RUN_COMPILE:
        mrt_compile(cfg.COMMON, cfg.COMPILE)

if __name__ == "__main__":
    assert len(sys.argv) in [2,3], len(sys.argv)
    yaml_file = sys.argv[1]
    if yaml_file.startswith("~"):
        yaml_file = path.expanduser(yaml_file)
    cfg = get_cfg_defaults()
    cfg.merge_from_file(yaml_file)
    cfg.freeze()
    if len(sys.argv) == 3:
        entry_name = sys.argv[2]
        if entry_name == "compile":
            entry_name = "mrt_compile"
        if not hasattr(thismodule, entry_name):
            raise RuntimeError("invalid entry_name: {}".format(entry_name))
        yaml_func = getattr(thismodule, entry_name)
        cm_cfg = cfg.COMMON
        if entry_name == "mrt_compile":
            cfg_name = "COMPILE"
        else:
            cfg_name = entry_name.upper()
        pass_cfg = getattr(cfg, cfg_name)
        yaml_func(cm_cfg, pass_cfg)
    else:
        yaml_main(cfg)
