from os import path
import sys

from mrt.V3.utils import get_cfg_defaults, merge_cfg
from mrt.V3.execute import run

def override_cfg_args(cfg, argv):
    if cfg.is_frozen():
        cfg.defrost()

    for i in range(2, len(argv), 2):
        attr, value = argv[i:i+2]
        try:
            value = eval(value)
        except NameError:
            pass
        pass_name, pass_attr = [s.upper() for s in attr[2:].split(".")]
        cnode = getattr(cfg, pass_name)
        setattr(cnode, pass_attr, value)
    cfg.freeze()
    return cfg

if __name__ == "__main__":
    assert len(sys.argv) >= 2 and len(sys.argv)%2 == 0, \
        "invalid length: {} of sys.argv: {}".format(length, sys.argv)
    yaml_file = sys.argv[1]
    cfg = get_cfg_defaults()
    cfg = merge_cfg(yaml_file)
    cfg = override_cfg_args(cfg, sys.argv)
    run(cfg)
