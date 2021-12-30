from os import path
import sys

from mrt.V3.utils import get_cfg_defaults, merge_cfg, override_cfg_args
from mrt.V3.execute import run

if __name__ == "__main__":
    assert len(sys.argv) >= 2 and len(sys.argv)%2 == 0, \
        "invalid length: {} of sys.argv: {}".format(len(sys.argv), sys.argv)
    yaml_file = sys.argv[1]
    cfg = get_cfg_defaults()
    cfg = merge_cfg(yaml_file)
    cfg = override_cfg_args(cfg, sys.argv[2:])
    run(cfg)
