from os import path
import sys

from mrt.V3.utils import get_cfg_defaults, merge_cfg, override_cfg_args
from mrt.V3.execute import run
import metric
import metric_v2

if __name__ == "__main__":
    assert len(sys.argv) >= 1 and len(sys.argv)%2 == 1, \
        "invalid length: {} of sys.argv: {}".format(
        len(sys.argv), sys.argv)
    yaml_file = path.join(
        path.dirname(path.realpath(__file__)), "yolov5s-0040.yaml")
    cfg = get_cfg_defaults()
    cfg = merge_cfg(yaml_file)
    cfg = override_cfg_args(cfg, sys.argv[1:])
    run(cfg)
