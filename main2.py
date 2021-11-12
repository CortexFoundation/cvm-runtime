from os import path

from mrt.V3.utils import (
    get_cfg_defaults, parser, merge_cfg, dest2yaml)
from mrt.V3.execute import run

parser.add_argument("yaml_file", type=str)
parser.add_argument(
    "--pass-name", type=str, default="all", choices=[
        "all", "prepare", "calibrate", "quantize", "evaluate", "compile"])

def override_cfg_argparse(cfg, args):
    if cfg.is_frozen():
        cfg.defrost()
    for dest in dir(args):
        if dest not in dest2yaml:
            continue
        pname, attr = dest2yaml[dest]
        cnode = getattr(cfg, pname)
        argv = getattr(args, dest)
        if argv is not None:
            setattr(cnode, attr, argv)
    cfg.freeze()
    return cfg

if __name__ == "__main__":
    args = parser.parse_args()
    yaml_file = args.yaml_file
    cfg = merge_cfg(yaml_file)
    cfg = override_cfg_argparse(cfg, args)
    pass_name = args.pass_name
    run(cfg, pass_name)
