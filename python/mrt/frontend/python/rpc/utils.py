import sys
import io

from yacs.config import CfgNode as CN

from mrt.V3.execute import run
from rpc import streamer
from rpc.log import get_logger

def get_streamer(yaml_file_str):
    cfg = CN().load_cfg(yaml_file_str)
    cfg.freeze()
    logger = get_logger(cfg.COMMON.VERBOSITY, streamer.printer)
    my_streamer = streamer.Streamer(run, (cfg, logger))
    return my_streamer

def stringify_cfg(cfg):
    # TODO(ryt): replace by appropriately 
    # configured yacs interface cfg.dump(**kwargs)
    old_stdout = sys.stdout
    sys.stdout = new_stdout = io.StringIO()
    print(cfg)
    yaml_file_str = new_stdout.getvalue()
    sys.stdout = old_stdout
    return yaml_file_str
