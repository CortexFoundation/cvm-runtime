from os import path
import argparse

from mrt.V3.utils import get_cfg_defaults
from rpc.service import local_addr, mrt_execute, mrt_submit
from rpc.utils import stringify_cfg

parser = argparse.ArgumentParser()
parser.add_argument("--host-addr", type=str, default=local_addr)

def test_execute(host_addr):
    cfg = get_cfg_defaults()
    tmp_yaml_file = path.expanduser("~/mrt_yaml_root/alexnet.yaml")
    cfg.merge_from_file(tmp_yaml_file)
    yaml_file_str = stringify_cfg(cfg)
    for message in mrt_execute(yaml_file_str, host_addr=host_addr):
        print(message)

def test_submit(host_addr):
    src_sym_file = path.expanduser("~/mrt_model/alexnet.json")
    src_prm_file = path.expanduser("~/mrt_model/alexnet.params")
    # dst_model_dir = path.expanduser("~/mrt_model_2")
    dst_model_dir = "/home/ycmtrivial/mrt_model"
    for message in mrt_submit(
        src_sym_file, src_prm_file, dst_model_dir,
        host_addr=host_addr):
        print(message)

if __name__ == "__main__":
    args = parser.parse_args()
    host_addr = args.host_addr
    test_execute(host_addr=host_addr)
    test_submit(host_addr=host_addr)
