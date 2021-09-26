import sys
from os import path

from mrt.defaults import get_cfg_defaults
from mrt.conf import YAML_ROOT
from mrt import mrt_entry as mentry

if __name__ == "__main__":
    assert len(sys.argv) == 2, len(sys.argv)
    model_name = sys.argv[1]
    yaml_file = path.join(YAML_ROOT, model_name+".yaml")
    cfg = get_cfg_defaults()
    cfg.merge_from_file(yaml_file)
    cfg.freeze()


    # if cfg.SYSTEM.NUM_GPUS > 0:
        # my_project.setup_multi_gpu_support()

    # model = my_project.create_model(cfg)
