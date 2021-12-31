from os import path
import logging
import json
from yacs.config import CfgNode as CN

import mxnet as mx

from mrt import conf
from mrt.common import log
from mrt.utils import extend_fname

# TODO: jiazhen branch code design

default_device_type = "cpu"
default_device_ids = [0]
default_batch = 16
default_ctx = mx.cpu()

MRT_CFG = CN()
MRT_CFG.COMMON = CN()
MRT_CFG.COMMON.PASS_NAME = "all"
MRT_CFG.COMMON.MODEL_DIR = conf.MRT_MODEL_ROOT
MRT_CFG.COMMON.MODEL_NAME = None
MRT_CFG.COMMON.VERBOSITY = "debug"
MRT_CFG.COMMON.START_AFTER = "initial"
MRT_CFG.COMMON.DEVICE_TYPE = default_device_type
MRT_CFG.COMMON.DEVICE_IDS = default_device_ids
MRT_CFG.COMMON.BATCH = default_batch
MRT_CFG.COMMON.RUN_EVALUATE = True
MRT_CFG.COMMON.RUN_COMPILE = True

def update_dest2yaml(dest2yaml_upt):
    for dest, cfg in dest2yaml_upt.items():
        if dest in dest2yaml:
            raise RuntimeError(
                "dest: {} already in dest2yaml: {}".format(
                    dest, dest2yaml.keys()))
        dest2yaml[dest] = cfg

def get_model_prefix(model_dir, model_name):
    if model_dir.startswith("~"):
        model_dir = path.expanduser(model_dir)
    assert path.exists(model_dir), \
        "model_dir: {} does not exist".format(model_dir)
    model_prefix = path.join(model_dir, model_name)
    return model_prefix

def get_logger(verbosity):
    log.Init(log.name2level(verbosity.upper()))
    logger = logging.getLogger("log.main")
    return logger

def set_batch(input_shape, batch):
    """Get the input shape with respect to a specified batch value and an original input shape.

    Parameters
    ----------
    input_shape : tuple
        The input shape with batch axis unset.
    batch : int
        The batch value.

    Returns
    -------
    ishape : tuple
        The input shape with the value of batch axis equal to batch.
    """
    return [batch if s == -1 else s for s in input_shape]

def load_fname(prefix, suffix=None, with_ext=False):
    """Get the model files at a given stage.

    Parameters
    ----------
    prefix : string
        The file path without and extension.
    suffix : string
        The file suffix with respect to a given stage of MRT.
    with_ext: bool
        Whether to include ext file.

    Returns
    -------
    files : tuple of string
        The loaded file names.
    """
    suffix = "."+suffix if suffix is not None else ""
    return extend_fname(prefix+suffix, with_ext)

def save_conf(fname, logger=logging, **conf_map):
    try:
        info_s = json.dumps(conf_map, indent=4)
    except:
        logger.error("Json seralize invalid with data: {}".format(conf_map))
        raise RuntimeError
    with open(fname, "w") as f:
        f.write(info_s)

def load_conf(fname, logger=logging):
    with open(fname, "r") as f:
        try:
            conf_map = json.load(f)
        except:
            logger.error("Json deserialize invalid, fname: {}".format(fname))
    return conf_map

def check_file_existance(*fpaths, logger=logging):
    for fpath in fpaths:
        if not path.exists(fpath):
            raise FileNotFoundError("fpath: {} does not exist".format(fpath))

def get_ctx(device_type, device_ids, dctx=default_ctx):
    if device_type is None:
        device_type = default_device_type
    if device_ids is None:
        device_ids = default_device_ids
    contex = dctx
    if device_type == "gpu":
        contex = mx.gpu(device_ids[0]) if len(device_ids) == 1 \
              else [mx.gpu(i) for i in device_ids]
    return contex

def get_batch_axis(input_shape):
    """Get the batch axis entry of an input shape.

    Parameters
    ----------
    input_shape : tuple
        The data shape related to dataset.

    Returns
    -------
    axis : int
        The batch axis entry of an input shape.
    """
    idx = [i for i, s in enumerate(input_shape) if s == -1]
    assert len(idx) == 1
    return idx[0]

def get_cfg_defaults():
    """Get a yacs CfgNode object with default values for mrt."""
    # Return a clone so that the defaults will not be altered
    # This is for the "local variable" use pattern
    return MRT_CFG.clone()

def merge_cfg(yaml_file):
    if yaml_file.startswith("~"):
        yaml_file = path.expanduser(yaml_file)
    cfg = get_cfg_defaults()
    cfg.merge_from_file(yaml_file)
    cfg.freeze()
    return cfg

def revise_cfg(cfg, stage, attr, value):
    if cfg.is_frozen():
        cfg.defrost()
    subcfg = getattr(cfg, stage)
    setattr(subcfg, attr, value)
    cfg.freeze()

def override_cfg_args(cfg, mrt_argv):
    if not mrt_argv:
        return cfg
    if cfg.is_frozen():
        cfg.defrost()

    for i in range(0, len(mrt_argv), 2):
        attr, value = mrt_argv[i:i+2]
        try:
            value = eval(value)
        except:
            pass
        pass_name, pass_attr = [s.upper() for s in attr[2:].split(".")]
        cnode = getattr(cfg, pass_name)
        setattr(cnode, pass_attr, value)
    cfg.freeze()
    return cfg
