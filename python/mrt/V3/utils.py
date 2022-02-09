"""
Utility Module for MRT V3.

Collector of utility functions including YAML configuration node manipulating
functions, MRT stage helper functions. Default YAML configurations for MRT
Common Stage options and Command line help prompt are also included.
"""

from os import path
import logging
import json
from yacs.config import CfgNode as CN

import mxnet as mx

from mrt import conf
from mrt.common import log
from mrt.utils import extend_fname

DOC = """
COMMON Stage Options:
    --common.pass_name          Stage to be executed, chosen from "all", "prepare", "calibrate", "quantize", "evaluate", "compile".
    --common.model_dir          Model root directory.
    --common.model_name         Name of the model file without file extension.
    --common.verbosity          Control the logger hiearchy, chosen from "debug", "info", "warning", "error", "critical".
    --common.start_after        Name of the stage to start the execution from, chosen from "initial", "prepare", "calibrate", "quantize".
    --common.device_type        Default context type for all stages chosen from "cpu" or "gpu".
    --common.device_ids         A comma list within square brackets specifying the context ids, eg.[0,1,2].
    --common.input_shape        Shape of the input data.
    --common.batch              Default batch size for all stages.
    --common.run_evaluate       Flag for determining whether to execute evaluation stage, "True" for execution, otherwise "False".
    --common.run_compile        Flag for determining whether to execute compilation stage, "True" for execution, otherwise "False".
"""

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

def get_model_prefix(model_dir, model_name):
    """
    Get the prefix of the pre-trained MRT model name.

    Parameters
    ----------
    model_dir : str
        Directory of the model file.
    model_name : str
        Name of the MRT pre-trained model.

    Returns
    -------
    model_prefix : str
        Prefix of the pre-trained MRT model name.
    """
    if model_dir.startswith("~"):
        model_dir = path.expanduser(model_dir)
    assert path.exists(model_dir), \
        "model_dir: {} does not exist".format(model_dir)
    model_prefix = path.join(model_dir, model_name)
    return model_prefix

def get_logger(verbosity):
    """
    Get the console logger

    Parameters
    ----------
    verbosity : str
        verbosity level chosen from `debug`, `info`, `warning`, `error`, `fatal`.

    Returns
    -------
    logger : logging.RootLogger
        Console logger.
    """
    log.Init(log.name2level(verbosity.upper()))
    logger = logging.getLogger("log.main")
    return logger

def set_batch(input_shape, batch):
    """
    Get the input shape with respect to a specified batch value
    and an original input shape.

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

def save_conf(fname, logger=logging.getLogger(""), **conf_map):
    """
    Save the JSON-formatted MRT configuration from configuration checkpoint file.

    Parameters
    ----------
    fname : str
        Path of the JSON-formatted MRT configuration checkpoint file.
    logger : logging.RootLogger
        Console logger.
    conf_map : dict
        Dictionary of the attribute-value pairs.
    """
    try:
        info_s = json.dumps(conf_map, indent=4)
    except:
        logger.error("Json seralize invalid with data: {}".format(conf_map))
        raise RuntimeError
    with open(fname, "w") as f:
        f.write(info_s)

def load_conf(fname, logger=logging.getLogger("")):
    """
    Load the JSON-formatted MRT configuration from configuration checkpoint file.

    Parameters
    ----------
    fname : str
        Path of the JSON-formatted MRT configuration checkpoint file.
    logger : logging.RootLogger
        Console logger.

    Returns
    -------
    conf_map : dict
        Dictionary of the attribute-value pairs.
    """
    with open(fname, "r") as f:
        try:
            conf_map = json.load(f)
        except:
            logger.error("Json deserialize invalid, fname: {}".format(fname))
    return conf_map

def check_file_existance(*fpaths, logger=logging.getLogger("")):
    """
    Check the existance of the listed file paths.

    Parameters
    ----------
    fpaths : list of str
        List of paths to be checked.
    logger : logging.RootLogger
        Console logger.
    """
    for fpath in fpaths:
        if not path.exists(fpath):
            raise FileNotFoundError("fpath: {} does not exist".format(fpath))

def get_ctx(device_type, device_ids, dctx=default_ctx):
    """
    Get the context instance of mxnet

    Parameters
    ----------
    device_type : str
        context type string chosen from `cpu` or `gpu`.
    device_ids : list
        list of context ids
    dctx: mx.context.Context
        default context

    Returns
    -------
    context : mx.context.Context
        The created context with respect to the device_type and device_ids.
    """
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
    """
    Get the batch axis entry of an input shape.

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
    """
    Get a yacs CfgNode object with default values for MRT.

    Returns
    -------
    cfg : yacs.config.CfgNode
        CfgNode represents an internal node in the configuration tree.
        It's a simple dict-like container that allows for
        attribute-based access to keys.
    """
    # Return a clone so that the defaults will not be altered
    # This is for the "local variable" use pattern
    return MRT_CFG.clone()

def merge_cfg(yaml_file):
    """
    Override the default YAML configuration node with the
    provided YAML-formatted configuration file.

    Parameters
    ----------
    yaml_file : str
        Path of the YAML-formatted configuration file.

    Returns
    -------
    cfg : yacs.config.CfgNode
        CfgNode represents an internal node in the configuration tree.
        It's a simple dict-like container that allows for
        attribute-based access to keys.
    """
    if yaml_file.startswith("~"):
        yaml_file = path.expanduser(yaml_file)
    cfg = get_cfg_defaults()
    cfg.merge_from_file(yaml_file)
    cfg.freeze()
    return cfg

def revise_cfg(cfg, stage, attr, value):
    """
    Revise MRT YAML configuration node with respect to the specified stage
    and attribute name into the provided value.

    Parameters
    ----------
    cfg : yacs.config.CfgNode
        CfgNode represents an internal node in the configuration tree.
        It's a simple dict-like container that allows for
        attribute-based access to keys.
    stage : str
        Stage name chosen from `common`, `prepare`, `calibrate`, `quantize`,
        `evaluate` or `compile`.
    attr : str
        Attribute name attribute to the provided stage.
    value : int, float, str, list, tuple, bool or NoneType
        The revision value to be applied, type of value
        should be supported by yacs.config.CfgNode
    """
    if cfg.is_frozen():
        cfg.defrost()
    subcfg = getattr(cfg, stage)
    setattr(subcfg, attr, value)
    cfg.freeze()

def override_cfg_args(cfg, mrt_argv):
    """
    Override YAML configuration node with command line optional arguments
    for the simplicity of MRT configuration revision.

    Parameters
    ----------
    cfg : yacs.config.CfgNode
        CfgNode represents an internal node in the configuration tree.
        It's a simple dict-like container that allows for
        attribute-based access to keys.
    mrt_argv : list
        list of even length which can be resoluted as key value pairs,
        the key could be split into stage name and attribute name.

    Returns
    -------
    cfg : yacs.config.CfgNode
        Overridden YAML configuration node.
    """
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
