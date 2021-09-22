import sys
from os import path
import argparse
from typing import Tuple, List, Union
import logging
import json

import mxnet as mx
from mxnet import gluon, ndarray as nd
import numpy as np

from mrt.conf import MRT_MODEL_ROOT, MRT_DATASET_ROOT
from mrt.common import cmd, log, thread
from mrt.transformer import Model, MRT, reduce_graph
from mrt.sym_utils import topo_sort
from mrt import utils
from mrt.gluon_zoo import save_model
from mrt import dataset as ds
from mrt import sym_utils as sutils
from mrt import sim_quant_helper as sim
import mrt.mrt_entry as mentry

# set up dependencies
__ROOT__ = path.dirname(path.realpath(__file__))
sys.path.insert(0, path.join(__ROOT__, "python"))

LOG_MSG = ",".join(["{}:{}".format(l, n) \
    for l, n in zip(log.LOG_LEVELS, log.LOG_NAMES)])

# @cmd.option("-v", "--verbosity", metavar="LEVEL",
            # choices=log.LOG_NAMES, default=log.level2name(log.DEBUG),
            # help="log verbosity to pring information, " + \
                # "available options: {}".format(log.LOG_NAMES) + \
                # " by default {}".format(log.level2name(log.DEBUG)))
# @cmd.global_options()
# def global_func(args):
    # log.Init(log.name2level(args.verbosity))

default_device_type = "cpu"
default_device_ids = [0]
default_ctx = mx.cpu()

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
    return utils.extend_fname(prefix+suffix, with_ext)

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

def save_conf(fname, logger=logging, **conf_map):
    try:
        info_s = json.dumps(conf_map, indent=4)
    except:
        logger.error("Json seralize invalid with data: {}".format(conf_map))
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

@cmd.option("model_name", type=str)
@cmd.option("--model-dir", type=str, default=MRT_MODEL_ROOT)
@cmd.module("modelprefix")
def get_model_prefix(args):
    model_dir = args.model_dir
    if model_dir.startswith("~"):
        model_dir = path.expanduser(model_dir)
    model_name = args.model_name
    assert path.exists(model_dir), \
        "model_dir: {} does not exist".format(model_dir)
    model_prefix = path.join(model_dir, model_name)
    return model_prefix

@cmd.option("--verbosity", type=str, default="debug",
            choices=["none", "debug", "info", "warning", "error", "critical"])
@cmd.module("logger")
def get_logger(args):
    log.Init(log.name2level(args.verbosity.upper()))
    logger = logging.getLogger("log.main")
    return logger

@cmd.option("--device-type-prepare", type=str, choices=["cpu", "gpu"])
@cmd.option("--device-ids-prepare", nargs="+", type=int)
@cmd.option("--input-shape", nargs="+", type=int, default=[-1, 3, 224, 224])
@cmd.option("--split-keys", nargs="+", type=str, default="")
@cmd.module("prepare", as_main=True, refs=["modelprefix", "logger"],
            description="""
MRT Python Tool: preparation stage
""")
def mrt_prepare(args):
    mentry.mrt_prepare(
        args.model_dir, args.model_name, args.verbosity,
        args.device_type_prepare, args.device_ids_prepare,
        args.input_shape, args.split_keys)

@cmd.option("--batch-calibrate", type=int, default=mentry.default_batch)
@cmd.option("--calibrate-num", type=int, default=1)
@cmd.option("--lambd", type=int)
@cmd.option("--dataset-name", type=str, default="imagenet",
            choices=list(ds.DS_REG.keys()))
@cmd.option("--dataset-dir", type=str, default=MRT_DATASET_ROOT)
@cmd.option("--device-type-calibrate", type=str, choices=["cpu", "gpu"])
@cmd.option("--device-ids-calibrate", nargs="+", type=int)
@cmd.module("calibrate", as_main=True, refs=["modelprefix", "logger"],
            description="""
MRT Python Tool: calibration stage
""")
def mrt_calibrate(args):
    mentry.mrt_calibrate(
        args.model_dir, args.model_name, args.verbosity, args.dataset_name,
        args.dataset_dir, args.device_type_calibrate, args.device_ids_calibrate,
        args.calibrate_num, args.lambd, batch=args.batch_calibrate)

@cmd.option("--restore-names", nargs="+", type=str, default=[])
@cmd.option("--input-precision", type=int)
@cmd.option("--output-precision", type=int)
@cmd.option("--device-type-quantize", type=str, choices=["cpu", "gpu"])
@cmd.option("--device-ids-quantize", nargs="+", type=int)
@cmd.option("--softmax-lambd", type=float)
@cmd.option("--shift-bits", type=int)
@cmd.option("--thresholds", type=str)
@cmd.option("--attribute-deps", type=str)
@cmd.option("--oscale-maps", type=str)
@cmd.module("quantize", as_main=True, refs=["modelprefix", "logger"],
            description="""
MRT Python Tool: quantization stage
""")
def mrt_quantize(args):
    mentry.mrt_quantize(
        args.model_dir, args.model_name, args.verbosity, args.restore_names,
        args.input_precision, args.output_precision, args.device_type_quantize,
        args.device_ids_quantize, args.softmax_lambd, args.shift_bits,
        args.thresholds, args.attribute_deps, args.oscale_maps)

@cmd.option("--batch-evaluate", type=int, default=mentry.default_batch)
@cmd.option("--device-type-evaluate", type=str, choices=["cpu", "gpu"])
@cmd.option("--device-ids-evaluate", nargs="+", type=int)
@cmd.option("--iter-num", type=int, default=0)
@cmd.module("evaluate", as_main=True, refs=["modelprefix", "logger"],
            description="""
MRT Python Tool: evaluation stage
""")
def mrt_evaluate(args):
    mentry.mrt_evaluate(
        args.model_dir, args.model_name, args.verbosity,
        args.device_type_evaluate, args.device_ids_evaluate, args.iter_num,
        batch=args.batch_evaluate)

@cmd.option("--batch-compile", type=int)
@cmd.option("--dump-dir", type=str, default="/data1/tmp")
@cmd.option("--device-type-compile", type=str, default="cpu",
            choices=["cpu", "gpu"])
@cmd.option("--device-ids-compile", nargs="+", type=int, default=[0])
@cmd.module("compile", as_main=True, refs=["modelprefix", "logger"],
            description="""
MRT Python Tool: compilation stage
""")
def mrt_compile(args):
    model_prefix = get_model_prefix(args)
    logger = get_logger(args)
    batch = 1 if args.batch_compile is None \
        else args.batch_compile
    conf_quant_file = model_prefix + ".quantize.conf"
    check_file_existance(conf_quant_file, logger=logger)
    conf_map = load_conf(conf_quant_file, logger=logger)
    if args.device_type_compile is None:
        args.device_type_compile = default_device_type
    if args.device_ids_compile is None:
        args.device_ids_compile = default_device_ids
    if len(args.device_ids_compile) > 1:
        raise RuntimeError(
            "device ids should be an integer in compilation stage")
    input_shape = conf_map["input_shape"]

    # compilation
    model_name_tfm = args.model_name + "_cvm"
    device_ids_compile = args.device_ids_compile[0]
    if conf_map.get("split_keys", "") != "":
        sym_all_file, prm_all_file, ext_all_file = load_fname(
            model_prefix, suffix="all.quantize", with_ext=True)
        check_file_existance(
            sym_all_file, prm_all_file, ext_all_file, logger=logger)
        qmodel = Model.load(sym_all_file, prm_all_file)
        oscales, inputs_ext = sim.load_ext(ext_all_file)
    else:
        sym_quant_file, prm_quant_file, ext_quant_file = load_fname(
            model_prefix, suffix="mrt.quantize", with_ext=True)
        check_file_existance(
            sym_quant_file, prm_quant_file, ext_quant_file, logger=logger)
        mrt = MRT.load(args.model_name+".mrt.quantize", datadir=args.model_dir)
        oscales = mrt.get_output_scales()
        inputs_ext = mrt.get_inputs_ext()
        qmodel = mrt.current_model
    qmodel.to_cvm(
        model_name_tfm, datadir=args.dump_dir,
        input_shape=set_batch(input_shape, batch),
        target=args.device_type_compile, device_ids=device_ids_compile)
    dataset = ds.DS_REG[conf_map["dataset_name"]](set_batch(input_shape, batch))
    dump_data, _ = dataset.iter_func()()
    dump_data = sim.load_real_data(
        dump_data.astype("float64"), "data", mrt.get_inputs_ext())
    model_root = path.join(args.dump_dir, model_name_tfm)
    np.save(
        path.join(model_root, "data.npy"), dump_data.astype("int8").asnumpy())
    infos = {
        "inputs_ext": inputs_ext,
        "oscales": oscales,
        "input_shapes": input_shape,
    }
    sim.save_ext(path.join(model_root, "ext"), infos)
    logger.info("compilation stage finished")

@cmd.option("--start-after", type=str,
            choices=["prepare", "calibrate", "quantize"])
@cmd.option("--device-type", type=str, default=default_device_type,
            choices=["cpu", "gpu"])
@cmd.option("--device-ids", nargs="+", type=int, default=default_device_ids)
@cmd.option("--batch", type=int, default=mentry.default_batch)
@cmd.option("--evaluate", action="store_true")
@cmd.option("--compile", action="store_true")
@cmd.module("mrt", as_main=True,
            refs=["prepare", "calibrate", "quantize",
                  "evaluate", "compile"],
            description="""
MRT Python Tool
""")
def mrt_main(args):
    # setting up attributes for all passes
    for prefix in ["batch", "device_type", "device_ids"]:
        for attr in dir(args):
            if attr.startswith(prefix+"_") and getattr(args, attr) is None:
                setattr(args, attr, getattr(args, prefix))

    start_pos = 0
    start_pos_map = {'prepare': 1, 'calibrate': 2, 'quantize': 3}
    if args.start_after in start_pos_map:
        start_pos = start_pos_map[args.start_after]
    if start_pos < 1:
        mrt_prepare(args)
    if start_pos < 2:
        mrt_calibrate(args)
    if start_pos < 3:
        mrt_quantize(args)
    if args.evaluate:
        mrt_evaluate(args)
    if args.compile:
        mrt_compile(args)

if __name__ == "__main__":
    logger = logging.getLogger("main")
    cmd.Run()
