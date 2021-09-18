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
default_batch = 16

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

def batch_axis(input_shape):
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
    model_prefix = get_model_prefix(args)
    logger = get_logger(args)
    conf_prep_file = model_prefix + ".prepare.conf"
    conf_map = {}

    # preparation
    sym_path, prm_path = load_fname(model_prefix)
    if not path.exists(sym_path) or not path.exists(prm_path):
        save_model(
            args.model_name, data_dir=args.model_dir,
            ctx=get_ctx(args.device_type_prepare, args.device_ids_prepare))
    model = Model.load(sym_path, prm_path)
    model.prepare(set_batch(args.input_shape, 1))
    sym_prep_file, prm_prep_file = load_fname(
        model_prefix, suffix="prepare")
    model.save(sym_prep_file, prm_prep_file)
    conf_map["input_shape"] = args.input_shape
    save_conf(conf_prep_file, logger=logger, **conf_map)
    logger.info("preparation stage finihed")

    # model splitting
    split_keys = args.split_keys
    if split_keys:
        sym_top_file, prm_top_file = load_fname(model_prefix, suffix='top')
        sym_base_file, prm_base_file = load_fname(
            model_prefix, suffix="base")
        base, top = model.split(split_keys)
        top.save(sym_top_file, prm_top_file)
        base.save(sym_base_file, prm_base_file)
        conf_map["split_keys"] = split_keys
        save_conf(conf_prep_file, logger=logger, **conf_map)
        logger.info("model splitting finished")
    else:
        logger.info("model splitting skipped")

@cmd.option("--batch-calibrate", type=int)
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
    batch = default_batch if args.batch_calibrate is None \
        else args.batch_calibrate
    model_prefix = get_model_prefix(args)
    logger = get_logger(args)
    conf_prep_file = model_prefix + ".prepare.conf"
    check_file_existance(conf_prep_file, logger=logger)
    conf_map = load_conf(conf_prep_file, logger=logger)

    # calibration
    if conf_map.get("split_keys", "") == "":
        sym_prep_file, prm_prep_file = load_fname(
            model_prefix, suffix="prepare")
        check_file_existance(sym_prep_file, prm_prep_file, logger=logger)
        mrt = Model.load(sym_prep_file, prm_prep_file).get_mrt()
    else:
        sym_base_file, prm_base_file = load_fname(
            model_prefix, suffix="base")
        check_file_existance(sym_base_file, prm_base_file, logger=logger)
        mrt = Model.load(sym_base_file, prm_base_file).get_mrt()
    shp = set_batch(conf_map["input_shape"], batch)
    dataset = ds.DS_REG[args.dataset_name](shp, root=args.dataset_dir)
    data_iter_func = dataset.iter_func()
    if len(args.device_ids_calibrate) > 1:
        raise RuntimeError(
            "device ids should be an integer in calibration stage")
    ctx = get_ctx(args.device_type_calibrate, args.device_ids_calibrate)
    for i in range(args.calibrate_num):
        data, _ = data_iter_func()
        mrt.set_data(data)
        mrt.calibrate(lambd=args.lambd, ctx=ctx)
    mrt.save(args.model_name+".mrt.calibrate", datadir=args.model_dir)
    conf_map["dataset_name"] = args.dataset_name
    save_conf(model_prefix+".calibrate.conf", logger=logger, **conf_map)
    logger.info("calibrate stage finished")

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
    model_prefix = get_model_prefix(args)
    logger = get_logger(args)
    conf_calib_file = model_prefix + ".calibrate.conf"
    check_file_existance(conf_calib_file, logger=logger)
    conf_map = load_conf(conf_calib_file, logger=logger)
    sym_calib_file, prm_calib_file, ext_calib_file = load_fname(
        model_prefix, suffix="mrt.calibrate", with_ext=True)
    check_file_existance(
        sym_calib_file, prm_calib_file, ext_calib_file, logger=logger)
    mrt = MRT.load(args.model_name+".mrt.calibrate", datadir=args.model_dir)
    conf_quant_file = model_prefix + ".quantize.conf"

    # restoration configuration
    restore_names = args.restore_names
    name_to_op = {}
    for sym in topo_sort(mrt.current_model.symbol):
        name, op_name = sym.attr('name'), sym.attr('op_name')
        if op_name not in name_to_op:
            name_to_op[op_name] = []
        name_to_op[op_name].append(name)
    new_names = []
    for name in restore_names:
        if name.startswith("_OP_") and name[4:] in name_to_op:
            for new_name in name_to_op[name[4:]]:
                new_names.append(new_name)
        else:
            new_names.append(name)
    restore_names = set(new_names)
    if '_ALL_EXCEPT_' in restore_names:
        from tfm_base import _pass_manager
        from tfm_ops import disabled_restore_ops

        quantize_ops = [op_name for op_name in _pass_manager["quantize"] \
                        if op_name not in disabled_restore_ops]
        restore_names_new = []
        for sym in topo_sort(mrt.current_model.symbol):
            name, op_name = sym.attr('name'), sym.attr('op_name')
            if op_name in quantize_ops and \
                name not in restore_names:
                restore_names_new.append(name)
        restore_names = set(restore_names_new)
    for name in restore_names:
        mrt.set_restore(name)

    # hyper parameters configuration
    if args.input_precision is not None:
        mrt.set_input_prec(args.input_precision)
    if args.output_precision is not None:
        mrt.set_output_prec(args.output_precision)
    ctx = get_ctx(args.device_type_quantize, args.device_ids_quantize)
    if args.softmax_lambd is not None:
        mrt.set_softmax_lambd(args.softmax_lambd)
    if args.shift_bits is not None:
        mrt.set_shift_bits(args.shift_bits)
    if args.thresholds is not None:
        thresholds = json.loads(args.thresholds)
        for name, threshold in thresholds.items():
            mrt.set_threshold(name, threshold)

    # quantization
    mrt.quantize()
    mrt.save(args.model_name+".mrt.quantize", datadir=args.model_dir)
    input_shape = conf_map["input_shape"]
    oscales = mrt.get_output_scales()
    inputs_ext = mrt.get_inputs_ext()
    infos = [oscales, inputs_ext]
    ext_all_file = model_prefix + ".all.quantize.ext"
    sim.save_ext(ext_all_file, *infos)
    save_conf(conf_quant_file, logger=logger, **conf_map)
    logger.info("quantization stage finished")

    # mergemodel
    if conf_map.get("split_keys", "") != "":
        qmodel = mrt.current_model
        if args.attribute_deps is None:
            raise RuntimeError("model merging, please specify --attribute_deps")
        attribute_deps = json.loads(args.attribute_deps)
        mrt_oscales = mrt.get_output_scales()
        name_idx = {mrt.get_maps().get(
            s.attr("name"), s.attr("name")): i \
            for i, s in enumerate(qmodel.symbol)}
        def mergefunc(node, params, graph):
            name, op_name = node.attr("name"), node.attr("op_name")
            childs, attr = sutils.sym_iter(
                node.get_children()), node.list_attr()
            if op_name in attribute_deps:
                attr_deps = attribute_deps[op_name]
                for attr_name, v in attr_deps.items():
                    val = sutils.get_attr(attr, attr_name, 0)
                    attr[attr_name] = int(val*mrt_oscales[name_idx[v]])
                node = sutils.get_mxnet_op(op_name)(
                    *childs, **attr, name=name)
            return node
        sym_top_file, prm_top_file = load_fname(model_prefix, suffix="top")
        check_file_existance(sym_top_file, prm_top_file, logger=logger)
        top = Model.load(sym_top_file, prm_top_file)
        model_merger = Model.merger(qmodel, top, mrt.get_maps())
        qmodel = model_merger.merge(callback=mergefunc)
        if args.oscale_maps is None:
            raise RuntimeError("model merging, please specify --oscale_maps")
        oscale_maps = json.loads(args.oscale_maps)
        oscales = model_merger.get_output_scales(
            mrt_oscales, oscale_maps)
        sym_all_file, prm_all_file, ext_all_file = load_fname(
            model_prefix, suffix="all.quantize", with_ext=True)
        qmodel.save(sym_all_file, prm_all_file)
        infos = [oscales, inputs_ext]
        sim.save_ext(ext_all_file, *infos)
        save_conf(conf_quant_file, logger=logger, **conf_map)
        logger.info("model merging finished")
    else:
        logger.info("model merging skipped")

@cmd.option("--batch-evaluate", type=int)
@cmd.option("--device-type-evaluate", type=str, choices=["cpu", "gpu"])
@cmd.option("--device-ids-evaluate", nargs="+", type=int)
@cmd.option("--iter-num", type=int, default=0)
@cmd.module("evaluate", as_main=True, refs=["modelprefix", "logger"],
            description="""
MRT Python Tool: evaluation stage
""")
def mrt_evaluate(args):
    model_prefix = get_model_prefix(args)
    logger = get_logger(args)
    batch = default_batch if args.batch_evaluate is None \
        else args.batch_evaluate
    conf_quant_file = model_prefix + ".quantize.conf"
    check_file_existance(conf_quant_file, logger=logger)
    conf_map = load_conf(conf_quant_file, logger=logger)
    ctx = get_ctx(args.device_type_evaluate, args.device_ids_evaluate)
    if isinstance(ctx, mx.Context):
        ctx = [ctx]

    # forward function for the orginal model
    omodel = Model.load(*load_fname(model_prefix))
    graph = omodel.to_graph(ctx=ctx)
    dataset_name = conf_map["dataset_name"]
    input_shape = conf_map["input_shape"]
    dataset = ds.DS_REG[dataset_name](set_batch(input_shape, batch))
    data_iter_func = dataset.iter_func()
    metric = dataset.metrics()
    baxis = batch_axis(input_shape)
    olen = len(omodel.symbol)

    def forward(net, data, ctx):
        """ Multiple xpu run support.
        """
        data = gluon.utils.split_and_load(
            data, ctx_list=ctx, batch_axis=baxis, even_split=False)
        outs = [net(d) for d in data]
        if olen == 1:
            outs = nd.concatenate(outs)
        else:
            outs = [nd.concatenate([outs[i][j] \
                for i in range(len(outs))]) for j in range(olen)]
        return outs

    def evalfunc(data, label):
        outs = forward(graph, data, ctx=ctx)
        acc = dataset.validate(metric, outs, label)
        return acc

    # forward function for the quantized model
    num_xpus = len(ctx)
    if batch % num_xpus:
        raise RuntimeError("Batch must be divisible by the number of xpus")
    split_batch = batch // num_xpus
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
    rqmodel = reduce_graph(qmodel, {
        'data': set_batch(input_shape, split_batch)})
    qgraph = rqmodel.to_graph(ctx=ctx)
    qmetric = dataset.metrics()

    def quantize(data, label):
        data = sim.load_real_data(data, 'data', inputs_ext)
        outs = forward(qgraph, data, ctx)
        outs = outs / oscales[0] if olen == 1 \
            else [(t / oscales[i]) for i, t in enumerate(outs)]
        acc = dataset.validate(qmetric, outs, label)
        return acc

    # evaluate
    if args.iter_num > 0:
        logger.info("Validating...")
        utils.multi_validate(
            evalfunc, data_iter_func, quantize, iter_num=args.iter_num,
            logger=logging.getLogger('mrt.validate'), batch_size=batch)
        logger.info("evaluatation stage finished")
    else:
        logger.info("evaluatation stage skipped")

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
@cmd.option("--batch", type=int, default=default_batch)
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
    return
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
