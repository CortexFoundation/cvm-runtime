from os import path
import logging
import json

import mxnet as mx

from mrt.gluon_zoo import save_model
from mrt.common import log
from mrt import utils
from mrt.transformer import Model, MRT
from mrt import dataset as ds
from mrt import sym_utils as sutils
from mrt import sim_quant_helper as sim

default_batch = 16
default_ctx = mx.cpu()

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
    return utils.extend_fname(prefix+suffix, with_ext)

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

def mrt_prepare(
    model_dir, model_name, verbosity, device_type, device_ids, input_shape,
    split_keys):
    model_prefix = get_model_prefix(model_dir, model_name)
    logger = get_logger(verbosity)
    conf_prep_file = model_prefix + ".prepare.conf"
    conf_map = {}

    # preparation
    sym_path, prm_path = load_fname(model_prefix)
    if not path.exists(sym_path) or not path.exists(prm_path):
        save_model(
            model_name, data_dir=model_dir,
            ctx=get_ctx(device_type, device_ids))
    model = Model.load(sym_path, prm_path)
    model.prepare(set_batch(input_shape, 1))
    sym_prep_file, prm_prep_file = load_fname(
        model_prefix, suffix="prepare")
    model.save(sym_prep_file, prm_prep_file)
    conf_map["input_shape"] = input_shape
    save_conf(conf_prep_file, logger=logger, **conf_map)
    logger.info("preparation stage finihed")

    # model splitting
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

def mrt_calibrate(
    model_dir, model_name, verbosity, dataset_name, dataset_dir,
    device_type, device_ids, calibrate_num, lambd, batch=default_batch):
    model_prefix = get_model_prefix(model_dir, model_name)
    logger = get_logger(verbosity)
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
    dataset = ds.DS_REG[dataset_name](shp, root=dataset_dir)
    data_iter_func = dataset.iter_func()
    if len(device_ids) > 1:
        raise RuntimeError(
            "device ids should be an integer in calibration stage")
    ctx = get_ctx(device_type, device_ids)
    for i in range(calibrate_num):
        data, _ = data_iter_func()
        mrt.set_data(data)
        mrt.calibrate(lambd=lambd, ctx=ctx)
    mrt.save(model_name+".mrt.calibrate", datadir=model_dir)
    conf_map["dataset_name"] = dataset_name
    save_conf(model_prefix+".calibrate.conf", logger=logger, **conf_map)
    logger.info("calibrate stage finished")

def mrt_quantize(
    model_dir, model_name, verbosity, restore_names, input_precision,
    output_precision, device_type, device_ids, softmax_lambd, shift_bits,
    thresholds, attribute_deps, oscale_maps):
    model_prefix = get_model_prefix(model_dir, model_name)
    logger = get_logger(verbosity)
    conf_calib_file = model_prefix + ".calibrate.conf"
    check_file_existance(conf_calib_file, logger=logger)
    conf_map = load_conf(conf_calib_file, logger=logger)
    sym_calib_file, prm_calib_file, ext_calib_file = load_fname(
        model_prefix, suffix="mrt.calibrate", with_ext=True)
    check_file_existance(
        sym_calib_file, prm_calib_file, ext_calib_file, logger=logger)
    mrt = MRT.load(model_name+".mrt.calibrate", datadir=model_dir)
    conf_quant_file = model_prefix + ".quantize.conf"

    # restoration configuration
    name_to_op = {}
    for sym in sutils.topo_sort(mrt.current_model.symbol):
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
        for sym in sutils.topo_sort(mrt.current_model.symbol):
            name, op_name = sym.attr('name'), sym.attr('op_name')
            if op_name in quantize_ops and \
                name not in restore_names:
                restore_names_new.append(name)
        restore_names = set(restore_names_new)
    for name in restore_names:
        mrt.set_restore(name)

    # hyper parameters configuration
    if input_precision is not None:
        mrt.set_input_prec(input_precision)
    if output_precision is not None:
        mrt.set_output_prec(output_precision)
    ctx = get_ctx(device_type, device_ids)
    if softmax_lambd is not None:
        mrt.set_softmax_lambd(softmax_lambd)
    if shift_bits is not None:
        mrt.set_shift_bits(shift_bits)
    if thresholds is not None:
        thresholds = json.loads(thresholds)
        for name, threshold in thresholds.items():
            mrt.set_threshold(name, threshold)

    # quantization
    mrt.quantize()
    mrt.save(model_name+".mrt.quantize", datadir=model_dir)
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
        if attribute_deps is None:
            raise RuntimeError("model merging, please specify --attribute_deps")
        attribute_deps = json.loads(attribute_deps)
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
        if oscale_maps is None:
            raise RuntimeError("model merging, please specify --oscale_maps")
        oscale_maps = json.loads(oscale_maps)
        oscales = model_merger.get_output_scales(mrt_oscales, oscale_maps)
        sym_all_file, prm_all_file, ext_all_file = load_fname(
            model_prefix, suffix="all.quantize", with_ext=True)
        qmodel.save(sym_all_file, prm_all_file)
        infos = [oscales, inputs_ext]
        sim.save_ext(ext_all_file, *infos)
        save_conf(conf_quant_file, logger=logger, **conf_map)
        logger.info("model merging finished")
    else:
        logger.info("model merging skipped")
