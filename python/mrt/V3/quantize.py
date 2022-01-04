from yacs.config import CfgNode as CN

from mrt.transformer import Model, MRT
from mrt import sym_utils as sutils
from mrt import sim_quant_helper as sim
from mrt.V3.utils import (
    MRT_CFG, default_device_type, default_device_ids, get_model_prefix,
    get_logger, load_fname, save_conf, load_conf, check_file_existance,
    get_ctx)

DOC = """
QUANTIZE Stage Options:
    --quantize.restore_names    Names of graph nodes restored from quantization.
    --quantize.input_precision  Input precision for quantization.
    --quantize.output_precision Output precision for quantization.
    --quantize.device_type      Context type for quantization stage chosen from "cpu" or "gpu".
    --quantize.device_ids       A comma list within square brackets specifying the context ids, eg.[0,1,2].
    --quantize.softmax_lambd    Hyperparameter for requant function.
    --quantize.shift_bits       Hyperparameter for operator requantization.
    --quantize.thresholds       Initial threshold for base symbol given model split.
    --quantize.attribute_deps   Adjust the top attributes with respect to the base oscales.
    --quantize.oscale_maps      Model merger output scales name map.
"""

MRT_CFG.QUANTIZE = CN()
MRT_CFG.QUANTIZE.RESTORE_NAMES = []
MRT_CFG.QUANTIZE.INPUT_PRECISION = None
MRT_CFG.QUANTIZE.OUTPUT_PRECISION = None
MRT_CFG.QUANTIZE.DEVICE_TYPE = default_device_type
MRT_CFG.QUANTIZE.DEVICE_IDS = default_device_ids
MRT_CFG.QUANTIZE.SOFTMAX_LAMBD = None
MRT_CFG.QUANTIZE.SHIFT_BITS = None
MRT_CFG.QUANTIZE.THRESHOLDS = []
MRT_CFG.QUANTIZE.ATTRIBUTE_DEPS = []
MRT_CFG.QUANTIZE.OSCALE_MAPS = []

def quantize(cm_cfg, pass_cfg, logger=None):
    model_dir = cm_cfg.MODEL_DIR
    model_name = cm_cfg.MODEL_NAME
    verbosity = cm_cfg.VERBOSITY
    restore_names = pass_cfg.RESTORE_NAMES
    input_precision = pass_cfg.INPUT_PRECISION
    output_precision = pass_cfg.OUTPUT_PRECISION
    device_type = pass_cfg.DEVICE_TYPE
    device_ids = pass_cfg.DEVICE_IDS
    softmax_lambd = pass_cfg.SOFTMAX_LAMBD
    shift_bits = pass_cfg.SHIFT_BITS
    thresholds = {opn: th for opn, th in pass_cfg.THRESHOLDS}
    attribute_deps = {attr: {sattr: opn for sattr, opn in attr_map} \
        for attr, attr_map in pass_cfg.ATTRIBUTE_DEPS}
    oscale_maps = {opn1: opn2 for opn1, opn2 in pass_cfg.OSCALE_MAPS}

    model_prefix = get_model_prefix(model_dir, model_name)
    if logger is None:
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
    if thresholds != {}:
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
        if attribute_deps == {}:
            raise RuntimeError("model merging, please specify --attribute_deps")
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
        if oscale_maps == {}:
            raise RuntimeError("model merging, please specify --oscale_maps")
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
