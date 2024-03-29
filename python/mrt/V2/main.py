import sys
from os import path
import configparser
import logging
import numpy as np

import mxnet as mx
from mxnet import gluon, ndarray as nd

from mrt import conf
from mrt.gluon_zoo import save_model
from mrt.main2 import set_batch, batch_axis, _check, \
                      _get_path, _get_ctx, str_t, int_t, \
                      bool_t, tuple_t, float_t, ARRAY, \
                      PAIR, _get_val, _cast_val, \
                      _load_fname, _checkpoint_exist
from mrt.V2.transformer import MRT, Model, reduce_graph
from mrt.V2.tfm_pass import deserialize

from mrt import dataset as ds
from mrt import sim_quant_helper as sim
from mrt import utils
from mrt import sym_utils as sutils

if __name__ == "__main__":
    assert len(sys.argv) == 2, "Please enter 2 python arguments."
    cfgPath = sys.argv[1]
    baseDir = path.abspath(path.dirname(cfgPath))
    fileName = path.basename(cfgPath)
    absCfgPath = path.join(baseDir, fileName)

    cfg = configparser.ConfigParser()
    cfg.read(absCfgPath)

    # default
    sec = 'DEFAULT'
    verbosity = _get_val(cfg, sec, 'Verbosity',
                         dtype=int_t, dval=logging.NOTSET)
    utils.log_init(level=verbosity)
    logger = logging.getLogger("log.main")
    default_dir = conf.MRT_MODEL_ROOT
    model_dir = _get_val(cfg, sec, 'Model_dir', dval=default_dir)
    assert path.exists(model_dir), \
        "Please create the folder `data` first"
    model_name = _get_val(cfg, sec, 'Model_name')
    model_prefix = path.join(model_dir, model_name)
    model_ctx = _get_ctx(cfg, sec)
    input_shape = _get_val(cfg, sec, 'Input_shape', dtype=tuple_t)
    start_pos = {'DEFAULT': 0, 'PREPARE': 1, 'SPLIT_MODEL': 2, \
                 'CALIBRATION': 3, 'QUANTIZATION': 4, \
                 'MERGE_MODEL': 5}
    start = _get_val(cfg, sec, 'Start', dtype=str_t, dval='DEFAULT')
    _check(start in start_pos.keys(), sec, 'Start',
           message="Please choose a value from `%s`" % start_pos.keys())
    start_point = start_pos[start]

    # prepare
    sec = 'PREPARE'
    sym_file, prm_file = _load_fname(model_prefix, suffix='prepare')
    sym_path, prm_path = _load_fname(model_prefix)
    if not path.exists(sym_path) or not path.exists(prm_path):
        save_model(model_name, data_dir=model_dir)
        # save_model(model_name, sym_path=sym_path, prm_path=prm_path)

    if start_point < 1:
        model = Model.load(sym_path, prm_path)
        model.prepare(set_batch(input_shape, 1))
        dump = _get_val(cfg, sec, 'Dump', dtype=bool_t, dval=False)
        if dump:
            model.save(sym_file, prm_file)
        logger.info("`%s` stage finihed" % sec)
    elif start_point == 1:
        _check(path.exists(sym_file) and path.exists(prm_file), 'DEFAULT',
               'Start', message="Check point of `%s` not found, " % sec + \
               "please move the start point earlier")
        model = Model.load(sym_file, prm_file)
        logger.info("`%s` stage checked" % sec)

    # split model
    sec = 'SPLIT_MODEL'
    keys = _get_val(cfg, sec, 'Keys', dtype=ARRAY(str_t), dval='')
    sym_top_file, prm_top_file = _load_fname(model_prefix, suffix='top')
    sym_base_file, prm_base_file = _load_fname(model_prefix, suffix='base')
    if keys == '':
        _check(start_point != 2, 'DEFAULT', 'Start',
               message="Invalid start point")
        if start_point <= 1:
            logger.info("`%s` stage skipped" % sec)
    elif start_point < 2:
        base, top = model.split(keys)
        dump = _get_val(cfg, sec, 'Dump', dtype=bool_t, dval=False)
        if dump:
            top.save(sym_top_file, prm_top_file)
            base.save(sym_base_file, prm_base_file)
        logger.info("`%s` stage finished" % sec)
    elif start_point == 2:
        _checkpoint_exist(
            sec, *[sym_top_file, prm_top_file,
            sym_base_file, prm_base_file])
        top = Model.load(sym_top_file, prm_top_file)
        base = Model.load(sym_base_file, prm_base_file)
        logger.info("`%s` stage checked" % sec)

    # calibration
    sec = 'CALIBRATION'
    model_name_calib = model_name + '.mrt.calibrate'
    batch = _get_val(cfg, sec, 'Batch', dtype=int_t, dval=16)
    ds_name = _get_val(cfg, sec, 'Dataset')
    dataset_dir = _get_val(cfg, sec, 'Dataset_dir', dval=conf.MRT_DATASET_ROOT)
    if start_point < 3:
        mrt = model.get_mrt() if keys == '' else base.get_mrt()
        calibrate_num = _get_val(
            cfg, sec, 'Calibrate_num', dtype=int_t, dval=1)
        cfg_groups = _get_val(
            cfg, sec, 'Cfg_groups',
            dtype=PAIR(str_t, str_t, str_t), dval={})
        cfg_dict = deserialize(cfg_groups)
        mrt.set_cfg_dict(cfg_dict)
        shp = set_batch(input_shape, batch)
        dataset = ds.DS_REG[ds_name](shp, root=dataset_dir)
        data_iter_func = dataset.iter_func()
        ctx = _get_ctx(cfg, sec, dctx=model_ctx)
        for i in range(calibrate_num):
            data, _ = data_iter_func()
            mrt.set_data(data)
            mrt.calibrate(ctx=ctx)
        dump = _get_val(cfg, sec, 'Dump', dtype=bool_t, dval=False)
        if dump:
            mrt.save(model_name_calib, datadir=model_dir)
        logger.info("`%s` stage finished" % sec)
    elif start_point == 3:
        _checkpoint_exist(
            sec, *list(utils.extend_fname(
            model_prefix+".mrt.calibrate", with_ext=True)))
        mrt = MRT.load(model_name_calib, datadir=model_dir)
        if keys != "":
            _checkpoint_exist(sec, sym_top_file, prm_top_file)
            top = Model.load(sym_top_file, prm_top_file)
        logger.info("`%s` stage checkd" % sec)

    # quantization
    sec = 'QUANTIZATION'
    model_name_quant = model_name + '.mrt.quantize'
    if start_point < 4:
        restore_names = _get_val(
            cfg, sec, 'Restore_name', dtype=ARRAY(str_t), dval=[])
        name_to_op = {}
        from mrt.sym_utils import topo_sort
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
            from mrt.V2.tfm_base import _pass_manager
            from mrt.tfm_ops import disabled_restore_ops

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
        input_precision = _get_val(
            cfg, sec, 'Input_precision', dtype=int_t, dval=None)
        if input_precision is not None:
            mrt.set_input_prec(input_precision)
        output_precision = _get_val(
            cfg, sec, 'Output_precision', dtype=int_t, dval=None)
        if output_precision is not None:
            mrt.set_output_prec(output_precision)
        ctx = _get_ctx(cfg, sec, dctx=model_ctx)
        softmax_lambd = _get_val(
            cfg, sec, 'Softmax_lambd', dtype=float_t, dval=None)
        if softmax_lambd is not None:
            mrt.set_softmax_lambd(softmax_lambd)
        shift_bits = _get_val(
            cfg, sec, 'Shift_bits', dtype=int_t, dval=None)
        if shift_bits is not None:
            mrt.set_shift_bits(shift_bits)
        thresholds = _get_val(
            cfg, sec, 'Thresholds', dtype=PAIR(str_t, float_t), dval=None)
        if thresholds is not None:
            for name, threshold in thresholds.items():
                mrt.set_threshold(name, threshold)
        mrt.quantize()
        inputs_ext = mrt.get_inputs_ext()
        dump = _get_val(cfg, sec, 'Dump', dtype=bool_t, dval=False)
        if dump:
            mrt.save(model_name_quant, datadir=model_dir)
            oscales = mrt.get_output_scales()
            inputs_ext = mrt.get_inputs_ext()
            infos = ['oscales: ', oscales,
                     'input_ext: ', inputs_ext,
                     'input shapes: ', input_shape]
            ext_all_file = path.join(model_dir, model_name+".all.quantize.ext")
            sim.save_ext(ext_all_file, *infos)
        logger.info("`%s` stage finished" % sec)
    elif start_point == 4:
        _checkpoint_exist(
            sec, *list(utils.extend_fname(
            model_prefix+'.mrt.quantize', with_ext=True)))
        mrt = MRT.load(model_name_quant, datadir=model_dir)
        inputs_ext = mrt.get_inputs_ext()
        dump = _get_val(cfg, sec, 'Dump', dtype=bool_t, dval=False)
        if keys != "":
            _checkpoint_exist(sec, sym_top_file, prm_top_file)
            top = Model.load(sym_top_file, prm_top_file)
        logger.info("`%s` stage checkd" % sec)

    # merge_model
    sec = 'MERGE_MODEL'
    sym_all_file, prm_all_file, ext_all_file = _load_fname(
        model_prefix, suffix='all.quantize', with_ext=True)
    if keys == '':
        _check(start_point != 5, 'DEFAULT', 'Start',
               message="Invalid start point")
        qmodel = mrt.current_model
        oscales = mrt.get_output_scales()
        logger.info("`%s` stage skipped" % sec)
    elif start_point < 5:
        qmodel = mrt.current_model
        mrt_oscales = mrt.get_output_scales()
        model_merger = Model.merger(qmodel, top, mrt.get_maps())
        attribute_deps = _get_val(
            cfg, sec, 'Attribute_deps', dtype=PAIR(str_t, str_t, str_t))

        name_idx = {mrt.get_maps().get(
            s.attr("name"), s.attr("name")): i \
            for i, s in enumerate(qmodel.symbol)}
        def mergefunc(node, params, graph):
            name, op_name = node.attr('name'), node.attr('op_name')
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

        qmodel = model_merger.merge(callback=mergefunc)
        oscale_maps = _get_val(
            cfg, sec, 'Oscale_maps', dtype=PAIR(str_t, str_t))
        oscales = model_merger.get_output_scales(
            mrt_oscales, oscale_maps)
        inputs_ext = mrt.get_inputs_ext()
        dump = _get_val(cfg, sec, 'Dump', dtype=bool_t, dval=False)
        if dump:
            qmodel.save(sym_all_file, prm_all_file)
            infos = ['oscales: ', oscales,
                     'input_ext: ', inputs_ext,
                     'input shapes: ', input_shape]
            sim.save_ext(ext_all_file, *infos)
        logger.info("`%s` stage finished" % sec)
    else:
        _check(start_point == 5, 'DEFAULT', 'Start',
               message='Start_point invalid')
        qmodel = Model.load(sym_all_file, prm_all_file)
        _, oscales, _, inputs_ext, _, _ = sim.load_ext(ext_all_file)
        logger.info("`%s` stage checked" % sec)

    # evaluation
    sec = 'EVALUATION'
    if sec in cfg.sections():
        iter_num = _get_val(cfg, sec, 'Iter_num', dtype=int_t, dval=0)
        batch = _get_val(cfg, sec, 'Batch', dtype=int_t, dval=batch)
        ctx = _get_ctx(cfg, sec, dctx=model_ctx)
        if isinstance(ctx, mx.Context):
            ctx = [ctx]
        org_model = Model.load(sym_path, prm_path)
        graph = org_model.to_graph(ctx=ctx)
        dataset = ds.DS_REG[ds_name](set_batch(input_shape, batch))
        data_iter_func = dataset.iter_func()
        metric = dataset.metrics()

        baxis = batch_axis(input_shape)
        olen = len(org_model.symbol)
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

        ngpus = len(ctx)
        _check(
            not batch % ngpus, sec, 'Device_ids',
            'Batch must be divisible by the number of gpus')
        split_batch = batch//ngpus
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

        if iter_num > 0:
            logger.info("Validating...")
            utils.multi_validate(evalfunc, data_iter_func, quantize,
                                 iter_num=iter_num,
                                 logger=logging.getLogger('mrt.validate'),
                                 batch_size=batch)
            logger.info("`%s` stage finished" % sec)

    # compilation
    sec = 'COMPILATION'
    if sec in cfg.sections():
        dump_dir = _get_path(
            cfg, sec, 'Dump_dir', is_dir=True, dpath=model_dir)
        batch = _get_val(cfg, sec, 'Batch', dtype=int_t, dval=batch)
        device_type = _get_val(cfg, sec, 'Device_type', dval='cpu')
        device_ids = _get_val(
            cfg, sec, 'Device_ids',
            dtype=ARRAY(int_t), dval=0)
        model_name_tfm = model_name + "_cvm"
        qmodel.to_cvm(model_name_tfm, datadir=dump_dir,
                      input_shape=set_batch(input_shape, batch),
                      target=device_type, device_ids=device_ids)

        dataset = ds.DS_REG[ds_name](set_batch(input_shape, batch))
        dump_data, _ = dataset.iter_func()()
        dump_data = sim.load_real_data(
            dump_data.astype("float64"), 'data', mrt.get_inputs_ext())
        model_root = path.join(dump_dir, model_name_tfm)
        np.save(path.join(model_root, "data.npy"),
                dump_data.astype('int8').asnumpy())
        infos = {
            "inputs_ext": inputs_ext,
            "oscales": oscales,
            "input_shapes": input_shape,
        }
        sim.save_ext(path.join(model_root, "ext"), infos)
        logger.info("`%s` stage finished" % sec)

