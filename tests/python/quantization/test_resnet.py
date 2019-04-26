import mxnet as mx
from mxnet import gluon
from mxnet import symbol
from mxnet.contrib import quantization as qm
from mxnet.gluon.model_zoo import vision
from mxnet import ndarray as nd
from mxnet.gluon import HybridBlock, nn
from mxnet.contrib import quantization as mquant

import tvm
from tvm.contrib import graph_runtime
import nnvm

import numpy as np
import logging
import os

from quant_op import *
from quant_utils import *
from utils import *
import quant_pass as qpass
import sym_calib as calib
import sim_quant_helper as sim

import resnet18 as resnet
# import resnet152 as resnet
#  import resnet50 as resnet

from sym_pass import *

def get_dump_fname(suffix="quant"):
    return '%s.%s'%(resnet.SYMBOL_FILE, suffix), \
        '%s.%s'%(resnet.PARAMS_FILE, suffix)

def mxnet_realize(quant_flag):
    logger = logging.getLogger("log.quant.main.mxnet")

    load_symbol_file, load_params_file = get_dump_fname("gluon.quant")

    inputs = mx.sym.var('data')
    ctx = mx.gpu(1)

    mxnet_symbol = mx.sym.load(load_symbol_file)
    params = nd.load(load_params_file)

    #  sym, params = quant_realize(mxnet_symbol, params, {}, quant_flag)

    save_symbol_file, save_params_file = get_dump_fname("post.quant")
    nd.save(save_params_file, params)
    print (params.keys())
    with open(save_symbol_file, 'w') as fout:
        fout.write(sym.tojson())

def gluon_quant_resnet(quant_flag, batch_size=10,
        iter_num=10, need_requant=False):
    logger = logging.getLogger("log.quant.main.gluon")
    logger.info("=== Model Quantazation ===")

    pass_name = "gluon.quant"
    quant_symbol_file, quant_params_file = get_dump_fname(pass_name)

    if not os.path.exists(resnet.SYMBOL_FILE):
        logger.info("save resnet symbol&params")
        resnet.save_graph(mx.gpu())

    inputs_ext = {
        'data': {
            'shape': (batch_size, 3, 224, 244),
        }
    }
    inputs = mx.sym.var('data')
    ctx = mx.cpu(0)

    logger.info("load dataset")
    data_iter = load_dataset(batch_size)
    calib_data = data_iter.next()

    logger.info("quantization model")
    tmp_params_file = quant_params_file + ".tmp"
    if (not need_requant) and os.path.exists(tmp_params_file):
        logger.debug("load quant params")
        qparams = nd.load(tmp_params_file)
    else:
        qparams = qpass.fuse_bn_parameters(nd.load(resnet.PARAMS_FILE), quant_flag)
        name_scope = "calib_"
        scope_graph = nn.HybridSequential(prefix=name_scope)
        with scope_graph.name_scope():
            graph = resnet.load_quant_graph(QuantFlag(is_fuse_bn=True,
                        calib_mode=CalibMode.NONE))
            scope_graph.add(graph)
        qparams = qpass.calibrate_parameters(scope_graph, qparams, ctx,
                calib_data.data[0], quant_flag, name_scope=name_scope)
        nd.save(tmp_params_file, qparams)

    graph = resnet.load_quant_graph(quant_flag)
    sym, qparams = graph(inputs), load_parameters(graph, qparams, ctx=ctx)

    sym, qparams = fold_cond_op(sym, qparams, {}, quant_flag)
    # sym, qparams = mx_sym_rewrite(sym, qparams, quant_flag, inputs_ext)
    # exit()

    nd.save(quant_params_file, qparams)
    with open(quant_symbol_file, 'w') as fout:
        fout.write(sym.tojson())

    logger.info("load quant/original model")
    qsym_block = nn.SymbolBlock(sym, [inputs])
    qsym_block.load_parameters(quant_params_file, ctx=ctx, ignore_extra=True)

    sym_block = resnet.load_graph(ctx)

    logger.info("calculate model accuracy")
    qacc, acc, diff, total = 0, 0, 0, 0
    for i in range(iter_num):
        image_data = calib_data.data[0]
        qimage_data, _ = quant_helper(image_data)

        res = sym_block.forward(image_data.as_in_context(ctx))

        if quant_flag.calib_mode == CalibMode.NONE:
            qimage_data = image_data
        qres = qsym_block.forward(qimage_data.as_in_context(ctx))

        assert res.shape == qres.shape
        for idx in range(res.shape[0]):
            res_label = res[idx].asnumpy().argmax()
            qres_label = qres[idx].asnumpy().argmax()
            image_label = calib_data.label[0][idx].asnumpy()

            diff += 0 if res_label == qres_label else 1
            acc += 1 if res_label == image_label else 0
            qacc += 1 if qres_label == image_label else 0
            total += 1

        try:
            calib_data = data_iter.next()
        except:
            exit()

        logger.info("Iteration: %5d | Accuracy: %.2f%% | Quant Acc: %.2f%%" +
                " | Difference: %.2f%% | Total Sample: %5d",
                i, 100.*acc/total, 100.*qacc/total, 100.*diff/total, total)

def test_quant_model(batch_size=10, iter_num=10):
    logger = logging.getLogger("log.test.mxnet")
    logger.info("=== Log Test Mxnet ===")

    load_symbol_file, load_params_file = get_dump_fname("post.quant")

    ctx = mx.gpu(1)
    inputs = mx.sym.var("data")

    sym = mx.sym.load(load_symbol_file)

    data_iter = load_dataset(batch_size)
    calib_data = data_iter.next()

    graph = nn.SymbolBlock(sym, [inputs])
    # print ('graph params:', sorted(list(graph.collect_params().keys())))
    # print ('params:', sorted(list(params.keys())))
    # params_dict = load_parameters(graph, params, ctx=ctx)

    graph.load_parameters(load_params_file, ctx=ctx)

    qacc, total = 0, 0
    for i in range(iter_num):
        qimage_data, _ = quant_helper(calib_data.data[0])

        # params['data'] = qimage_data
        # graph = sym.bind(ctx, params)
        qres = graph.forward(qimage_data.as_in_context(ctx))

        for idx in range(qres.shape[0]):
            qres_label = qres[idx].asnumpy().argmax()
            image_label = calib_data.label[0][idx].asnumpy()

            qacc += 1 if qres_label == image_label else 0
            total += 1

        try:
            calib_data = data_iter.next()
        except:
            exit()

        logger.info("Iteration: %5d | Quant Acc: %.2f%% | Total Sample: %5d",
                i, 100.*qacc/total, total)

def test_nnvm_load(batch_size=10, iter_num=10):
    logger = logging.getLogger("log.test.nnvm")
    logger.info("=== Log Test NNVM ===")

    target = "llvm -mcpu=core-avx2 -libs=cvm"
    ctx = tvm.context(target, 0)

    load_symbol_fname, load_params_fname = get_dump_fname("gluon.quant")

    in_shape = (batch_size, 3, 224, 224)
    data_iter = load_dataset(batch_size)
    calib_data = data_iter.next()

    params = nd.load(load_params_fname)

    sym = mx.sym.load(load_symbol_fname)
    nnvm_sym, _ = nnvm.frontend.from_mxnet(sym)

    nnvm_sym, params = quant_realize(nnvm_sym, params, {}, quant_flag)
    # , ctx=tvm.context("opencl", 0))

    nnvm_graph = nnvm.graph.create(nnvm_sym)
    save_symbol_file, _ = get_dump_fname("nnvm.realize")
    with open(save_symbol_file, "w") as fout:
       fout.write(nnvm_graph.ir())
    use_dtype = "int32"
    for key, value in list(params.items()):
        params[key] = tvm.nd.array(value.asnumpy().astype(use_dtype), ctx)
    with nnvm.compiler.build_config(opt_level=0):
#, add_pass=["PrecomputePrune"]):
        deploy_graph, lib, params = nnvm.compiler.build(
            nnvm_sym, target=target,
            shape={"data": in_shape},
            params=params, dtype=use_dtype)
        ret = deploy_graph.apply('SaveJSON')
        graph_str = ret.json_attr('json')

        with open("graph_str.log", "w") as fout:
            fout.write(graph_str)
        with open("deploy.log", "w") as fout:
            fout.write(deploy_graph.ir())

    module = graph_runtime.create(deploy_graph, lib, ctx)
    param_bytes = nnvm.compiler.save_param_dict(params)
    module.set_input(**params)

    qacc, total = 0, 0
    for i in range(iter_num):
        qimage_data, _ = quant_helper(calib_data.data[0])
        qimage_data = tvm.nd.array(qimage_data.asnumpy(), ctx)

        module.run(data=qimage_data.asnumpy())
        qres = module.get_output(0).asnumpy()

        for idx in range(qres.shape[0]):
            qres_label = qres[idx].argmax()
            image_label = calib_data.label[0][idx].asnumpy()

            qacc += 1 if qres_label == image_label else 0
            total += 1

        try:
            calib_data = data_iter.next()
        except:
            exit()

        logger.info("Iteration: %5d | Quant Acc: %.2f%% | Total Sample: %5d",
                i, 100.*qacc/total, total)

def test_sym_nnvm(batch_size=10, iter_num=10):
    logger = logging.getLogger("log.test.nnvm")
    logger.info("=== Log Test NNVM ===")

    target = "cuda"
    tvm_ctx = tvm.context(target, 1)
    mx_ctx = mx.gpu(2)
    inputs_ext = {
        'data': {
            'shape': (batch_size, 3, 224, 224),
        }
    }
    inputs = [mx.sym.var(name) for name in inputs_ext]
    inputs_shape = {k:v['shape'] for k,v in inputs_ext.items()}

    data_iter = load_dataset(batch_size)
    def data_iter_func():
        data = data_iter.next()
        return data.data[0], data.label[0]
    data_iter_func()

    dump_symbol, dump_params = get_dump_fname("sym.nnvm.compile")
    _, dump_lib = get_dump_fname("nnvm.so")

    load_symbol_fname, load_params_fname = get_dump_fname("sym.sim.quant")
    sym, params = mx.sym.load(load_symbol_fname), nd.load(load_params_fname)
    graph = nn.SymbolBlock(sym, inputs)
    load_parameters(graph, params, ctx=mx_ctx)
    sim.load_ins_ext(params, inputs_ext)
    def graph_func(data):
        data = sim.load_real_data(data, 'data', params)
        return graph.forward(data.as_in_context(mx_ctx))

    nnvm_sym, _ = nnvm.frontend.from_mxnet(sym)
    nnvm_sym, real_params = nnvm_realize(nnvm_sym, params, inputs_ext)

    nnvm_graph = nnvm.graph.create(nnvm_sym)
    save_symbol_file, _ = get_dump_fname("nnvm.realize")
    with open(save_symbol_file, "w") as fout:
       fout.write(nnvm_graph.json())

    use_dtype = "int32"
    for key, value in list(real_params.items()):
        real_params[key] = tvm.nd.array(value.asnumpy().astype(use_dtype), tvm_ctx)

    with nnvm.compiler.build_config(opt_level=0): #, add_pass=["PrecomputePrune"]):
        deploy_graph, lib, real_params = nnvm.compiler.build(
            nnvm_sym, target=target, shape=inputs_shape,
            params=real_params, dtype=use_dtype, runtime="tvm")
    with open(dump_symbol, "w") as fout:
        fout.write(deploy_graph.json())
    with open(dump_params, "wb") as fout:
        param_bytes = nnvm.compiler.save_param_dict(real_params)
        fout.write(param_bytes)
    lib.export_library(dump_lib)

    exit()

    module = graph_runtime.create(deploy_graph, lib, tvm_ctx)
    module.load_params(param_bytes)
    def nnvm_real(data):
        data = sim.load_quant_data(data, 'data', params)
        data = tvm.nd.array(data.asnumpy(), tvm_ctx)
        module.run(data=data.asnumpy())
        return nd.array(module.get_output(0).asnumpy())

    multi_eval_accuracy(graph_func, data_iter_func, nnvm_real,
            iter_num=iter_num, logger=logger)

def test_sym_pass(quant_flag, batch_size=10, iter_num=10):
    logger = logging.getLogger("log.test.sym.pass")

    ctx = mx.gpu(2)
    inputs_ext = {
        'data': {
            'shape': (batch_size, 3, 224, 224),
        }
    }
    inputs = [mx.sym.var(name) for name in inputs_ext]

    logger.info("load dataset, symbol and parameters")
    data_iter = load_dataset(batch_size)
    def data_iter_func():
        data = data_iter.next()
        return data.data[0], data.label[0]
    calib_data, _ = data_iter_func()

    symbol_file, params_file = resnet.SYMBOL_FILE, resnet.PARAMS_FILE
    sym, params = mx.sym.load(symbol_file), nd.load(params_file)
    sym, params = sym_quant_prepare(sym, params, inputs_ext)
    graph = nn.SymbolBlock(sym, inputs)
    load_parameters(graph, params, ctx=ctx)
    def graph_func(data):
        return graph.forward(data.as_in_context(ctx))

    qsym, qparams= calib.sym_simulate(sym,
            params, inputs_ext, calib_data, ctx)
    qsym, qparams = calib.sym_realize(qsym, qparams, inputs_ext)

    dump_sym, dump_params = get_dump_fname('sym.sim.quant')
    sim.save_ins_ext(qparams, inputs_ext)
    nd.save(dump_params, qparams)
    with open(dump_sym, 'w') as fout:
        fout.write(qsym.tojson())
    graph_sim = nn.SymbolBlock(qsym, inputs)
    load_parameters(graph_sim, qparams, ctx=ctx)
    sim.load_ins_ext(qparams, inputs_ext)
    def simulate(data):
        data = sim.load_real_data(data, 'data', inputs_ext)
        return graph_sim.forward(data.as_in_context(ctx))

    multi_eval_accuracy(graph_func, data_iter_func, simulate,
            iter_num=iter_num, logger=logger)

def save_data():
    batch_size = 1024
    data_iter = load_dataset(batch_size)
    calib_data = data_iter.next()
    x, _ = quant_helper(calib_data.data[0])
    np.save('/tmp/imagenet.x', x.asnumpy())
    np.save('/tmp/imagenet.y', calib_data.label[0].asnumpy())

if __name__ == "__main__":
    logging.basicConfig(level=logging.NOTSET)
    formatter = ColoredFormatter(
            fmt="[ %(asctime)s %(name)s.%(levelname)s ] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S")

    allows=["log.quant", "log.calib", "log.main", "log.test"]
    disables = ["log.quant.op.requant.helper", "autotvm"]

    log_filter = FilterList(
                allows=allows, disables=disables,
                # keywords=["layer=pool", "calib_pool"],
                log_level=logging.INFO,
                default=False)
    for handler in logging.root.handlers:
        handler.addFilter(log_filter)
        handler.setFormatter(formatter)

    quant_flag = QuantFlag(is_fuse_bn=True, calib_mode=CalibMode.NAIVE,
            log_level=logging.DEBUG,
            disabled_layers=["relu", "pool0", "activation"])

    # resnet.save_graph(mx.gpu())

    # enable quantization
    if False:
        gluon_quant_resnet(quant_flag, batch_size=16, iter_num=10000, need_requant=False)
    # save_data()

    # test_nnvm_load(batch_size=16, iter_num=10)
    # test_sym_pass(quant_flag, batch_size=16, iter_num=10000)
    test_sym_nnvm(batch_size=100, iter_num=10)


