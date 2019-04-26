
import mxnet as mx
from mxnet import ndarray as nd
from mxnet.gluon import nn

import tvm
from tvm.contrib import graph_runtime
import nnvm

import sym_calib as calib
import utils
import gluon_zoo as zoo
import sym_pass as spass
import sim_quant_helper as sim
import cvm_op as cvm

import logging
import numpy as np

def get_dump_fname(suffix="quant"):
    return './data/mobilenet1_0.json.%s'%suffix, \
        './data/mobilenet1_0.params.%s'%suffix

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

    data_iter = utils.load_dataset(batch_size)
    def data_iter_func():
        data = data_iter.next()
        return data.data[0], data.label[0]
    data_iter_func()

    dump_symbol, dump_params = get_dump_fname("sym.nnvm.compile")
    _, dump_lib = get_dump_fname("nnvm.so")

    load_symbol_fname, load_params_fname = get_dump_fname("sym.sim.pass")
    sym, params = mx.sym.load(load_symbol_fname), nd.load(load_params_fname)
    sim.load_ins_ext(params, inputs_ext)
    graph = nn.SymbolBlock(sym, inputs)
    utils.load_parameters(graph, params, ctx=mx_ctx)
    def graph_func(data):
        data = sim.load_real_data(data, 'data', inputs_ext)
        return graph.forward(data.as_in_context(mx_ctx))

    nnvm_sym, _ = nnvm.frontend.from_mxnet(sym)
    nnvm_sym, real_params = spass.nnvm_realize(nnvm_sym, params, inputs_ext)

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
            params=real_params, dtype=use_dtype)
    with open(dump_symbol, "w") as fout:
        fout.write(deploy_graph.json())
    with open(dump_params, "wb") as fout:
        param_bytes = nnvm.compiler.save_param_dict(real_params)
        fout.write(param_bytes)

    exit()

    module = graph_runtime.create(deploy_graph, lib, tvm_ctx)
    module.load_params(param_bytes)
    def nnvm_real(data):
        data = sim.load_real_data(data, 'data', inputs_ext)
        data = tvm.nd.array(data.asnumpy(), tvm_ctx)
        module.run(data=data.asnumpy())
        return nd.array(module.get_output(0).asnumpy())

    utils.multi_eval_accuracy(graph_func, data_iter_func, nnvm_real,
            iter_num=iter_num, logger=logger)

def test_sym_pass(batch_size=10, iter_num=10):
    logger = logging.getLogger("log.test.sym.pass")

    ctx = mx.gpu(2)
    inputs_ext = {
        'data': {
            'shape': (batch_size, 3, 224, 224),
        }
    }
    inputs = [mx.sym.var(name) for name in inputs_ext]

    logger.info("load dataset, symbol and parameters")
    data_iter = utils.load_dataset(batch_size)
    def data_iter_func():
        data = data_iter.next()
        return data.data[0], data.label[0]
    data, _ = data_iter_func()

    symbol_file, params_file = "./data/mobilenet1_0.json", "./data/mobilenet1_0.params"
    sym, params = mx.sym.load(symbol_file), nd.load(params_file)
    sym, params = spass.sym_quant_prepare(sym, params, inputs_ext)
    dump_sym, dump_params = get_dump_fname('sym.sim.prepare')
    nd.save(dump_params, params)
    with open(dump_sym, 'w') as fout:
       fout.write(sym.tojson())
    graph_comp = nn.SymbolBlock(sym, inputs)
    utils.load_parameters(graph_comp, params, ctx=ctx)
    def graph_func(data):
        return graph_comp.forward(data.as_in_context(ctx))

    qsym, qparams= calib.sym_simulate(sym,
            params, inputs_ext, data, ctx)
    qsym, qparams = calib.sym_realize(qsym, qparams, inputs_ext)

    sim.save_ins_ext(qparams, inputs_ext)
    dump_sym, dump_params = get_dump_fname('sym.sim.pass')
    nd.save(dump_params, qparams)
    with open(dump_sym, 'w') as fout:
       fout.write(qsym.tojson())
    qsym, qparams = mx.sym.load(dump_sym), nd.load(dump_params)
    sim.load_ins_ext(qparams, inputs_ext)
    qgraph = nn.SymbolBlock(qsym, inputs)
    utils.load_parameters(qgraph, qparams, ctx=ctx)
    def simulate(data):
        #  data = sim.load_sim_data(data, 'data', inputs_ext)
        data = sim.load_real_data(data, 'data', inputs_ext)
        return qgraph.forward(data.as_in_context(ctx))

    utils.multi_eval_accuracy(graph_func, data_iter_func, simulate,
            iter_num=iter_num, logger=logger)

if __name__ == '__main__':
    utils.log_init()

    # zoo.save_mobilenet1_0()
    # test_sym_pass(16, 10)
    test_sym_nnvm(1, 100)
