import mxnet as mx
from mxnet.gluon import nn, HybridBlock
from mxnet import ndarray as nd

import tvm
from tvm.contrib import graph_runtime, util
import nnvm

import logging

from quant_utils import *
from quant_op import *
import quant_pass as qpass
from sym_pass import *
import sym_calib as calib
import sim_quant_helper as sim
from utils import *


class SimpleNet(HybridBlock):
    def __init__(self, quant_flag, **kwargs):
        super(SimpleNet, self).__init__(**kwargs)

        self.forward = nn.HybridSequential(prefix='')

        self.forward.add(nn.Conv2D(128, kernel_size=3, strides=1,
            padding=1, use_bias=False))
        requant_helper(self.forward, quant_flag)

        self.forward.add(nn.Activation('relu'))

        # self.forward.add(nn.Dense(10, use_bias=True, prefix='fc0_'))
        self.forward.add(nn.Flatten())
        self.forward.add(Dense(quant_flag))

    def hybrid_forward(self, F, x):
        x = self.forward(x)
        return x

def get_dump_fname(suffix="quant"):
    return './data/simplenet.json.%s'%suffix, \
        './data/simplenet.params.%s'%suffix

def data_xform(data):
    """Move channel axis to the beginning, cast to float32, and normalize to [0, 1]."""
    return nd.moveaxis(data, 2, 0).astype('float32') / 255

def load_dataset(batch_size=10):
    data = mx.gluon.data.vision.MNIST(train=False).transform_first(data_xform)
    loader = mx.gluon.data.DataLoader(data, shuffle=False, batch_size=batch_size)

    return iter(loader)

def test_load_simplenet(quant_flag, batch_size=10, iter_num=10):
    logger = logging.getLogger("log.test.main")

    ctx = mx.gpu(2)
    inputs_ext = {
        'data': {
            'shape': (batch_size, 1, 28, 28),
        }
    }
    inputs = [mx.sym.var(name) for name in inputs_ext]

    logger.info("load dataset, symbol and parameters")
    data_iter = load_dataset(batch_size)
    data, label = next(data_iter)

    symbol_file, params_file = "./data/simplenet.json", "./data/simplenet.params"
    sym = mx.sym.load(symbol_file)
    qparams = nd.load(params_file)

    logger.info("quantization")
    scope_graph = nn.HybridSequential(prefix='calib_')
    with scope_graph.name_scope():
        graph = SimpleNet(QuantFlag(is_fuse_bn=True, calib_mode=CalibMode.NONE,
                    matrix_decomposition=True))
        scope_graph.add(graph)

    qparams = qpass.matrix_decomposition(qparams, quant_flag)
    qparams = qpass.calibrate_parameters(scope_graph, qparams, ctx,
            data, quant_flag, name_scope='calib_')

    graph= SimpleNet(quant_flag)
    print ("SSSSSS", graph.collect_params().keys(), "\n", qparams.keys())
    qsym, qparams = graph(inputs[0]), load_parameters(graph, qparams, ctx=ctx)
    qsym, qparams = fold_cond_op(qsym, qparams, {}, quant_flag)

    dump_sym, dump_params = get_dump_fname("matrix")
    nd.save(dump_params, qparams)
    with open(dump_sym, 'w') as fout:
        fout.write(qsym.tojson())

    logger.info("load model&quant_model")
    mx_graph = nn.SymbolBlock(sym, inputs)
    load_parameters(mx_graph, nd.load(params_file), ctx=ctx, prefix='calib_')

    qgraph = nn.SymbolBlock(qsym, inputs)
    load_parameters(qgraph, qparams, ctx=ctx)

    def graph_func(data):
        quant_data, _ = quant_helper(data)
        return qgraph.forward(quant_data.as_in_context(ctx))
    def graph_comp_func(data):
        return mx_graph.forward(data.as_in_context(ctx))
    def data_iter_func():
        try:
            return next(data_iter)
        except:
            exit()

    logger.info("calculate model accuracy")
    eval_accuracy(graph_func, data_iter_func, iter_num,
            graph_comp_func, logger)

def test_nnvm_load(quant_flag, batch_size=10, iter_num=10):
    logger = logging.getLogger("log.test.nnvm")
    logger.info("=== Log Test NNVM ===")

    target = "cuda"
    ctx = tvm.context(target, 1)
    in_shape= (batch_size, 1, 28, 28)

    data_iter = load_dataset(batch_size)
    def data_iter_func():
        return next(data_iter)
    data_iter_func()

    dump_sym, dump_params = get_dump_fname("sym.quant")
    sym, params = mx.sym.load(dump_sym), nd.load(dump_params)
    logger.info("Mxnet graph operators: %s", sym_collect_attr(sym))

    nnvm_sym, _ = nnvm.frontend.from_mxnet(sym)
    nnvm_sym, params = nnvm_realize(nnvm_sym, params, ['data'])
    logger.info("NNVM  graph operators: %s", sym_collect_attr(nnvm_sym))


    nnvm_graph = nnvm.graph.create(nnvm_sym)
    tmp_sym, tmp_params = get_dump_fname("nnvm.realize")
    with open(tmp_sym, 'w') as fout:
        fout.write(nnvm_graph.json())

    use_dtype = "int32"
    for key, value in list(params.items()):
        params[key] = tvm.nd.array(value.asnumpy().astype(use_dtype), ctx)

    with nnvm.compiler.build_config(opt_level=0): #, add_pass=["PrecomputePrune"]):
        deploy_graph, lib, params = nnvm.compiler.build(
            nnvm_sym, target=target, shape={"data": in_shape},
            params=params, dtype=use_dtype)

    param_bytes = nnvm.compiler.save_param_dict(params)

    module = graph_runtime.create(deploy_graph, lib, ctx)
    module.load_params(param_bytes)
    def graph_func(data):
        data, _ = sim.nd_quant(data, target_bit=8, logger=None)
        data = tvm.nd.array(data.asnumpy(), ctx)
        module.run(data=data.asnumpy())
        return nd.array(module.get_output(0).asnumpy())

    eval_accuracy(graph_func, data_iter_func, iter_num, logger=logger)

def test_sym_pass(quant_flag, batch_size=10, iter_num=10):
    logger = logging.getLogger("log.test.sym.pass")

    ctx = mx.gpu(2)
    inputs_ext = {
        'data': {
            'shape': (batch_size, 1, 28, 28),
        }
    }
    inputs = [mx.sym.var(name) for name in inputs_ext]

    logger.info("load dataset, symbol and parameters")
    data_iter = load_dataset(batch_size)
    def data_iter_func():
        return next(data_iter)

    symbol_file, params_file = "./data/simplenet.json", "./data/simplenet.params"
    sym = mx.sym.load(symbol_file)

    # quantization
    qsym, qparams = sym_quant_prepare(sym, nd.load(params_file), inputs_ext)
    dump_sym, dump_params = get_dump_fname('sym.pass')
    nd.save(dump_params, qparams)
    with open(dump_sym, 'w') as fout:
        fout.write(qsym.tojson())

    graph_comp = nn.SymbolBlock(qsym, inputs)
    load_parameters(graph_comp, qparams, ctx=ctx)
    def graph_comp_func(data):
        return graph_comp.forward(data.as_in_context(ctx))

    calib_data, _ = data_iter_func()
    qsym, qparams, inputs_sb = calib.sym_calib_quant(qsym,
            qparams, inputs_ext, calib_data, ctx)
    dump_sym, dump_params = get_dump_fname('sym.quant')
    nd.save(dump_params, qparams)
    with open(dump_sym, 'w') as fout:
        fout.write(qsym.tojson())

    graph = nn.SymbolBlock(qsym, inputs)
    load_parameters(graph, qparams, ctx=ctx)
    def graph_func(data):
        data, _ = sim.nd_quant(data, shift_bits=inputs_sb['data'],
                target_bit=8, logger=None)
        return graph.forward(data.as_in_context(ctx))

    eval_accuracy(graph_func, data_iter_func, iter_num,
            graph_comp_func, logger)



if __name__ == "__main__":
    logging.basicConfig(level=logging.NOTSET)

    formatter = ColoredFormatter(
            fmt="[ %(asctime)s %(name)s.%(levelname)s ] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S")

    allows=["log.quant", "log.calib", "log.main", "log.test"]
    disables = ["log.quant.op.requant.helper.ddd", "autotvm"]

    log_filter = FilterList(
                allows=allows, disables=disables,
                # keywords=["layer=pool", "calib_pool"],
                log_level=logging.INFO,
                default=False)
    for handler in logging.root.handlers:
        handler.addFilter(log_filter)
        handler.setFormatter(formatter)

    quant_flag = QuantFlag(is_fuse_bn=True, calib_mode=CalibMode.NAIVE,
            log_level=logging.DEBUG, use_scalar=False,
            matrix_decomposition=True,
            disabled_layers=["relu", "pool0", "activation"])

    # test_load_simplenet(quant_flag, batch_size=10, iter_num=10)
    test_sym_pass(quant_flag, batch_size=10, iter_num=10)
    test_nnvm_load(quant_flag, batch_size=10, iter_num=10)
