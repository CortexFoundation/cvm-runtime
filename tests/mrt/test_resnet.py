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

import utils
import dataset as ds
import sym_utils as sutils
import sym_pass as spass
import sym_annotate as anno
import sym_calib as calib
import sim_quant_helper as sim
import mrt as _mrt
import gluon_zoo as zoo

# import resnet18 as resnet
# import resnet152 as resnet
# import resnet50 as resnet

from sym_pass import *

def load_fname(version, suffix=None, with_ext=False):
    suffix = "."+suffix if suffix is not None else ""
    fname = "./data/resnet%s%s"%(version, suffix)
    return utils.extend_fname(fname, with_ext)

version = "18_v1b_0.89"

def test_sym_nnvm(batch_size=10):
    dump_sym, dump_params, dump_ext = load_fname(version, "sym.quantize", True)
    sym, params = mx.sym.load(dump_sym), nd.load(dump_params)
    (inputs_ext,) = sim.load_ext(dump_ext)
    data_iter = utils.load_dataset(batch_size)
    data = data_iter.next().data[0]

    _mrt.std_dump(sym, params, inputs_ext, data, "resnet"+version)

def test_sym_pass(batch_size=10, iter_num=10, quantize=True):
    logger = logging.getLogger("log.test.sym.pass")
    calib_ctx = mx.gpu(2)
    ctx = [mx.gpu(int(i)) for i in "1,2,3,4".split(',') if i.strip()]
    inputs_ext = { 'data': {
            'shape': (batch_size, 3, 224, 224),
    } }
    inputs = [mx.sym.var(name) for name in inputs_ext]

    logger.info("load dataset, symbol and parameters")
    data_iter = ds.load_imagenet_rec(batch_size)
    def data_iter_func():
        data = data_iter.next()
        return data.data[0], data.label[0]
    data, _ = data_iter_func()

    net1 = utils.load_model(*load_fname(version), inputs, ctx=ctx)
    acc_top1 = mx.metric.Accuracy()
    acc_top5 = mx.metric.TopKAccuracy(5)
    acc_top1.reset()
    acc_top5.reset()
    def resnet(data, label):
        data = gluon.utils.split_and_load(data, ctx_list=ctx, batch_axis=0, even_split=False)
        res = [net1.forward(d) for d in data]
        res = nd.concatenate(res)
        acc_top1.update(label, res)
        _, top1 = acc_top1.get()
        acc_top5.update(label, res)
        _, top5 = acc_top5.get()
        return "top1={:6.2%} top5={:6.2%}".format(top1, top5)

    sym_fname, param_fname = load_fname(version)
    sym, params = mx.sym.load(sym_fname), nd.load(param_fname)
    sym, params = spass.sym_quant_prepare(sym, params, inputs_ext)

    if quantize:
        mrt = _mrt.MRT(sym, params, inputs_ext)
        #  mrt.set_pure_int8()
        mrt.set_data('data', data)
        mrt.calibrate(ctx=calib_ctx, lambd=16)
        mrt.set_output_prec(8)
        qsym, qparams, inputs_ext = mrt.quantize()

        dump_sym, dump_params, dump_ext = load_fname(version, "sym.quantize", True)
        sim.save_ext(dump_ext, inputs_ext)
        nd.save(dump_params, qparams)
        open(dump_sym, "w").write(qsym.tojson())

    dump_sym, dump_params, dump_ext = load_fname(version, "sym.quantize", True)
    (inputs_ext,) = sim.load_ext(dump_ext)
    net3 = utils.load_model(dump_sym, dump_params, inputs, ctx=ctx)
    qacc_top1 = mx.metric.Accuracy()
    qacc_top5 = mx.metric.TopKAccuracy(5)
    qacc_top1.reset()
    qacc_top5.reset()
    def cvm_quantize(data, label):
        data = sim.load_real_data(data, 'data', inputs_ext)
        data = gluon.utils.split_and_load(data, ctx_list=ctx, batch_axis=0, even_split=False)
        res = [net3.forward(d) for d in data]
        res = nd.concatenate(res)
        qacc_top1.update(label, res)
        _, top1 = qacc_top1.get()
        qacc_top5.update(label, res)
        _, top5 = qacc_top5.get()
        return "top1={:6.2%} top5={:6.2%}".format(top1, top5)

    utils.multi_validate(resnet, data_iter_func,
            cvm_quantize,
            iter_num=iter_num, logger=logger)

if __name__ == "__main__":
    utils.log_init()

    # resnet.save_graph(mx.gpu())
    # zoo.save_model('resnet50_v1')
    # zoo.save_model('resnet18_v1')
    # zoo.save_model('resnet50_v1d_0.86')
    # zoo.save_model('resnet18_v1b_0.89')
    # zoo.save_model("resnet50_v2")
    #  exit()

    # save_data()
    if False:
        dump_sym, dump_params, dump_ext = load_fname(version, "sym.quantize", True)
        sym, params = mx.sym.load(dump_sym), nd.load(dump_params)
        (inputs_ext,) = sim.load_ext(dump_ext)
        data_iter = utils.load_dataset(1)
        while(1000):
            data = data_iter.next().data[0]
            inputs_ext['data']['data'] = sim.load_real_data(data, 'data', inputs_ext)
            spass.sym_dump_ops(sym, params, inputs_ext,
                    datadir="/data/wlt", ctx=mx.gpu(2))
        exit()

    test_sym_pass(batch_size=16, iter_num=10)
    # test_sym_pass(batch_size=160, iter_num=1000, quantize=False)
    #  test_sym_nnvm(batch_size=1)
    # test_performance(16, 10)


