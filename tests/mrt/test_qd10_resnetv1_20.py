import gluon_zoo as gz
import mxnet as mx
from mxnet import ndarray as nd
from mxnet import gluon

import sym_pass as spass
import dataset as ds
import sym_calib as calib
import sim_quant_helper as sim
import ops_generator as opg
import utils
import mrt as _mrt

version = "20_v2"
#  gz.save_model("cifar_resnet" + version )
#  exit(0)

def load_fname(version, suffix=None, with_ext=False):
    suffix = "."+suffix if suffix is not None else ""
    #prefix = "./data/cifar_resnext29_%s%s" % (version, suffix)
    prefix = "./data/quick_raw_qd_animal10_2_cifar_resnet%s%s" % (version, suffix)
    return utils.extend_fname(prefix, with_ext=with_ext)

batch_size = 16
input_size = 28
inputs_ext = { 'data': {
    'shape': (batch_size, 1, input_size, input_size)
}}
inputs = [mx.sym.var(n) for n in inputs_ext]
calib_ctx = mx.gpu(2)
ctx = [mx.gpu(int(i)) for i in "2,3,4,5".split(',') if i.strip()]

utils.log_init()

val_data = ds.load_quickdraw10(batch_size)
data_iter = iter(val_data)
def data_iter_func():
    data, label = next(data_iter)
    return data, label
data, _ = next(data_iter)

sym_file, param_file = load_fname(version)
net1 = utils.load_model(sym_file, param_file, inputs, ctx=ctx)
acc_top1 = mx.metric.Accuracy()
acc_top5 = mx.metric.TopKAccuracy(5)
acc_top1.reset()
acc_top5.reset()
def squeezenet(data, label):
    data = gluon.utils.split_and_load(data, ctx_list=ctx, batch_axis=0, even_split=False)
    res = [net1.forward(d) for d in data]
    res = nd.concatenate(res)
    acc_top1.update(label, res)
    _, top1 = acc_top1.get()
    acc_top5.update(label, res)
    _, top5 = acc_top5.get()
    return "top1={:6.2%} top5={:6.2%}".format(top1, top5)

if True:
    sym, params = mx.sym.load(sym_file), nd.load(param_file)
    infer_shapes = (spass.sym_infer_shape(sym, params, inputs_ext))
    sym, params = spass.sym_quant_prepare(sym, params, inputs_ext)

    if True:
        mrt = _mrt.MRT(sym, params, inputs_ext)
        mrt.set_data('data', data)
        mrt.calibrate(ctx=calib_ctx)
        mrt.set_output_prec(8)
        qsym, qparams, inputs_ext = mrt.quantize()
    else:
        inputs_ext['data']['data'] = data
        th_dict = calib.sym_calibrate(sym, params, inputs_ext, ctx=calib_ctx)
        qsym, qparams, precs, _ = calib.sym_simulate(sym, params, inputs_ext, th_dict)
        qsym, qparams = calib.sym_realize(qsym, qparams, inputs_ext, precs, "cvm")
    dump_sym, dump_params, dump_ext = load_fname(version, "sym.quantize", True)
    sim.save_ext(dump_ext, inputs_ext)
    nd.save(dump_params, qparams)
    open(dump_sym, "w").write(qsym.tojson())

dump_sym, dump_params, dump_ext = load_fname(version, "sym.quantize", True)
sym, params = mx.sym.load(dump_sym), nd.load(dump_params)
(inputs_ext,) = sim.load_ext(dump_ext)

if False:
    #  data = data[0, :].reshape((1, 1, input_size, input_size))
    _mrt.std_dump(sym, params, inputs_ext, data, "animal_10",
            batch=False)
            # dump_ops=["cifarresnetv215_stage1_bn_conv2_fwd"])
    # opg.dump_file("conv2d",
    #         ["cifarresnetv215_stage1_bn_conv2_fwd_0.mrt.dump.in.npy",
    #          "cifarresnetv215_stage1_bn_conv2_fwd_1.mrt.dump.in.npy",
    #          "cifarresnetv215_stage1_bn_conv2_fwd_2.mrt.dump.in.npy"],
    #         ["cifarresnetv215_stage1_bn_conv2_fwd_0.mrt.dump.out.npy"],
    #         "cifarresnetv215_stage1_bn_conv2_fwd.attr",
    #         root="/data/std_out/qd10_resnet20_v2")
    exit()

inputs = [mx.sym.var(n) for n in inputs_ext]
net2 = utils.load_model(dump_sym, dump_params, inputs, ctx=ctx)
qacc_top1 = mx.metric.Accuracy()
qacc_top5 = mx.metric.TopKAccuracy(5)
qacc_top1.reset()
qacc_top5.reset()
def cvm_quantize(data, label):
    data = sim.load_real_data(data, 'data', inputs_ext)
    data = gluon.utils.split_and_load(data, ctx_list=ctx, batch_axis=0, even_split=False)
    res = [net2.forward(d) for d in data]
    res = nd.concatenate(res)
    qacc_top1.update(label, res)
    _, top1 = qacc_top1.get()
    qacc_top5.update(label, res)
    _, top5 = qacc_top5.get()
    return "top1={:6.2%} top5={:6.2%}".format(top1, top5)

utils.multi_validate(squeezenet, data_iter_func,
        cvm_quantize,
        iter_num=100000)
# utils.multi_eval_accuracy(nnvm_real, data_iter,
#        iter_num=10000)
