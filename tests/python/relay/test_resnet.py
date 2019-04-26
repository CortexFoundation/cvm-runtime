import tvm
from tvm import relay
from tvm.relay import quantize as qtz
from tvm.relay.frontend import mxnet as tvm_mxnet

import logging
import mxnet
import mxnet as mx
import numpy as np

from mxnet.contrib import quantization as qm

import time

def make_dataset(size=100, shape=(3, 224, 224)):
    seed = int(time.time())
    print ("Seed ", seed)
    np.random.seed(seed)

    dataset = []
    label = []
    for i in range(size):
        dataset.append(np.random.randint(0, 256, size=shape, dtype='int32'))
        label.append(np.ones((1000,)))

    return mx.io.NDArrayIter(data = {"data": dataset}, label={'softmax_label': label}, batch_size=1, shuffle=True)

def load_mxnet_resnet():
    symbol_file = "/home/wlt/tvm-cvm/data/resnet-152-symbol.json"
    params_file = "/home/wlt/tvm-cvm/data/resnet-152-0000.params"

    resnet_symbol = mx.symbol.load(symbol_file)
    resnet_params = mx.nd.load(params_file)
    arg_params = {}
    aux_params = {}
    for k, v in resnet_params.items():
        tp, name = k.split(':', 1)
        if tp == 'arg':
            arg_params[name] = v
        if tp == 'aux':
            aux_params[name] = v

    input_shape = (3, 224, 224)
    output_shape = (1, 1000,)

    data_iter = make_dataset(100, input_shape)

    ctx = mx.gpu()

    # print ("=== Predict Model ===")
    # mod = mx.mod.Module(resnet_symbol, context=ctx)
    # mod.bind(for_training=False, data_shapes=[('data', input_shape)],
    #         label_shapes=[('softmax_label', output_shape)])
    # mod.set_params(arg_params, aux_params)

    # res = mod.predict(data_iter, num_batch=10).asnumpy()
    # print (res.shape)
    # res = res[0]
    # print (res.argmax(), res[res.argmax()], res.max())

    print ("=== Quantize Model ===")
    data_iter.reset()
    # print (data_iter.provide_label)
    excluded_sym_names = ['flatten0', 'fc1', 'pooling0']
    resnet_symbol = resnet_symbol.get_backend_symbol('MKLDNN') 
    qsym = qm.quantize_model(resnet_symbol, arg_params, aux_params,
            calib_data=data_iter,
            num_calib_examples=50, calib_mode='naive',
            quantized_dtype='uint8',
            excluded_sym_names = excluded_sym_names,
            label_names=["softmax_label"],
            calib_quantize_op = True,
            logger=logger,
            ctx=ctx)
    print (qsym)


if __name__ == "__main__":
    logger = logging.getLogger('logger')
    logger.setLevel(logging.INFO)
    load_mxnet_resnet()
