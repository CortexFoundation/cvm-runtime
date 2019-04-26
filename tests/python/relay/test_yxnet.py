import tvm
from tvm import relay
from tvm.relay.build_module import BuildConfig
from tvm.relay.backend import graph_runtime_codegen as _graph_gen
import nnvm

from functools import reduce
import numpy as np
import math
import struct
import inspect

shift_bits_dict = {
    # 'cv0_sb': relay.const(8, dtype='int32')
}

def quantize(data, shift_bits, target_bits=relay.const(7, dtype='int32')):
    """Quantize output of layer, to be consistent with source code @yx

    Question: should the shift_bits participating to network control flow?
            At mxnet quantization with truman's code, the bits number of max_v
            is converted to normal interger using function `asscalar()`. However,
            I cannot find the related function in relay.
            I am confused with the control flow logic in model network, whether
            the condition `shift_bits == -1` should join in model network or just
            left it in python code flow. By Longtao.Wang

    Parameters
    ----------
    shift_bits: tvm.relay.Expr
        The shift_bits parameter is never used according to @yx's source code,
        which always be constant Expr(-1).
    """
    max_v = relay.max(relay.abs(data))
    min_v = relay.min(data)

    ln_max_v = relay.log(relay.cast(max_v, 'float32'))
    ln_2 = relay.log(relay.const(2.))
    total_bits = relay.ceil(relay.divide(ln_max_v, ln_2)) # ceil( ln(max_v) / ln(2) )
    shift_bits = relay.subtract(total_bits.astype('int32'), target_bits)
    shift_bits = relay.maximum(shift_bits, relay.const(0))

    denominator = relay.left_shift(relay.const(1),
            relay.cast(shift_bits, 'int32'))
    out = relay.divide(data, denominator)
    # According to @yx's code, use divide operation instead of shift op for
    # possible negative number round.
    # out = relay.right_shift(data, shift_bits)

    out = relay.cast(relay.clip(out, a_min=-128, a_max=127), 'int8')
    return out, max_v, min_v, shift_bits


def make_conv_relu(data, kernel_size, padding, strides, channels, prefix="conv", skip_relu=False):
    prefix = "_conv_" + prefix
    weight = relay.var(prefix+"_weight", dtype="int8")
    out = relay.nn.conv2d(data, weight, kernel_size=kernel_size,
                          padding=padding, strides=strides,
                          channels=channels, out_dtype="int32")

    bias = relay.var(prefix+"_bias", dtype="int32")
    out = relay.nn.bias_add(out, bias)

    global shift_bits_dict
    sb_name = prefix + "_sb"
    shift_bits = shift_bits_dict[sb_name] if sb_name in shift_bits_dict else relay.const(-1)

    out, max_v, min_v, shift_bits_dict[sb_name] = quantize(out, shift_bits)
    if not skip_relu:
        out = relay.nn.relu(out)

    return out, max_v, min_v, shift_bits_dict[sb_name]

def make_max_pool(data):
    out = relay.nn.max_pool2d(data, pool_size=(2, 2), strides=(2, 2))
    return out

def make_dense(data, units, prefix="dense"):
    prefix = "_dense_" + prefix
    weight = relay.var(prefix+"_weight", dtype="int32")
    # need to specify out type as int32
    out = relay.nn.dense(data.astype('int32'), weight, units)

    bias = relay.var(prefix+"_bias", dtype="int32")
    out = relay.nn.bias_add(out, bias)

    global shift_bits_dict
    sb_name = prefix + "_sb"
    shift_bits = shift_bits_dict[sb_name] if sb_name in shift_bits_dict else relay.const(-1)
    out, max_v, min_v, shift_bits_dict[sb_name] = quantize(out, shift_bits)

    return out, max_v, min_v, shift_bits_dict[sb_name]

def make_mnist_graph():
    data = relay.var("data", relay.TensorType((1, 1, 28, 28), "int8"))
    out, _, _, sb0 = make_conv_relu(data, (3, 3), (1, 1), (1, 1), 32, "cv0")
    out, max_v, min_v, sb1 = make_conv_relu(out, (3, 3), (1, 1), (1, 1), 32, "cv1")

    mp = make_max_pool(out)
    out, _,_,_ = make_conv_relu(mp, (1, 1), (0, 0), (1, 1), 32, "cv2")
    out, _,_,_ = make_conv_relu(out, (3, 3), (1, 1), (1, 1), 32, "cv3")

    out = relay.add(relay.divide(out, relay.const(2, dtype='int8')),
       relay.divide(mp, relay.const(2, dtype='int8'))) # shortcut layer

    out = make_max_pool(out)

    out = relay.nn.batch_flatten(out).astype('int8')
    out, _, _, _ = make_dense(out, 256, "dense0")
    out = relay.nn.relu(out)
    out, max_v, min_v, sb = make_dense(out, 10, "dense1")

    print ("Free vars: ", relay.ir_pass.free_vars(out))
    out = relay.Function(relay.ir_pass.free_vars(out), out)
    return out

def make_dog_cat_graph():
    data = relay.var("data", relay.TensorType((1, 3, 224, 224), "int8"))
    out, _,_,_ = make_conv_relu(data, (3, 3), (1, 1), (1, 1), 64, "cv0")
    out, _,_,_ = make_conv_relu(out, (3, 3), (1, 1), (1, 1), 64, "cv1")
    out = make_max_pool(out)

    out, _,_,_ = make_conv_relu(out, (3, 3), (1, 1), (1, 1), 128, "cv2")
    out, _,_,_ = make_conv_relu(out, (3, 3), (1, 1), (1, 1), 128, "cv3")
    out = make_max_pool(out)

    out, _,_,_ = make_conv_relu(out, (3, 3), (1, 1), (1, 1), 256, "cv4")
    out, _,_,_ = make_conv_relu(out, (3, 3), (1, 1), (1, 1), 256, "cv5")
    out, _,_,_ = make_conv_relu(out, (3, 3), (1, 1), (1, 1), 256, "cv6")
    out = make_max_pool(out)

    out, _,_,_ = make_conv_relu(out, (3, 3), (1, 1), (1, 1), 512, "cv7")
    out, _,_,_ = make_conv_relu(out, (3, 3), (1, 1), (1, 1), 512, "cv8")
    out, _,_,_ = make_conv_relu(out, (3, 3), (1, 1), (1, 1), 512, "cv9")
    out = make_max_pool(out)

    out, _,_,_ = make_conv_relu(out, (3, 3), (1, 1), (1, 1), 512, "cv10")
    out, _,_,_ = make_conv_relu(out, (3, 3), (1, 1), (1, 1), 512, "cv11")
    out, _,_,_ = make_conv_relu(out, (3, 3), (1, 1), (1, 1), 512, "cv12")
    out = make_max_pool(out)

    out = relay.nn.batch_flatten(out).astype('int8')
    out, _,_,_ = make_dense(out, 1, "dense0")

    out = relay.Function(relay.ir_pass.free_vars(out), out)
    return out

def load_parameters(graph, params_name):
    args = relay.ir_pass.infer_type(graph).params
    print ('args = ', args)

    # Load int8 parameters from params file
    data = np.fromfile(params_name, dtype=np.int8)
    count = 0
    bind_dict = {}
    params_list = {}
    for idx, arg in enumerate(args):
        if arg.name_hint == 'data':
            pass
        else:
            # Weight size is batch*channel*size*size
            # Bias size is batch
            shape = arg.checked_type.shape
            # print ('shape = ', shape, arg, idx)
            assert len(shape) > 0, "parameter size should be 1 at least: " + str(arg)
            size = int(reduce((lambda x, y: x * y), shape))
            print (arg.name_hint, ": ", size, data[count:count+2], shape)

            # Please refer to source code @yx for params shape details.
            # Current data format is (batch, channel, size, size)'s tensor.
            params = np.array(data[count:count+size], dtype='int8') \
                    .reshape([int(x) for x in shape])

            # Within @yx's code, weight params at dense layer is convert(weight),
            # where convert() is:
            #   weight = weight.reshape(reversed(weight.shape)).transpose()
            #
            # According to code at file `infernet/src/int_connected_layer.c`,
            # the dense layer was implemented falsely with expression as below:
            #   Y = X * W + B, instead of correctly format Y = W * X + B
            # which Y is output, X is input, W is weight and B is bias.
            # For more details of matrix computation in c source code, please refer to
            # file `infernet/src/trivial_mul_kernels.cu`, which leads to weight params
            # in python code should be transformed with convert() function.
            if arg.name_hint.startswith("_dense_"):
                params = params.reshape(list(reversed(list(params.shape)))).transpose()
                # print ('dense transpose = ', params.flatten()[0:2])

            if arg.name_hint.endswith("_bias"):
                params = params.astype("int32")

            if arg.name_hint.startswith("_dense_") and arg.name_hint.endswith("_weight"):
                params = params.astype("int32")

            params_list[arg.name_hint] = params
            bind_dict[arg] = relay.const(params)
            count += size

    # print (bind_dict.keys())
    graph = relay.expr.bind(graph, bind_dict)

    print ("Parameters length: ", len(data), count)
    return graph, params_list

def test_yxnet_mnist():
    graph = make_mnist_graph()
    _, bd = load_parameters(graph,
            "/home/wlt/warehouse/.tmp/ca3d0286d5758697cdef653c1375960a868ac08a/data/params")

    with relay.build_config(opt_level=0):
        func = graph
        func = relay.ir_pass.infer_type(func)
        func = relay.ir_pass.fuse_ops(func, 0)
    #  print (relay.Function(relay.ir_pass.free_vars(func), func))
        func = relay.ir_pass.infer_type(func)
        graph_gen = _graph_gen.GraphRuntimeCodegen(mod=None, target='llvm')
        graph_json, lowered_funcs, params = graph_gen.codegen(func)

        dump_sym = './data/yxnet_mnist.symbol'
        dump_params = './data/yxnet_mnist.params'
        with open(dump_sym, 'w') as fout:
            fout.write(graph_json)
        with open(dump_params, "wb") as fo:
            fo.write(relay.save_param_dict(params))

    data = np.load('data.npy')
    executor = relay.create_executor()

    res = executor.evaluate(graph)([data.astype(np.int8)], **bd).asnumpy()
    np.save('/tmp/relay.res', res)
    print (res.flatten()[:100])

def test_yxnet_dog_cat():
    graph = make_dog_cat_graph()
    _, bd = load_parameters(graph,
            "/home/wlt/warehouse/.tmp/4d8bc8272b882f315c6a96449ad4568fac0e6038/data/params")

    data = np.load('/home/wlt/warehouse/.tmp/9f8a5cce5dc2e1e18944512c2dcd796b940dd23b/data')
    executor = relay.create_executor('graph')

    res = executor.evaluate(graph)([data.astype(np.int8)], **bd)
    print (res.asnumpy())

def test_naive():
    data = relay.var("data", relay.TensorType((1, 3, 224, 224), "int8"))
    out = data
    prefix = "_conv_" + 'cv0'
    weight = relay.var(prefix+"_weight", dtype="int8")
    out = relay.nn.conv2d(data, weight, kernel_size=(3, 3),
                          padding=(1, 1), strides=(1, 1),
                          channels=64, out_dtype="int32")
    out = relay.nn.leaky_relu(out, alpha = 0.1)
    out = relay.Function(relay.ir_pass.free_vars(out), out)
    graph = out
    with relay.build_config(opt_level=0):
        func = graph
        func = relay.ir_pass.infer_type(func)
        func = relay.ir_pass.fuse_ops(func, 0)
        func = relay.ir_pass.infer_type(func)
        graph_gen = _graph_gen.GraphRuntimeCodegen(mod=None, target='llvm')
        graph_json, lowered_funcs, params = graph_gen.codegen(func)
        print (graph_json)

if __name__ == "__main__":
    test_yxnet_mnist()

    #  test_yxnet_dog_cat()

