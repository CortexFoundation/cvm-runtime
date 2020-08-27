import logging
import math
import numpy as np

from mxnet import ndarray as nd
import mxnet as mx
import cvm

from mrt.sym_utils import get_attr, sym_iter, is_params, is_inputs, \
                          nd_array, get_mxnet_op, get_nnvm_op, \
                          nd_const, get_entry_id
from mrt.tfm_base import N, MAX_BIT
from mrt.tfm_pass import OUT_KEY
from mrt.gen.tfm_base import register_pass, register_transformer, \
                             Transformer
from mrt.gen.tfm_types import get_quantizer

from mrt import sim_quant_helper as sim
from mrt import sym_utils as sutils
from mrt import tfm_ops as tops


@register_pass("validate")
@register_pass("fuse_transpose")
@register_pass("rewrite")
@register_pass("quantize")
@register_pass("calculate_ops")
@register_pass("prepare_for_compile")
@register_transformer("Flatten")
class Flatten(tops.Flatten):
    pass


@register_pass("validate")
@register_pass("rewrite")
@register_pass("fuse_transpose")
@register_pass("prepare_for_compile")
@register_transformer("null")
class Null(tops.Null):
    def quantize(self, op, **kwargs):
        if is_inputs(op, kwargs['params']):
            name, attr = op.attr('name'), op.list_attr()
            prec = kwargs['precs'][name][OUT_KEY]
            quantizer = get_quantizer("UniformSymmetric")
            ft = kwargs['features'][name]
            kwargs['buffers'][name] = quantizer.get_buffer(prec, ft)
            # raise NotImplementedError(
                # "Quantization type not implementated," + \
                # " op: %20s, quantizer: %20s" % (op_name, ))
            extra_attr = {'precision': str(prec)}
            return mx.sym.var(name, **attr, attr=extra_attr)
        return op


@register_pass("fuse_transpose")
@register_pass("rewrite")
@register_pass("prepare_for_compile")
@register_transformer("Pooling")
class Pooling(tops.Pooling):
    def quantize(self, op, **kwargs):
        op = Transformer().quantize(op, **kwargs)
        return op


@register_pass("validate")
@register_pass("calculate_ops")
@register_pass("rewrite")
@register_pass("quantize")
@register_pass("prepare_for_compile")
@register_transformer("Dropout")
class Dropout(tops.Dropout):
    pass


@register_pass("rewrite")
@register_pass("quantize")
@register_transformer("Activation")
class Activation(tops.Activation):
    pass


@register_pass("validate")
@register_pass("fuse_transpose")
@register_pass("prepare_for_compile")
@register_transformer("FullyConnected")
class FullyConnected(tops.FullyConnected):
    def rewrite(self, op, **kwargs):
        op = super().rewrite(op, **kwargs)
        op = separate_bias(op, **kwargs)
        return op

    def quantize(self, op, **kwargs):
        pass


@register_pass("fuse_transpose")
@register_pass("prepare_for_compile")
@register_transformer("Convolution")
class Convolution(tops.Convolution):
    def rewrite(self, op, **kwargs):
        op = super().rewrite(op, **kwargs)
        op = separate_pad(op, **kwargs)
        op = separate_bias(op, **kwargs)
        return op

    def quantize(self, op, **kwargs):
        pass


@register_pass("validate")
@register_pass("calculate_ops")
@register_pass("fuse_transpose")
@register_pass("rewrite")
@register_pass("quantize")
@register_pass("prepare_for_compile")
@register_transformer('Pad')
class Pad(Transformer):
    pass

def separate_bias(op, **kwargs):
    name, op_name = op.attr('name'), op.attr('op_name')
    attrs, childs = op.list_attr(), sym_iter(op.get_children())

    if len(childs) < 3 or op_name not in \
        [Convolution.op_name, FullyConnected.op_name]:
        return op

    attrs['no_bias'] = True
    op = get_mxnet_op(op_name)(
        childs[0], childs[1], **attrs, name=N.n(name))
    B = mx.sym.expand_dims(childs[2], axis=0)
    if op_name == Convolution.op_name:
        assert 'layout' in attrs and attrs['layout'] == 'NCHW'
        B = mx.sym.expand_dims(B, axis=-1)
        B = mx.sym.expand_dims(B, axis=-1)
    op = mx.sym.broadcast_add(op, B, name=name)
    return op

def separate_pad(op, **kwargs):
    name, op_name = op.attr('name'), op.attr('op_name')
    attrs, childs = op.list_attr(), sym_iter(op.get_children())

    if op_name not in [Convolution.op_name]:
        return op

    assert 'layout' in attrs and attrs['layout'] == 'NCHW'
    PH, PW = get_attr(attrs, 'pad', (0,0))
    if PH == 0 and PW == 0:
        return op
    del attrs['pad']

    childs[0] = mx.sym.pad(
        childs[0], pad_width=(0,0,0,0,PH,PH,PW,PW),
        mode='constant', constant_value=0, name=N.n(name))
    op = get_mxnet_op(op_name)(*childs, **attrs, name=name)
    return op

