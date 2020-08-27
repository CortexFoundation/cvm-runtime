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
from mrt.gen.tfm_base import register_pass, register_transformer
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
        """ Customized quantize pass Introduction.

            Transform the input data.
        """
        if is_inputs(op, kwargs['params']):
            name, attr = op.attr('name'), op.list_attr()
            prec = kwargs['precs'][name][OUT_KEY]
            quantizer = get_quantizer("UniformSymmetric")
            ft = kwargs['features'][name]
            kwargs['buffers'][name] = quantizer.get_buffer(prec, ft)
            # else:
                # raise NotImplementedError(
                    # "Quantization type not implementated," + \
                    # " op: %20s, quantizer: %20s" % (op_name, ))
            extra_attr = {'precision': str(prec)}
            return mx.sym.var(name, **attr, attr=extra_attr)
        return op


@register_pass("fuse_transpose")
@register_pass("quantize")
@register_pass("prepare_for_compile")
@register_transformer("Pooling")
class Pooling(tops.Pooling):
    pass


@register_pass("validate")
@register_pass("calculate_ops")
@register_pass("rewrite")
@register_pass("quantize")
@register_pass("prepare_for_compile")
@register_transformer("Dropout")
class Dropout(tops.Dropout):
    pass


@register_pass("quantize")
@register_transformer("Activation")
class Activation(tops.Activation):
    pass


@register_pass("validate")
@register_pass("fuse_transpose")
@register_pass("prepare_for_compile")
@register_transformer("FullyConnected")
class FullyConnected(tops.FullyConnected):
    def quantize(self, op, **kwargs):
        """ Customized quantize pass Introduction.

            See :func:`mrt.tfm_ops._quantize_xwb <._quantize_xwb>` for reference
        """
        return _quantize_xwb(op, **kwargs)


@register_pass("fuse_transpose")
@register_pass("prepare_for_compile")
@register_transformer("Convolution")
class Convolution(tops.Convolution):
    def quantize(self, op, **kwargs):
        """ Customized quantize pass Introduction.

            See :func:`mrt.tfm_ops._quantize_xwb <._quantize_xwb>` for reference
        """
        return _quantize_xwb(op, **kwargs)

def _quantize_xwb(op, **kwargs):
    """ quantization function with the inputs form of:

        .. math::
            Y = X*W + B

        The input and weight are quantized into the same precision level. 
        Bias is quantized with respect to the product of input and weight.

        the infer precision equals to the sum of quantized input precision, 
        quantized weight precision and the product precision.
    """
    features, buffers = kwargs['features'], kwargs['buffers']
    name, op_name = op.attr('name'), op.attr('op_name')
    childs, attr = sym_iter(op.get_children()), op.list_attr()
    cns = [c.attr('name') for c in childs] if childs else []

    oprec = kwargs['op_input_precs'][op_name]
    X, xprec, xs = requant_operator(childs[0], oprec, oname=name, **kwargs)
    W, wprec, ws = requant_parameter(cns[1], oprec, oname=name, **kwargs)
    B, bprec = None, None
    if not get_attr(attr, 'no_bias', False):
        bs = ws * xs
        bias_prec = get_bit(th_dict[cns[2]] * bs)
        B, bprec, _ = requant_parameter(
            cns[2], bias_prec, bs, oname=name, **kwargs)
    scales[name] = ws * xs
    op = get_mxnet_op(op_name)(X, W, B, **attr, name=name)

    shp = kwargs['params'][childs[1].attr('name')].shape
    k = int(nd.prod(nd_array(shp[1:])).asscalar())
    kprec = get_bit_cnt(k)
    infer_prec = kprec + xprec + wprec
    if not get_attr(attr, 'no_bias', False):
        infer_prec = max(infer_prec, bprec) + 1
    kwargs['precs'][name][OUT_KEY] = infer_prec

    logger = logging.getLogger('log.mrt.realize')
    logger.debug("operator  %-20s name=%-40s oscale=%s, iscale=%s",
                 op_name, name, scales[name], cns)
    return op
