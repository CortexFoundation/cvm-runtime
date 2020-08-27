""" Generalized Op-level realization of Model Representation Tool.

    Implementation of validation, equivalent transformation, 
    quantization, transpose fusion, ops calculation, preparation and compilation.

    Only **crucial parts** of the custommized pass implementation are elaborated.
"""

import logging
import math
import numpy as np

from mxnet import ndarray as nd
import mxnet as mx
import cvm

from mrt.sym_utils import get_attr, sym_iter, is_params, is_inputs, \
                          nd_array, get_mxnet_op, get_nnvm_op, nd_const, \
                          get_entry_id
from mrt import sym_utils as sutils
from mrt.gen.tfm_base import register_pass, register_transformer
from mrt.gen.tfm_base import Transformer
from mrt.tfm_base import N, MAX_BIT
from mrt import sim_quant_helper as sim


@register_pass("validate")
@register_pass("fuse_transpose")
@register_pass("rewrite")
@register_pass("quantize")
@register_pass("calculate_ops")
@register_pass("prepare_for_compile")
@register_transformer("Flatten")
class Flatten(Transformer):
    def compile(self, op, **kwargs):
        childs = kwargs['childs']
        attrs = kwargs['attr']
        sym = get_nnvm_op("flatten")(*childs, name=N.n(), **attrs)
        return sym


@register_pass("validate")
@register_pass("rewrite")
@register_pass("fuse_transpose")
@register_pass("prepare_for_compile")
@register_transformer("null")
class Null(Transformer):
    def quantize(self, op, **kwargs):
        """ Customized quantize pass Introduction.

            Transform the input data.
        """
        if is_inputs(op, kwargs['params']):
            name, attr = op.attr('name'), op.list_attr()
            prec = kwargs['precs'][name][OUT_KEY]
            kwargs['scales'][name] = scale(kwargs['th_dict'][name], prec)
            extra_attr = {'precision': str(prec)}
            return mx.sym.var(name, **attr, attr=extra_attr)
        return op

    def compile(self, op, **kwargs):
        return cvm.sym.var(op.attr('name'), **kwargs['attr'])

    def calculate_ops(self, op, **kwargs):
        return 0


@register_pass("fuse_transpose")
@register_pass("quantize")
@register_pass("prepare_for_compile")
@register_transformer("Pooling")
class Pooling(Transformer):
    def validate(self, op, **kwargs):
        """ Customized validate pass Introduction.

            The 'layout' only support 'NCHW'.

            The 'pool_type' only support 'max' and 'avg'.
            And if 'pool_type' is 'avg' and 'pooling_convention' is 'full', 
            then 'global_pool' must be True.
            And if 'pool_type' is 'avg' and 'pooling_convention'  
            is not 'full', then 'pooling_convention' must be 'valid' 
            and 'global_pool' must be True.

            The 'count_include_pad' must be True.
        """
        name, op_name = op.attr('name'), op.attr('op_name')
        attr = op.list_attr()
        layout = get_attr(attr, 'layout', 'NCHW')
        assert layout == 'NCHW'
        pool_type = get_attr(attr, 'pool_type', 'max')
        assert pool_type in ['max', 'avg'], \
            "Pooling(%s) only supported type for max and avg." % name
        assert get_attr(attr, 'count_include_pad', True), \
            "Pooling(%s) only supported count_include_pad for True." % name

        if pool_type == 'avg':
            global_pool = get_attr(attr, 'global_pool', False)
            pooling_convention = get_attr(attr, 'pooling_convention', 'valid')
            if pooling_convention == 'full':
                msg = "%s(%s attr=%s) not match attribute %s (%s vs. %s)"
                assert global_pool, msg % (name, op_name, attr,
                                           'pooling_convention&global_pool',
                                           [attr['pooling_convention'],
                                            attr['global_pool']],
                                           ['full', 'True'])
            else:
                assert pooling_convention == 'valid' or global_pool, \
                    "Pooling(%s) only supported convention for valid." % name
        return op

    def compile(self, op, **kwargs):
        childs = kwargs['childs']
        attrs = kwargs['attr']
        kernel = get_attr(attrs, 'kernel')
        global_pool = 'global' if get_attr(attrs, 'global_pool', False) else ''
        pool_type = get_attr(attrs, 'pool_size', 'max')
        op_name = '_'.join([global_pool, pool_type, 'pool2d']).strip('_')
        new_attrs = {}
        if not global_pool:
            new_attrs['pool_size'] = kernel
            new_attrs['strides'] = get_attr(attrs, 'stride', (1, 1))
            new_attrs['padding'] = get_attr(attrs, 'pad', (0, 0))
            new_attrs['ceil_mode'] = (get_attr(attrs, 'pooling_convention',
                                               'valid') == 'full')
            if pool_type == 'avg':
                new_attrs['count_include_pad'] = \
                        get_attr(attrs, 'count_include_pad', True)
        return get_nnvm_op(op_name)(*childs, name=N.n('pooling'),
                                    **new_attrs)

    def rewrite(self, op, **kwargs):
        """ Customized rewrite pass Introduction.

            **Case 1. 'pool_type' is 'avg' and 'global_pool' is True**

            .. math::
                scale\_sym = 1 / (xshp[2] * xshp[3])

            where 'xshp' is the infer shape of the input 'X'.

            .. math::
                op\_s = sum(x, axis=(2, 3))

            .. math::
                op = op\_s * scale\_sym

            **Case 2. 'pool_type' is 'avg' and 'global_pool' is False**

            .. code-block:: python

                conv_attr = {
                    'no_bias': 'True',
                    'dilate': '(1, 1)',
                    'kernel': kernel,
                    'stride': stride,
                    'pad': pad,
                    'layout': 'NCHW',
                    'num_filter': xshp[1],
                    'num_group': xshp[1],
                }

            where 'kernel' is the pooling kernel size, 
            'stride' is the stride for pooling, 
            'pad' is the pad for pooling.

            The 'Activation' operator could be converted into 'Convolution'. 
            First, set up the attributes:

            .. math::
                W = full(shape=wshp, val=1/product(kernel))

            .. math::
                op = Convolution(X, W, conv\_attr)
        """
        params, graph = kwargs['params'], kwargs['graph']
        infer_shapes = kwargs['infer_shapes']
        attr = op.list_attr()
        childs = sym_iter(op.get_children())
        pool_type = get_attr(attr, 'pool_type', 'max')
        is_global = get_attr(attr, 'global_pool', False)
        if pool_type == 'avg' and is_global:
            X = childs[0]
            X_name = X.attr('name')
            X_shape = infer_shapes[X_name][get_entry_id(X)]
            scale_name = N.n('avg_scale')
            graph[scale_name] = scale_sym = mx.sym.var(scale_name, shape=(1,))
            params[scale_name] = nd_array([1. / (X_shape[2] * X_shape[3])])
            op = mx.sym.sum(childs[0], axis=(2, 3), name=N.n('sum'), keepdims=True)
            op = mx.sym.broadcast_mul(op, scale_sym, name=N.n('braodcast_mul'))
        elif pool_type == 'avg':
            X = childs[0]
            X_shape = infer_shapes[X.attr('name')][get_entry_id(X)]
            in_channel = X_shape[1]
            kernel = get_attr(attr, 'kernel')
            if isinstance(kernel, int):
                kernel = (kernel, kernel)
            conv_attr = {
                'no_bias': 'True',
                'dilate': '(1, 1)',
                'kernel': kernel,
                'stride': attr['stride'],
                'pad': attr['pad'],
                'layout': 'NCHW',
                'num_filter': in_channel,
                'num_group': in_channel,
            }
            conv_name = N.n('pool_conv')
            W_name = N.n('weight')
            W_shape = (in_channel, 1, *kernel)
            graph[W_name] = W = mx.sym.var(W_name, shape=W_shape)
            params[W_name] = sutils.nd_full(shape=W_shape,
                                            val=(1/np.product(kernel)))
            op = mx.sym.Convolution(X, W, **conv_attr, name=conv_name)
        return op

    def calculate_ops(self, op, **kwargs):
        X, attr = sym_iter(op.get_children())[0], op.list_attr()
        pool_type = get_attr(attr, 'pool_type', 'max')
        infer_shapes = kwargs['infer_shapes']
        if get_attr(attr, 'global_pool', False):
            _, _, K1, K2 = infer_shapes[X.attr('name')][get_entry_id(X)]
        else:
            K1, K2 = get_attr(attr, 'kernel')
        kwargs['base_ops'] = K1 * K2
        if pool_type == 'avg':
            kwargs['base_ops'] += 1
        return super().calculate_ops(op, **kwargs)


@register_pass("validate")
@register_pass("calculate_ops")
@register_pass("rewrite")
@register_pass("quantize")
@register_pass("prepare_for_compile")
@register_transformer("Dropout")
class Dropout(Transformer):
    def compile(self, op, **kwargs):
        childs = kwargs['childs']
        return childs[0]

    def fuse_transpose(self, op, **kwargs):
        """ Customized fuse_transpose pass Introduction.

            See :func:`mrt.tfm_ops.reverse_transpose <.reverse_transpose>` for reference.
        """
        return reverse_transpose(op)


@register_pass("quantize")
@register_transformer("Activation")
class Activation(Transformer):
    def validate(self, op, **kwargs):
        """ Customized validate pass Introduction.

            The activation function only support `relu`.
        """
        attr = op.list_attr()
        assert attr['act_type'] in [Relu.op_name], \
            "Only supported relu activation"
        return op

    def fuse_transpose(self, op, **kwargs):
        attr = op.list_attr()
        if attr['act_type'] == Relu.op_name:
            op = Relu().fuse_transpose(op, **kwargs)
        return op

    def rewrite(self, op, **kwargs):
        attr = op.list_attr()
        if attr['act_type'] == Relu.op_name:
            op = Relu().rewrite(op, **kwargs)
        return op

    def calculate_ops(self, op, **kwargs):
        attr = op.list_attr()
        if attr['act_type'] == Relu.op_name:
            op = Relu().calculate_ops(op, **kwargs)
        return op

    def prepare_for_compile(self, op, **kwargs):
        attr = op.list_attr()
        if attr['act_type'] == Relu.op_name:
            op = Relu().prepare_for_compile(op, **kwargs)
        return op

    def compile(self, op, **kwargs):
        attrs = kwargs['attr']
        act_type = attrs['act_type']
        if act_type == Relu.op_name:
            nkwargs = {k: v for k, v in kwargs.items() if k != 'attr'}
            nattrs = {k: v for k, v in attrs.items() if k != 'act_type'}
            nkwargs['attr'] = nattrs
            sym = Relu().compile(op, **nkwargs)
        return sym


@register_pass("validate")
@register_pass("fuse_transpose")
@register_pass("prepare_for_compile")
@register_transformer("FullyConnected")
class FullyConnected(Transformer):
    def rewrite(self, op, **kwargs):
        """ Customized rewrite pass Introduction.

            Using matrix decomposition to avoid overflow.

            .. math::
                Y = B + X*W^T = B + X1*W1^T + X2*W2^T + ...

            .. math::
                Wi.shape = (numHidden, step), W = [W1, W2, ...]

            .. math::
                Xi.shape = (batchSize, step), X = [X1, X2, ...]
        """
        infer_shapes, params = kwargs['infer_shapes'], kwargs['params']
        op = self._matrix_decomposition(op, params, infer_shapes)
        return op

    def quantize(self, op, **kwargs):
        """ Customized quantize pass Introduction.

            See :func:`mrt.tfm_ops._quantize_xwb <._quantize_xwb>` for reference
        """
        return _quantize_xwb(op, **kwargs)

    def compile(self, op, **kwargs):
        childs = kwargs['childs']
        attrs = kwargs['attr']
        op_name, new_attrs = 'dense', {}
        new_attrs['units'] = get_attr(attrs, 'num_hidden')
        new_attrs['use_bias'] = not get_attr(attrs, 'no_bias', False)
        try:
            mx.sym.FullyConnected(mx.sym.var('x'), num_hidden=1, flatten=True)
            has_flatten = True
        except mx.base.MXNetError:
            has_flatten = False
        use_flatten = get_attr(attrs, 'flatten', True)
        if has_flatten and use_flatten:
            childs[0] = cvm.symbol.flatten(childs[0], name=N.n('flatten'))
        return get_nnvm_op(op_name)(*childs, name=N.n('fullyconnected'),
                                    **new_attrs)

    def _matrix_decomposition(self, op, params, infer_shapes):
        attr = op.list_attr()
        childs = sym_iter(op.get_children())
        X, W = childs[:2]

        MATRIX_MAXIMUM_SIZE = 65536
        C = infer_shapes[W.attr('name')][get_entry_id(W)][1]
        if C <= MATRIX_MAXIMUM_SIZE:
            return op

        if X.attr('op_name') != Flatten.op_name:
            X = mx.sym.flatten(X, name=N.n('flatten'))

        no_bias = get_attr(attr, 'no_bias', False)
        attr['no_bias'] = True

        nodes, step, start = [], MATRIX_MAXIMUM_SIZE, 0
        wgt = params[W.attr('name')]
        while start < C:
            stop = min(start+step, C)
            Xk = mx.sym.slice_axis(X, axis=1,
                                   begin=start, end=stop,
                                   name=N.n("slice_axis"))
            Wk_name = N.n('slice_axis')
            params[Wk_name] = wgt.slice_axis(axis=1, begin=start, end=stop)
            Wk = mx.sym.var(Wk_name, shape=params[Wk_name].shape)
            tmp = mx.sym.FullyConnected(Xk, Wk, name=N.n("dense"), **attr)
            nodes.append(tmp)
            start += step

        while len(nodes) > 1:
            a, b = nodes.pop(0), nodes.pop(0)
            tmp = mx.sym.elemwise_add(a, b, name=N.n("elemwise_add"))
            nodes.append(tmp)

        op = nodes[0]
        if not no_bias:
            op = mx.sym.broadcast_add(op, childs[2], name=N.n('broadcast_add'))

        return op

    def calculate_ops(self, op, **kwargs):
        W = sym_iter(op.get_children())[1]
        infer_shapes = kwargs['infer_shapes']
        W_shape = infer_shapes[W.attr('name')][get_entry_id(W)]
        kwargs['base_ops'] = np.product(W_shape[1:]) * 2
        if not get_attr(op.list_attr(), 'no_bias', False):
            kwargs['base_ops'] += 1
        return super().calculate_ops(op, **kwargs)


@register_pass("fuse_transpose")
@register_pass("prepare_for_compile")
@register_transformer("Convolution")
class Convolution(Transformer):
    def validate(self, op, **kwargs):
        op = self._validate_layout(op, **kwargs)
        return op

    def _validate_overflow(self, op, **kwargs):
        W = sym_iter(op.get_children())[1]
        W_shp = kwargs['infer_shapes'][W.attr('name')][get_entry_id(W)]
        assert np.prod(W_shp[1:]) < 65536, "Convolution ops overflow"

    def _validate_layout(self, op, **kwargs):
        params = kwargs['params']
        name, op_name = op.attr('name'), op.attr('op_name')
        childs, attr = sym_iter(op.get_children()), op.list_attr()
        X, W = childs[0], childs[1]
        W_name = W.attr('name')

        layout = get_attr(attr, 'layout', "NCHW")
        if layout == "NCW":
            no_bias = get_attr(attr, 'no_bias', False)
            dilate, kernel = get_attr(attr, 'dilate'), get_attr(attr, 'kernel')
            pad, stride = get_attr(attr, 'pad'), get_attr(attr, 'stride')
            num_filter = get_attr(attr, 'num_filter')
            num_group = get_attr(attr, 'num_group', 1)
            attr = {
                'layout': "NCHW", 'no_bias': no_bias,
                'dilate': (*dilate, 1), 'kernel': (*kernel, 1),
                'pad': (*pad, 0), 'stride': (*stride, 1),
                'num_filter': num_filter, 'num_group': num_group,
            }
            X = mx.sym.expand_dims(X, axis=3)
            params[W_name] = params[W_name].expand_dims(axis=3)
            W = kwargs['graph'][W_name] = mx.sym.var(W_name, shape=params[W_name].shape)
            B = None if no_bias else childs[2]
            op = get_mxnet_op(op_name)(X, W, B, **attr, name=name)
            self._validate_overflow(op, **kwargs)
            op = mx.sym.squeeze(op, axis=3)
        else:
            assert layout == "NCHW", "Convolution(%s) only supported \
                    NCHW layout vs. %s" % (name, layout)
            self._validate_overflow(op, **kwargs)
        return op

    def rewrite(self, op, **kwargs):
        """ Customized rewrite pass Introduction.

            Input node with layout `NCW` is equivalently
            rewriten into layout `NCHW`
            The parameters if attached dimension is set by default
        """
        #TODO: matrix decomposition
        # op = self._fuse_bias(op, kwargs["infer_shapes"])
        params = kwargs['params']
        name, op_name = op.attr('name'), op.attr('op_name')
        childs, attr = sym_iter(op.get_children()), op.list_attr()
        X, W = childs[0], childs[1]
        W_name = W.attr('name')

        layout = get_attr(attr, 'layout', "NCHW")
        if layout == "NCW":
            no_bias = get_attr(attr, 'no_bias', False)
            dilate, kernel = get_attr(attr, 'dilate'), get_attr(attr, 'kernel')
            pad, stride = get_attr(attr, 'pad'), get_attr(attr, 'stride')
            num_filter = get_attr(attr, 'num_filter')
            num_group = get_attr(attr, 'num_group', 1)
            attr = {
                'layout': "NCHW", 'no_bias': no_bias,
                'dilate': (*dilate, 1), 'kernel': (*kernel, 1),
                'pad': (*pad, 0), 'stride': (*stride, 1),
                'num_filter': num_filter, 'num_group': num_group,
            }
            X = mx.sym.expand_dims(X, axis=3)
            params[W_name] = params[W_name].expand_dims(axis=3)
            W = kwargs['graph'][W_name] = mx.sym.var(W_name, shape=params[W_name].shape)
            B = None if no_bias else childs[2]
            op = get_mxnet_op(op_name)(X, W, B, **attr, name=name)
            self._validate_overflow(op, **kwargs)
            op = mx.sym.squeeze(op, axis=3)
        return op

    def _fuse_bias(self, op, infer_shapes):
        name = op.attr('name')
        childs, attr = sym_iter(op.get_children()), op.list_attr()
        if get_attr(attr, 'no_bias', False):
            return op

        attr['no_bias'] = True
        X, W, B = childs
        oshp = infer_shapes[op.attr('name')][0]
        op = mx.sym.Convolution(X, W, **attr, name=name)
        B = mx.sym.reshape(B, (1, oshp[1], 1, 1), name=N.n('reshape'))
        op = mx.sym.broadcast_add(op, B, name=N.n('broadcast_add'))
        return op

    def quantize(self, op, **kwargs):
        """ Customized quantize pass Introduction.

            See :func:`mrt.tfm_ops._quantize_xwb <._quantize_xwb>` for reference
        """
        return _quantize_xwb(op, **kwargs)

    def calculate_ops(self, op, **kwargs):
        W = sym_iter(op.get_children())[1]
        infer_shapes = kwargs['infer_shapes']
        W_shape = infer_shapes[W.attr('name')][get_entry_id(W)]
        kwargs['base_ops'] = np.product(W_shape[1:]) * 2
        if not get_attr(op.list_attr(), 'no_bias', False):
            kwargs['base_ops'] += 1
        return super().calculate_ops(op, **kwargs)

    def compile(self, op, **kwargs):
        childs = kwargs['childs']
        attrs = kwargs['attr']
        kernel = get_attr(attrs, 'kernel')
        layout = get_attr(attrs, 'layout', 'NCHW')
        kernel_layout = get_attr(attrs, 'kernel_layout', 'OIHW')
        op_name, new_attrs = 'conv2d', {}
        new_attrs['channels'] = get_attr(attrs, 'num_filter')
        new_attrs['kernel_size'] = kernel
        new_attrs['strides'] = get_attr(attrs, 'stride', (1, 1))
        new_attrs['padding'] = get_attr(attrs, 'pad', (0, 0))
        new_attrs['dilation'] = get_attr(attrs, 'dilate', (1, 1))
        new_attrs['groups'] = get_attr(attrs, 'num_group', 1)
        new_attrs['layout'] = layout
        new_attrs['kernel_layout'] = kernel_layout
        new_attrs['use_bias'] = not get_attr(attrs, 'no_bias', False)
        return get_nnvm_op(op_name)(*childs, name=N.n('convolution'),
                                    **new_attrs)

def reverse_transpose(op):
    """
        For symbol with single Transpose input, 
        reverse these sequence if this two op is swapable.

        .. code-block:: none

            X -> Transpose -> op

        after reverse sequence is

        .. code-block:: none

            X -> op -> Transpose

        Notice:
            After and before swap the axis of the Transpose remains the same.
    """
    name, op_name = op.attr('name'), op.attr('op_name')
    childs, attrs = sutils.sym_iter(op.get_children()), op.list_attr()
    assert len(childs) == 1
    X = childs[0]
    if X.attr('op_name') == Transpose.op_name:
        t_name, t_attr = X.attr('name'), X.list_attr()
        X = X.get_children()[0]
        op = get_mxnet_op(op_name)(X, **attrs, name=name)
        op = mx.sym.transpose(op, name=t_name, **t_attr)
    return op

def _quantize_xwb(op, **kwargs):
    """ quantization function with the inputs form of:

        .. math::
            Y = X*W + B

        The input and weight are quantized into the same precision level. 
        Bias is quantized with respect to the product of input and weight.

        the infer precision equals to the sum of quantized input precision, 
        quantized weight precision and the product precision.
    """
    th_dict, scales = kwargs['th_dict'], kwargs['scales']
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
