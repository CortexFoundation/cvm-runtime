""" Op-level realization of Model Representation Tool.
    Implementation of validation, 
    equivalent transformation, 
    quantization, transpose fusion, 
    ops calculation, preparation and compilation.

    Only **crucial parts** of the custommized pass implementation are elaborated.
"""

import logging
import math
import numpy as np

from mxnet import ndarray as nd
import mxnet as mx
import cvm

from .tfm_utils import get_bit, get_range, scale, get_bit_cnt, \
                      requant, requant_operator, requant_parameter, \
                      realize
from .sym_utils import get_attr, sym_iter, is_params, is_inputs, \
                      nd_array, get_mxnet_op, get_nnvm_op, nd_const, \
                      get_entry_id
from . import sym_utils as sutils
from .tfm_base import register_pass, register_transformer, Transformer, \
                     N, OUT_KEY, MAX_BIT
from . import sim_quant_helper as sim


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


@register_pass("rewrite")
@register_pass("quantize")
@register_pass("calculate_ops")
@register_pass("prepare_for_compile")
@register_pass('compile')
@register_transformer("transpose")
class Transpose(Transformer):
    def validate(self, op, **kwargs):
        infer_shapes = kwargs['infer_shapes']
        shp = infer_shapes[op.attr('name')][get_entry_id(op)]
        childs, attr = sym_iter(op.get_children()), op.list_attr()
        if not get_attr(attr, 'axes', []):
            attr['axes'] = list(reversed(range(len(shp))))
            op = mx.sym.transpose(*childs, **attr)
        return op

    def fuse_transpose(self, op, **kwargs):
        """ Customized fuse_transpose pass Introduction.

            For continuous transpose sequence like:

            .. math::
                Z = \\text{Transpose}(Y, axes_2)

            .. math::
                Y = \\text{Transpose}(X, axes_1)

            Exert equivalent transformation on the adjacent two transpose ops:

            .. math::
                Z(j) = \\text{Transpose}(Y, axes_2) \\\\
                     = \\text{Transpose}(\\text{Tranpose}(X, axes_1), axes_2) \\\\
                     = \\text{Transpose}(X, axes_3), \\\\

                \\text{where } axes_3(j) = axes_1(axes_2(j))
        """
        name, attr = op.attr('name'), op.list_attr()
        axes = get_attr(attr, 'axes')
        X = sym_iter(op.get_children())[0]
        if X.attr('op_name') == Transpose.op_name:
            tattr = X.list_attr()
            caxes = get_attr(tattr, 'axes')
            axes = [caxes[ii] for ii in axes]
            op = X.get_children()[0]
            if axes != sorted(axes):
                op = mx.sym.transpose(op, axes=axes, name=name)
        return op


@register_pass("validate")
@register_pass("rewrite")
@register_pass("calculate_ops")
@register_pass("quantize")
# @register_pass("prepare_for_compile")
@register_pass("compile")
@register_transformer("relu")
class Relu(Transformer):
    def fuse_transpose(self, op, **kwargs):
        """ Customized fuse_transpose pass Introduction.

            See :func:`mrt.tfm_ops.reverse_transpose <.reverse_transpose>` for reference.
        """
        return reverse_transpose(op)

    def prepare_for_compile(self, op, **kwargs):
        # The reverse process is comment, refer to 
        #   `reverse_sequence` for more details.
        # X = sym_iter(op.get_children())[0]
        # if is_fusable_cvm_precision(X):
        #     op = reverse_sequence(op)
        return op


@register_pass("calculate_ops")
@register_transformer("LeakyReLU")
class LeakyReLU(Transformer):
    def validate(self, op, **kwargs):
        """ Customized validate pass Introduction.

            The activation function only support `leaky`.
        """
        name, attr = op.attr('name'), op.list_attr()
        act = get_attr(attr, 'act_type', 'leaky')
        assert act == 'leaky', "Unsupported LeakyReLU %s for act_type: %s" \
                % (name, act)
        return op

    def fuse_transpose(self, op, **kwargs):
        """ Customized fuse_transpose pass Introduction.

            See :func:`mrt.tfm_ops.reverse_transpose <.reverse_transpose>` for reference.
        """
        return reverse_transpose(op)

    def rewrite(self, op, **kwargs):
        """ Customized rewrite pass Introduction.

            LeakyReLU can be equivalently transformed to be supported by cvm.

            .. math::
                LeakyReLU(X) = relu(X) - slope*relu(-X)
        """
        childs, attr = sym_iter(op.get_children()), op.list_attr()

        slope = get_attr(attr, 'slope', 0.25)
        X = childs[0]
        posi_X = mx.sym.relu(X)
        nega_X = mx.sym.negative(X)
        nega_X = mx.sym.relu(nega_X)
        slope_name = N.n('slope')
        kwargs['params'][slope_name] = nd_array([slope])
        kwargs['graph'][slope_name] = slope_sym = \
                mx.sym.var(slope_name, shape=(1,))
        scale_X = mx.sym.broadcast_mul(nega_X, slope_sym)
        op = posi_X - scale_X
        return op


@register_pass("validate")
@register_pass("calculate_ops")
@register_pass("fuse_transpose")
@register_transformer("_mul_scalar")
class MulScalar(Transformer):
    def rewrite(self, op, **kwargs):
        """ Customized rewrite pass Introduction.

            Transform into broadcast_mul.
        """
        params, graph = kwargs['params'], kwargs['graph']
        name = op.attr('name')
        scalar = get_attr(op.list_attr(), 'scalar')

        X = op.get_children()[0]
        var = nd_const(scalar, graph, params)
        return mx.sym.broadcast_mul(X, var, name=name)


@register_pass("validate")
@register_pass("calculate_ops")
@register_pass("fuse_transpose")
# @register_pass("prepare_for_compile") # only for restore
@register_transformer("_div_scalar")
class DivScalar(Transformer):
    def rewrite(self, op, **kwargs):
        """ Customized rewrite pass Introduction.

            Transform into broadcast_mul.
        """
        graph = kwargs['graph']
        name = op.attr('name')
        attr, childs = op.list_attr(), sym_iter(op.get_children())

        scalar = get_attr(attr, 'scalar')
        sname = N.n('scalar')
        kwargs['params'][sname] = nd_array([1/scalar])
        graph[sname] = mx.sym.var(sname, shape=(1,))
        return mx.sym.broadcast_mul(childs[0], graph[sname], name=name)


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
        elif attr['act_type'] == Sigmoid.op_name:
            op = Sigmoid().fuse_transpose(op, **kwargs)
        return op

    def rewrite(self, op, **kwargs):
        """ Equivalent transform of rewrite operator
            Only applies when the attribute act_type equals to relu or sigmoid,
            which indicates that rewrite could be directly tranformed into
            the corresponding operator.
        """
        attr = op.list_attr()
        if attr['act_type'] == Relu.op_name:
            op = Relu().rewrite(op, **kwargs)
        elif attr['act_type'] == Sigmoid.op_name:
            childs = sym_iter(op.get_children())
            op = mx.sym.sigmoid(childs[0])
        return op

    def calculate_ops(self, op, **kwargs):
        attr = op.list_attr()
        if attr['act_type'] == Relu.op_name:
            op = Relu().calculate_ops(op, **kwargs)
        elif attr['act_type'] == Sigmoid.op_name:
            op = Sigmoid().calculate_ops(op, **kwargs)
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
            op = Relu().compile(op, **nkwargs)
        return op


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


@register_pass("validate")
@register_pass("calculate_ops")
@register_pass("fuse_transpose")
@register_pass("rewrite")
@register_pass("quantize")
@register_pass("prepare_for_compile")
@register_transformer('Pad')
class Pad(Transformer):
    #TODO(ryt): currently pad_value is taken as 0
    # revise pad_value with respect to scale
    # if other constant value is taken into consideration
    def compile(self, op, **kwargs):
        """ Customized compile pass Introduction.

            Only support *constant* padding type.
        """
        childs = kwargs['childs']
        attrs = kwargs['attr']

        assert attrs['mode'] == 'constant', \
            "nnvm pad symbol only support `constant` pad"
        del attrs['mode']

        pad_value = eval(attrs.get('constant_value', '0'))
        assert type(pad_value).__name__ in ['int', 'float'], \
            "not a valid value: attrs['constant_value']"
        attrs['pad_value'] = pad_value
        if 'constant_value' in attrs:
            del attrs['constant_value']

        pad_width = list(eval(attrs['pad_width']))
        assert all([type(val).__name__ == 'int' for val in pad_width]), \
            "not a valid value: attrs['pad_width']"
        #  attrs['pad_width'] = tuple([tuple((pad_width[i:i+2])) \
            #  for i in range(0, len(pad_width), 2)])

        return get_nnvm_op('pad')(*childs, name=N.n('pad'), **attrs)


@register_pass("validate")
@register_pass("calculate_ops")
@register_pass("fuse_transpose")
@register_pass("rewrite")
@register_pass("quantize")
@register_pass("prepare_for_compile")
@register_transformer('expand_dims')
class ExpandDims(Transformer):
    def compile(self, op, **kwargs):
        childs = kwargs['childs']
        attrs = kwargs['attr']
        op_name, new_attrs = 'expand_dims', {}
        new_attrs['axis'] = get_attr(attrs, 'axis', 'expand_dims')
        return get_nnvm_op(op_name)(*childs, **new_attrs)


@register_pass("validate")
@register_pass("calculate_ops")
@register_pass("fuse_transpose")
@register_pass("rewrite")
@register_pass("prepare_for_compile")
@register_transformer('Embedding')
class Embedding(Transformer):
    def quantize(self, op, **kwargs):
        """ Customized quantize pass Introduction.

            The input array to the embedding operator should be scaled to 1.
        """
        th_dict, scales = kwargs['th_dict'], kwargs['scales']
        name, op_name = op.attr('name'), op.attr('op_name')
        attr, childs = op.list_attr(), sym_iter(op.get_children())
        cns = [c.attr('name') for c in childs]

        iprec = kwargs['op_input_precs'][op_name]
        X, xs = childs[0], scales[cns[0]]
        if xs != 1:
            X, _, _ = requant(X, 32, oscale=1, oname=name, **kwargs)
        W, _, ws = requant_parameter(cns[1], iprec, oname=name, **kwargs)
        th_dict[name] = th_dict[cns[1]]
        scales[name] = ws
        kwargs['precs'][name][OUT_KEY] = get_bit(th_dict[name]*ws)
        op = get_mxnet_op(op_name)(X, W, **attr, name=name)

        return op

    def compile(self, op, **kwargs):
        childs = kwargs['childs']
        indices, weight = childs
        op_name = 'take'
        return get_nnvm_op(op_name)(weight, indices, axis=0)


@register_pass("validate")
@register_pass("rewrite")
@register_pass("calculate_ops")
@register_pass("fuse_transpose")
@register_pass("quantize")
@register_pass("prepare_for_compile")
@register_transformer("repeat")
class Repeat(Transformer):
    def compile(self, op, **kwargs):
        childs = kwargs['childs']
        attrs = kwargs['attr']
        data = childs[0]
        new_attrs = {}
        op_name = 'repeat'
        new_attrs['repeats'] = get_attr(attrs, 'repeats', 'repeat')
        if 'axis' in attrs:
            new_attrs['axis'] = get_attr(attrs, 'axis')
        else:
            data = get_nnvm_op('flatten')(data)
            new_attrs['axis'] = 0
        return get_nnvm_op(op_name)(childs[0], **new_attrs)


@register_pass("calculate_ops")
@register_pass("validate")
@register_pass("fuse_transpose")
@register_pass("rewrite")
@register_pass("prepare_for_compile")
@register_transformer('_contrib_box_nms')
class BoxNms(Transformer):
    def compile(self, op, **kwargs):
        childs = kwargs['childs']
        attrs = kwargs['attr']
        force_suppress = get_attr(attrs, 'force_suppress', False)

        iou_thresh = get_attr(attrs, 'overlap_thresh', 0.5) * 100
        iou_thresh = int(iou_thresh)

        top_k = get_attr(attrs, 'topk', -1)

        valid_thresh = get_attr(attrs, 'valid_thresh', 0)
        valid_thresh = int(valid_thresh)

        coord_start = get_attr(attrs, 'coord_start', 2)
        score_index = get_attr(attrs, 'score_index', 1)
        id_index = get_attr(attrs, 'id_index', -1)
        op_name = 'get_valid_counts'
        ret = get_nnvm_op(op_name)(childs[0], score_threshold=valid_thresh)
        op_name = 'non_max_suppression'
        nms_out = get_nnvm_op(op_name)(ret[1], ret[0],
                                       iou_threshold=iou_thresh,
                                       force_suppress=force_suppress,
                                       top_k=top_k,
                                       coord_start=coord_start,
                                       score_index=score_index,
                                       id_index=id_index,
                                       return_indices=False,
                                       invalid_to_bottom=True)
        return nms_out


@register_pass("validate")
@register_pass("rewrite")
@register_pass("calculate_ops")
@register_pass("fuse_transpose")
@register_pass("prepare_for_compile")
@register_transformer("slice_like")
class SliceLike(Transformer):
    def quantize(self, op, **kwargs):
        """ Customized quantize pass Introduction.

            See :func:`mrt.tfm_ops._quantize_scale <._quantize_scale>` for reference.
        """
        return _quantize_scale(op, **kwargs)

    def compile(self, op, **kwargs):
        childs = kwargs['childs']
        attrs = kwargs['attr']
        new_attrs = {'axis': get_attr(attrs, 'axes', ())}
        op_name = 'slice_like'
        return get_nnvm_op(op_name)(*childs, **new_attrs)


@register_pass("calculate_ops")
@register_pass("fuse_transpose")
@register_transformer("slice_axis")
class SliceAxis(Transformer):
    def validate(self, op, **kwargs):
        childs, attr = sym_iter(op.get_children()), op.list_attr()
        axis = get_attr(attr, 'axis')
        X = childs[0]
        cshape = kwargs['infer_shapes'][X.attr('name')][get_entry_id(X)]
        ndims = len(cshape)

        axis = get_attr(attr, 'axis')
        assert axis in range(-ndims, ndims)
        return op

    def rewrite(self, op, **kwargs):
        """ Customized rewrite pass Introduction.

            'SliceAxis' can be equivalently transformed into 'Slice' which is supported by cvm.
        """
        name = op.attr('name')
        childs, attr = sym_iter(op.get_children()), op.list_attr()
        X = childs[0]
        cshape = kwargs['infer_shapes'][X.attr('name')][get_entry_id(X)]
        ndims = len(cshape)

        axis = get_attr(attr, 'axis')
        axis = axis if axis >= 0 else axis+ndims
        axis_begin = get_attr(attr, 'begin')
        axis_end = get_attr(attr, 'end')
        axis_end = axis_end if axis_end else cshape[axis]

        begin = [None if i != axis else axis_begin for i in range(len(cshape))]
        end = [None if i != axis else axis_end for i in range(len(cshape))]
        op = mx.sym.slice(X, begin=begin, end=end, name=name)
        return op


@register_pass("validate")
@register_transformer("SliceChannel")
class SliceChannel(Transformer):
    pass


@register_pass("fuse_transpose")
@register_pass("rewrite")
@register_pass("quantize")
@register_pass("prepare_for_compile")
@register_transformer('UpSampling')
class UpSampling(Transformer):
    def validate(self, op, **kwargs):
        attrs = op.list_attr()
        sample_type = get_attr(attrs, "sample_type")
        assert sample_type == "nearest"

    def compile(self, op, **kwargs):
        childs = kwargs['childs']
        attrs = kwargs['attr']
        sc = get_attr(attrs, 'scale')
        op_name, new_attrs = 'upsampling', {'scale': int(sc)}
        return get_nnvm_op(op_name)(childs[0], **new_attrs)


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
        name = op.attr('name')
        attr, childs = op.list_attr(), sym_iter(op.get_children())
        cns = [c.attr('name') for c in childs]
        infer_shapes, params = kwargs['infer_shapes'], kwargs['params']
        xshp = infer_shapes[cns[0]][get_entry_id(childs[0])]

        if len(xshp) > 2:
            op = self.reduce(op, **kwargs)
            return op

        op = self._matrix_decomposition(op, params, infer_shapes)
        return op

    def reduce(self, op, **kwargs):
        """ Dimension reduction function considering
            both flatten cases.

            Denote the input as X and transformed operator as Y.
            If flatten is true, only one reduction of the high dimension input
            to 2 dimension is needed.

            .. math::
                RX = reshape(X)
                Y = FullyConnected(RX)

            If flatten is false, firstly one reduction of the input to 2
            dimension is needed. After FullyConnected op, the ouput should
            be reshaped to the correct output shape.

            .. math::
                RX = reshape(X)
                out = FullyConnected(RX)
                Y = reshape(out)
        """
        name = op.attr('name')
        attr, childs = op.list_attr(), sym_iter(op.get_children())
        cns = [c.attr('name') for c in childs]
        X, W = childs[:2]
        infer_shapes, params = kwargs['infer_shapes'], kwargs['params']
        xshp = infer_shapes[cns[0]][get_entry_id(X)]

        no_bias = get_attr(attr, 'no_bias')
        flatten = get_attr(attr, "flatten")
        num_hidden = get_attr(attr, "num_hidden")

        rshp_name = N.n("pre_reshape")
        if flatten:
            shape = (-1,) + xshp[1:]
            rshp = mx.sym.reshape(X, shape=shape, name=rshp_name)
            if no_bias:
                op = mx.sym.FullyConnected(
                    rshp, W, no_bias=no_bias, flatten=flatten,
                    num_hidden=num_hidden, name=name)
            else:
                op = mx.sym.FullyConnected(
                    rshp, W, childs[2], no_bias=no_bias, flatten=flatten,
                    num_hidden=num_hidden, name=name)
            op = self._matrix_decomposition(op, params, infer_shapes)
        else:
            fc_name = N.n("reduced_fc")
            default_batch_axis = 0
            batch_axis = \
                kwargs.get("batch_axes", {}).get(name, default_batch_axis)
            assert batch_axis < len(xshp), \
                "invalid batch_axis: {}, length of xshp: {}".format(
                batch_axis, len(xshp))
            if batch_axis == len(xshp)-1:
                product = int(nd.prod(nd.array(xshp)).asscalar())
                res_shp = int(product/xshp[batch_axis])
                shape = (res_shp, -1)
            else:
                shape = (-1, xshp[-1])
            rshp = mx.sym.reshape(X, shape=shape, name=rshp_name)
            if no_bias:
                fc = mx.sym.FullyConnected(
                    rshp, W, no_bias=no_bias, flatten=flatten,
                    num_hidden=num_hidden, name=fc_name)
            else:
                fc = mx.sym.FullyConnected(
                    rshp, W, childs[2], no_bias=no_bias, flatten=flatten,
                    num_hidden=num_hidden, name=fc_name)
            fc = self._matrix_decomposition(fc, params, infer_shapes)
            if batch_axis == len(xshp)-1:
                shape = xshp[:-1] + (num_hidden,)
            else:
                shape = \
                    xshp[:batch_axis] + (-1,) + \
                    xshp[batch_axis+1:-1] + (num_hidden,)
            op = mx.sym.reshape(fc, shape=shape, name=name)

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


@register_pass("calculate_ops")
@register_pass("validate")
@register_pass("rewrite")
@register_pass("fuse_transpose")
# @register_pass("prepare_for_compile") # only for restore
@register_transformer("sigmoid")
class Sigmoid(Transformer):
    def quantize(self, op, **kwargs):
        """ Customized quantize pass Introduction.

            See :func:`mrt.tfm_ops._quantize_table <._quantize_table>` for reference
        """
        return _quantize_table(op, **kwargs)


@register_pass("calculate_ops")
@register_pass("validate")
@register_pass("rewrite")
@register_pass("fuse_transpose")
# @register_pass("prepare_for_compile") # only for restore
@register_transformer("exp")
class Exp(Transformer):
    def quantize(self, op, **kwargs):
        """ Customized quantize pass Introduction.

            See :func:`mrt.tfm_ops._quantize_table <._quantize_table>` for reference
        """
        return _quantize_table(op, **kwargs)


@register_pass("validate")
@register_pass("rewrite")
@register_pass("fuse_transpose")
# @register_pass("prepare_for_compile") # only for restore
@register_transformer("softmax")
class Softmax(Transformer):
    def calculate_ops(self, op, **kwargs):
        infer_shapes = kwargs['infer_shapes']
        X = sym_iter(op.get_children())[0]
        xshp = infer_shapes[X.attr('name')][get_entry_id(X)]
        axis = get_attr(op.list_attr(), 'axis', -1)
        kwargs['base_ops'] = 2 + 2 * xshp[axis]
        return super().calculate_ops(op, **kwargs)

    def quantize(self, op, **kwargs):
        """ Customized quantize pass Introduction.

            **Step 1. Requant Input**

            .. math::
                Xq, xprec, xs = requant\_operator(X, iprec, oscale)

            **Step 2. Calculate Norm Value**

            First, calculate the bias with respect to unscaled input (denoted as 'alpha') as:

            .. math::
                alpha = int(lambd * xs)

            where 'lambd' stands for a hyperparameter configured by user.

            Then, calculate offset value with respect to each axis, and clip:

            .. math::
                max\_axis = max(Xq, axis)

            .. math::
                offset = broadcast\_mul(max\_axis - var)

            .. math::
                offset\_c = clip(norm, xprec)

            Next, calculate norm and clip:

            .. math::
                norm = relu(Xq - offset)

            .. math::
                norm = clip(norm, xprec)

            **Step 3. Create Lookup Table**

            .. math::
                dim = alpha + 1

            .. math::
                data = range(0, dim)

            .. math::
                table = exp(data / xs)

            .. math::
                table\_prec = get_bit(exp(||table||_2))

            for reference of 'get_bit', 
            see :func:`mrt.tfm_utils.get_bit <.get_bit>`.

            .. math::
                table\_c = clip(table, min=0, max=get_range(table\_prec))

            for reference of 'get_range', 
            see :func:`mrt.tfm_utils.get_range <.get_range>`.

            .. math::
                weight = reshape(round(table\_c), (dim, 1))

            **Step 4. Get Lookup value**

            The cvm customized operator 'cvm_lut' has been adopted.

            .. math::
                lut = cvm\_lut(norm, weight, dim)

            **Step 5. Get Output**

            .. math::
                sum_lut = sum(lut, axis)

            .. math::
                oprec = min(15, 31-tprec)

            .. math::
                oscale = get_range(oprec)

            .. math::
                prob = lut * oscale

            .. math::
                half_lut = realize(sum_lut, 1, 31)

            for reference of 'realize', 
            see :func:`mrt.tfm_utils.realize <.realize>`.

            .. math::
                prob\_b = prob + half_lut

            .. math::
                prob = prob\_b / sum_lut
        """
        params, graph = kwargs['params'], kwargs['graph']
        scales, precs = kwargs['scales'], kwargs['precs']
        name, op_name = op.attr('name'), op.attr('op_name')
        childs, attr = sym_iter(op.get_children()), op.list_attr()
        cns = [c.attr('name') for c in childs] if childs else []

        oprec = kwargs['op_input_precs'][op_name]
        xs = scale(kwargs['th_dict'][childs[0].attr('name')], oprec)
        X, xprec, xs = requant_operator(childs[0], oprec, xs,
                                        oname=name, **kwargs)
        axis = get_attr(attr, 'axis', -1)
        lambd = kwargs['softmax_lambd']
        alpha = int(lambd*xs)
        var = nd_const(alpha, graph, params)
        max_axis = mx.sym.max(X, axis=axis, keepdims=True)
        offset = mx.sym.broadcast_sub(max_axis, var, name=N.n('softmax_offset'))
        offset = realize(offset, 0, xprec)
        norm = mx.sym.broadcast_sub(X, offset, name=N.n('softmax_normalize'))
        norm = mx.sym.relu(norm, name=N.n('Softmax_filter'))
        norm = realize(norm, 0, xprec)

        data = sutils.nd_arange(0, alpha+1)
        table = nd.exp(data/xs)

        tprec = get_bit(math.exp(lambd))
        table = nd.clip(table, a_min=0, a_max=get_range(tprec))
        W_name = N.n('cvm_lut_weight')
        params[W_name] = weight = table.round().reshape(alpha+1, 1)
        wattr = {'precision': str(tprec)}
        W = graph[W_name] = mx.sym.var(W_name, shape=weight.shape, attr=wattr)
        # lut = mx.sym.Custom(norm, W, in_dim=alpha+1,
        #                     name=name, op_type='cvm_lut')
        lut = mx.sym.Custom(norm, W, in_dim=alpha+1,
                            name=N.n('softmax_lut'), op_type='cvm_lut')
        sum_lut = mx.sym.sum(lut, axis=axis, keepdims=True,
                             name=N.n("softmax_sum"))

        oprec = min(15, 31 - tprec)
        assert oprec > 8, "operator softmax(%s) lambda(%d) is too large" \
                % (name, lambd)
        oscale = get_range(oprec)
        var_scale = nd_const(oscale, graph, params)
        prob = mx.sym.broadcast_mul(lut, var_scale,
                                    name=N.n("softmax_output_scale"))
        half_lut = realize(sum_lut, 1, 31)
        prob = mx.sym.broadcast_add(prob, half_lut, name=N.n("softmax_round"))
        op = mx.sym.broadcast_div(prob, sum_lut, name=N.n("softmax_prob"))
        op = op.astype('int32').astype('float32')
        # op = mx.sym.floor(op) # simulate integer division
        # op = realize(op, 0, oprec)
        op = realize(op, 0, oprec, name=name)
        # oname = op.attr('name')
        precs[name][OUT_KEY] = oprec
        # precs[oname] = {OUT_KEY: oprec}
        # scales[oname] = scales[name] = oscale
        scales[name] = oscale

        logger = logging.getLogger('log.mrt.realize')
        logger.debug("operator  %-20s name=%-40s oscale=%s, iscale=%s",
                     op_name, name, scales[name], cns)
        return op


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
@register_pass("fuse_transpose")
@register_pass("rewrite")
@register_pass("compile")
@register_transformer("broadcast_mul")
class BroadcastMul(Transformer):
    def quantize(self, op, **kwargs):
        """ Customized quantize pass Introduction.

            .. math::
                Xq, xprec, xs = requant(X, oprec)

            .. math::
                Bq, bprec, bs = requant(B, oprec)

            where 'oprec' stands for the default quantization 
            precision for 'BroadcastMul'.

            See :func:`mrt.tfm_utils.requant <.requant>` for reference.

            .. math::
                op = Xq * Bq

            The infer precision equals to 'xprec' plus 'bprec'.
        """
        precs, scales = kwargs['precs'], kwargs['scales']
        name, op_name = op.attr('name'), op.attr('op_name')
        childs, attr = sym_iter(op.get_children()), op.list_attr()
        cns = [c.attr('name') for c in childs] if childs else []

        oprec = kwargs['op_input_precs'][op_name]
        X, xprec, xs = requant(childs[0], oprec, oname=name, **kwargs)
        B, bprec, bs = requant(childs[1], oprec, oname=name, **kwargs)

        op = get_mxnet_op(op_name)(X, B, **attr, name=name)

        if bprec == 1 and bs == 1:
            # special case: childs[1] is 0
            scales[name] = 1
            precs[name][OUT_KEY] = 1
        else:
            scales[name] = xs * bs
            infer_prec = xprec + bprec
            precs[name][OUT_KEY] = infer_prec

        logger = logging.getLogger('log.mrt.realize')
        logger.debug("operator  %-20s name=%-40s oscale=%s, iscale=%s",
                     op_name, name, scales[name], cns)
        return op

    def prepare_for_compile(self, op, **kwargs):
        """ Customized prepare_for_compile pass Introduction.

            If either one of the input equals to zero, 
            the op can be merged into zero.
        """
        params = kwargs['params']
        graph = kwargs['graph']

        name = op.attr('name')
        childs = sym_iter(op.get_children())
        cns = [c.attr('name') for c in childs]

        fuse = any([is_params(c, params) and \
                   params[c.attr('name')].abs().max().asscalar() == 0 \
                   for c in childs])
        if fuse:
           ishp = kwargs['infer_shapes'][name][get_entry_id(op)]
           attr = {'precision': str(1)}
           op = graph[name] = mx.sym.var(name, shape=ishp, attr=attr)
           params[name] = nd.zeros(list(ishp))

        return op


@register_pass("validate")
@register_pass("calculate_ops")
@register_pass("fuse_transpose")
@register_pass("rewrite")
@register_pass("prepare_for_compile")
@register_pass("compile")
@register_transformer("broadcast_add")
class BroadcastAdd(Transformer):
    def quantize(self, op, **kwargs):
        """ Customized quantize pass Introduction.

            See :func:`mrt.tfm_ops._quantize_scale <._quantize_scale>` for reference.
        """
        params = kwargs['params']
        th_dict = kwargs['th_dict']
        precs = kwargs['precs']
        scales = kwargs['scales']

        name = op.attr('name')
        childs = sym_iter(op.get_children())
        cns = [c.attr('name') for c in childs]
        ths = [th_dict[cn] for cn in cns]

        if ths[0] == 0 or ths[1] == 0:
            if ths[0] == 0 and ths[1] == 0:
                th_dict[name], precs[name], scales[name] = 0, {OUT_KEY: 1}, 1
                return op
            cn = cns[1] if ths[0] == 0 else cns[0]
            bit = get_bit(params[cn]) if cn in params else precs[cn][OUT_KEY]
            scales[name] = 1 if cn in params else scales[cn]
            precs[name] = {OUT_KEY: bit}
            th_dict[name] = get_range(bit)
            return op

        return _quantize_scale(op, **kwargs)


@register_pass("prepare_for_compile")
@register_pass("compile")
@register_transformer("broadcast_div")
class BroadcastDiv(Transformer):
    pass


@register_pass("calculate_ops")
@register_pass("prepare_for_compile")
@register_pass("compile")
@register_transformer("broadcast_sub")
class BroadcastSub(Transformer):
    def quantize(self, op, **kwargs):
        """ Customized quantize pass Introduction.

            See :func:`mrt.tfm_ops._quantize_scale <._quantize_scale>` for reference.
        """
        params = kwargs['params']
        th_dict = kwargs['th_dict']
        precs = kwargs['precs']
        scales = kwargs['scales']

        name = op.attr('name')
        childs = sym_iter(op.get_children())
        cns = [c.attr('name') for c in childs]
        ths = [th_dict[cn] for cn in cns]

        if ths[0] == 0 or ths[1] == 0:
            if ths[0] == 0 and ths[1] == 0:
                th_dict[name], precs[name], scales[name] = 0, {OUT_KEY: 1}, 1
                return op
            cn = cns[1] if ths[0] == 0 else cns[0]
            bit = get_bit(params[cn]) if cn in params else precs[cn][OUT_KEY]
            scales[name] = 1 if cn in params else scales[cn]
            precs[name] = {OUT_KEY: bit}
            th_dict[name] = get_range(bit)
            return op

        return _quantize_scale(op, **kwargs)


@register_pass("prepare_for_compile")
@register_pass("compile")
@register_transformer("broadcast_to")
class BroadcastTo(Transformer):
    pass


@register_pass("prepare_for_compile")
@register_pass("compile")
@register_transformer("broadcast_greater")
class BroadcastGreater(Transformer):
    pass


@register_pass("rewrite")
@register_pass("validate")
@register_pass("calculate_ops")
@register_pass("prepare_for_compile")
@register_transformer("Concat")
class Concat(Transformer):
    def fuse_transpose(self, op, **kwargs):
        """ Customized fuse_transpose pass Introduction.

            Suppose the inputs are all 'Transpose' about 'axis':

            .. code-block:: none

                    cA         cB   ..    cC
                    |          |          |
                    |          |          |
                Transpose  Transpose  Transpose
                  (axis)    (axis)  ..  (axis)
                       \       |        /
                        \      |       /
                         \     |      /
                          \    |     /
                             Concat

            then, the graph can be transformed into:

            .. code-block:: none

                cA    cB .. cC
                 \    |    /
                  \   |   /
                   \  |  /
                    Concat
                      |
                  Transpose
                    (axis)

            where 'Transpose' here is also about 'axis'.
        """
        name, childs = op.attr('name'), sym_iter(op.get_children())
        if any([c.attr('op_name') != Transpose.op_name for c in childs]):
            return op
        axeses = [tuple(get_attr(c.list_attr(), 'axes')) for c in childs]
        axeses = set([axes for axes in axeses])
        if len(axeses) == 1:
            dim = get_attr(op.list_attr(), 'dim')
            axes = get_attr(childs[0].list_attr(), 'axes')
            Xs = [X.get_children()[0] for X in childs]
            op = mx.sym.concat(*Xs, dim=axes[dim], name=name)
            op = mx.sym.transpose(op, axes=axes, name=N.n('fuse_transpose'))
        return op

    def quantize(self, op, **kwargs):
        """ Customized quantize pass Introduction.

            See :func:`mrt.tfm_ops._quantize_scale <._quantize_scale>` for reference.
        """
        return _quantize_scale(op, **kwargs)

    def compile(self, op, **kwargs):
        attrs = kwargs['attr']
        childs = kwargs['childs']
        op_name = 'concatenate'
        new_attrs = {'axis': get_attr(attrs, 'dim', 1)}
        return get_nnvm_op(op_name)(*childs, name=N.n('concat'), **new_attrs)


@register_pass("prepare_for_compile")
@register_pass('compile')
@register_pass("rewrite")
@register_transformer("sum")
class Sum(Transformer):
    def validate(self, op, **kwargs):
        X, attr = sym_iter(op.get_children())[0], op.list_attr()
        name = op.attr('name')
        xshp = kwargs['infer_shapes'][X.attr('name')][get_entry_id(X)]
        axis = get_attr(attr, 'axis', [])
        # convert exclude into False
        if get_attr(attr, 'exclude', False):
            attr['axis'] = [i for i, _ in enumerate(xshp) if i not in axis]
            attr['exclude'] = False
            if len(attr['axis']) == 0:
                return X
        op = mx.sym.sum(X, **attr, name=name)
        return op

    def fuse_transpose(self, op, **kwargs):
        """ Customized fuse_transpose pass Introduction.

             Suppose 'keepdims' is True and the input is 'Transpose'.

             .. code-block:: none

                       cX
                       |
                 Transpose(axis)
                       |
                   sum(dims1)

             then, the graph can be transformed into:

             .. code-block:: none

                       cX
                       |
                   Sum(dims2)

             where:

             .. code-block:: python

                 dims2 = [axis[i] for i in dims1]
        """
        name, attr, X = op.attr('name'), op.list_attr(), op.get_children()[0]
        xshp = kwargs['infer_shapes'][X.attr('name')][get_entry_id(X)]
        axis = get_attr(attr, 'axis', [i for i in range(len(xshp))])
        keepdims = get_attr(attr, 'keepdims', False)
        if X.attr('op_name') == Transpose.op_name and not keepdims:
            axes, op = get_attr(X.list_attr(), 'axes'), X.get_children()[0]
            axis = [axes[i] for i in axis]
            op = mx.sym.sum(op, axis=axis, keepdims=keepdims, name=name)
        return op

    def calculate_ops(self, op, **kwargs):
        infer_shapes = kwargs['infer_shapes']
        oshp = infer_shapes[op.attr('name')][get_entry_id(op)]
        X = sym_iter(op.get_children())[0]
        ishp = infer_shapes[X.attr('name')][get_entry_id(X)]
        kwargs['base_ops'] = np.product(oshp) / np.product(ishp)
        return super().calculate_ops(op, **kwargs)

    def quantize(self, op, **kwargs):
        """ Customized quantize pass Introduction.

            .. math::
                Xq, xprec, xs = requant(X, oprec)

            where 'oprec' stands for the default quantization 
            precision for 'Sum'.

            See :func:`mrt.tfm_utils.requant <.requant>` for reference.

            .. math::
                k = int(product(ishp) / product(oshp))

            .. math::
                kprec = get\_bit\_cnt(k)

            where 'ishp' and 'oshp' respectively stands for the 
            input shape and output shape.

            See :func:`mrt.tfm_utils.get_bit_cnt <.get_bit_cnt>` for reference.

            The infer precision equals to 'xprec' plus 'kprec'.
        """
        infer_shapes = kwargs['infer_shapes']
        scales = kwargs['scales']
        name, op_name = op.attr('name'), op.attr('op_name')
        childs, attr = sym_iter(op.get_children()), op.list_attr()
        cns = [c.attr('name') for c in childs] if childs else []
        oshp = infer_shapes[name][get_entry_id(op)]

        oprec = kwargs['op_input_precs'][op_name]
        X, xprec, xs = requant_operator(childs[0], oprec, oname=name, **kwargs)
        scales[name] = xs
        op = get_mxnet_op(op_name)(X, **attr, name=name)

        ishp = infer_shapes[cns[0]][get_entry_id(childs[0])]
        k = int(nd.prod(nd_array(ishp)).asscalar() / \
            nd.prod(nd_array(oshp)).asscalar())
        kprec = get_bit_cnt(k)
        infer_prec = kprec + xprec
        kwargs['precs'][name][OUT_KEY] = infer_prec

        logger = logging.getLogger('log.mrt.realize')
        logger.debug("operator  %-20s name=%-40s oscale=%s, iscale=%s",
                     op_name, name, scales[name], cns)
        return op


@register_pass("validate")
@register_pass("fuse_transpose")
@register_pass("prepare_for_compile")
@register_transformer("BatchNorm")
class BatchNorm(Transformer):
    def rewrite(self, op, **kwargs):
        """ Customized rewrite pass Introduction.

            **Case 1. Convolution Input**

            When the graph looks like this:

            .. code-block:: none

                   X    W    B
                    \   |   /
                   Convolution gamma beta data_mean data_var
                        |        |     |     |         |
                        |        |     |     |         |
                        --------------------------------
                                       |
                                   BatchNorm

            .. math::
                sc = gamma / sqrt(data_var + eps)

            .. math::
                weight = W * sc.reshape(sc.shape, 1, 1, 1)

            .. math::
                bias = beta - sc * data\_mean + B

            .. math::
                op = Convolution(X, weight, bias, conv_attr)

            where 'conv_attr' is the attribute of 'X'.

            **Case 2. Other Cases**

            .. code-block:: none

                        X  gamma beta data_mean data_var
                        |    |     |     |         |
                        |    |     |     |         |
                        ----------------------------
                                   |
                               BatchNorm

            .. code-block:: python

                rshp = [s if i == axis else 1 for i, s in enumerate(ishp)]

            where 'axis' is attribute of operator 
            and ishp is the input shape of 'X'

            .. math::
                weight = reshape(rc, rshp)

            .. math::
                bias = reshape(beta - sc * data\_mean, rshp)

            .. math::
                op = X * weight + bias
        """
        params, infer_shapes = kwargs["params"], kwargs["infer_shapes"]
        name = op.attr('name')
        childs, attr = sym_iter(op.get_children()), op.list_attr()
        X, X_name = childs[0], childs[0].attr('name')
        gamma = params[childs[1].attr('name')]
        beta = params[childs[2].attr('name')]
        data_mean = params[childs[3].attr('name')]
        data_var = params[childs[4].attr('name')]

        fix_gamma = get_attr(attr, 'fix_gamma', True)
        gamma = 1 if fix_gamma else gamma
        axis = get_attr(attr, 'axis', 1)

        epsilon = float(attr['eps']) if 'eps' in attr else 1e-5
        sc = gamma / nd.sqrt(data_var + epsilon)
        bias = beta - sc * data_mean

        if X.attr('op_name') == 'Convolution':
            # Since convolution is "NCHW" format, axis must be one
            assert axis == 1, "Channel in input must be axis 1"
            cchilds, cattr = sym_iter(X.get_children()), X.list_attr()

            conv_name = N.n(name)
            W_name = cchilds[1].attr('name')
            weight = params[W_name]
            wn = N.n(W_name)
            params[wn] = weight * sc.reshape(*sc.shape, 1, 1, 1)
            W = mx.sym.var(wn, shape=params[W_name].shape)

            B_name = N.n('bias')
            if not get_attr(cattr, 'no_bias', False):
                B_name = cchilds[2].attr('name')
                bias += params[B_name]
            params[B_name] = bias
            B = mx.sym.var(B_name, shape=bias.shape)

            cattr['no_bias'] = False
            op = mx.sym.Convolution(cchilds[0], W,
                                    B, **cattr, name=conv_name)
        else:
            ishp = infer_shapes[X_name][get_entry_id(X)]
            reshp = [s if i == axis else 1 for i, s in enumerate(ishp)]
            w_name = N.n('weight')
            params[w_name] = sc.reshape(reshp)
            W = mx.sym.var(w_name, shape=reshp)
            node = mx.sym.broadcast_mul(X, W, name=N.n("broadcast_mul"))
            bias_name = N.n('bias')
            params[bias_name] = bias.reshape(reshp)
            B = mx.sym.var(bias_name, shape=reshp)
            op = mx.sym.broadcast_add(node, B, name=N.n("broadcast_add"))
        return op

    def calculate_ops(self, op, **kwargs):
        kwargs['base_ops'] = 4
        return super().calculate_ops(op, **kwargs)


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


@register_pass("prepare_for_compile")
@register_transformer("floor")
class Floor(Transformer):
    def compile(self, op, **kwargs):
        return kwargs['childs'][0]


@register_pass("prepare_for_compile")
@register_transformer("ceil")
class Ceil(Transformer):
    def compile(self, op, **kwargs):
        return kwargs['childs'][0]


@register_pass("prepare_for_compile")
@register_transformer("round")
class Round(Transformer):
    def compile(self, op, **kwargs):
        return kwargs['childs'][0]


@register_pass("prepare_for_compile")
@register_transformer("fix")
class Fix(Transformer):
    def compile(self, op, **kwargs):
        return kwargs['childs'][0]


@register_pass("calculate_ops")
@register_pass("prepare_for_compile")
@register_transformer("Cast")
class Cast(Transformer):
    def compile(self, op, **kwargs):
        return kwargs['childs'][0]


@register_pass("calculate_ops")
@register_pass("fuse_transpose")
@register_pass("validate")
@register_pass("rewrite")
@register_pass("quantize")
@register_pass("prepare_for_compile")
@register_transformer("slice")
class Slice(Transformer):
    def _fix_attr(self, **kwargs):
        infer_shapes = kwargs['infer_shapes']
        childs, attr = kwargs['childs'], kwargs['attr']
        X = childs[0]
        cshape = infer_shapes[X.attr('name')][0]
        begin = get_attr(attr, 'begin')
        end = get_attr(attr, 'end')
        attr['begin'] = [0 if s is None else s for s in begin]
        attr['end'] = [cshape[i] if s is None else s for i, s in enumerate(end)]
        return attr

    def compile(self, op, **kwargs):
        attrs = self._fix_attr(**kwargs)
        childs = kwargs['childs']
        # TODO: check default value
        begin = attrs['begin']
        end = attrs['end']
        stride = get_attr(attrs, 'step', None)
        new_attrs = {'begin': begin, 'end': end}
        if stride is not None:
            new_attrs['stride'] = stride
        return get_nnvm_op('strided_slice')(childs[0],
                                            name=N.n('slice'),
                                            **new_attrs)


@register_pass("validate")
@register_pass("calculate_ops")
@register_pass("fuse_transpose")
@register_pass("rewrite")
@register_pass("quantize")
@register_pass("prepare_for_compile")
@register_transformer("Reshape")
class Reshape(Transformer):
    def compile(self, op, **kwargs):
        childs = kwargs['childs']
        attrs = kwargs['attr']
        op_name = 'reshape'
        new_attrs = {}
        new_attrs['shape'] = get_attr(attrs, 'shape', 'reshape')
        return get_nnvm_op(op_name)(*childs,
                                    name=N.n('reshape'), **new_attrs)


@register_pass("calculate_ops")
@register_transformer("Custom")
class Custom(Transformer):
    def validate(self, op, **kwargs):
        """ Customized validate pass Introduction.

            The op type only support 'cvm_clip', 
            `cvm_left_shift`, 'cvm_right_shift', 'cvm_lut'.
        """
        attr = op.list_attr()
        op_type = attr['op_type']
        assert op_type in ['cvm_clip', 'cvm_left_shift',
                           'cvm_right_shift', 'cvm_lut'], \
            "Invalid op_type:%s in Custom operator" % op_type
        return op

    def prepare_for_compile(self, op, **kwargs):
        # name = op.attr('name')
        # X = sym_iter(op.get_children())[0]
        # if is_fusable_cvm_precision(op) and is_fusable_cvm_precision(X):
        #     p1, s1 = fusable_cvm_precision_attr(op)
        #     p2, s2 = fusable_cvm_precision_attr(X)
        #     X = sym_iter(X.get_children())[0]
        #     op = realize(X, (s1 + s2), min(p1, p2),
        #             name=name)
        return op


    def compile(self, op, **kwargs):
        childs = kwargs['childs']
        attr = kwargs['attr']
        op_type = attr['op_type']
        new_attrs = {}
        if op_type == 'cvm_clip':
            new_attrs['precision'] = attr['precision']
            sym = get_nnvm_op(op_type)(*childs, name=N.n('cvm_clip'),
                                       **new_attrs)
        elif op_type == 'cvm_lut':
            new_attrs['in_dim'] = attr['in_dim']
            sym = get_nnvm_op(op_type)(*childs, name=N.n('cvm_lut'),
                                       **new_attrs)
        else:
            new_attrs['precision'] = attr['precision']
            new_attrs['shift_bit'] = attr['shift_bit']
            sym = get_nnvm_op(op_type)(*childs, name=N.n('cvm_shift'),
                                       **new_attrs)
        return sym


@register_pass("rewrite")
@register_pass("calculate_ops")
@register_pass("prepare_for_compile")
@register_transformer("clip")
class Clip(Transformer):
    def validate(self, op, **kwargs):
        return op
        # For Relu6 transformer
        # attrs = op.list_attr()
        # a_min = sutils.get_attr(attrs, "a_min")
        # a_max = sutils.get_attr(attrs, "a_max")
        # assert a_min == 0 and a_max > a_min

    def fuse_transpose(self, op, **kwargs):
        """ Customized fuse_transpose pass Introduction.

            See :func:`mrt.tfm_ops.reverse_transpose <.reverse_transpose>` for reference.
        """
        return reverse_transpose(op)

    def quantize(self, op, **kwargs):
        """ Customized quantize pass Introduction.

            .. math::
                amin = int(a\_min * iscale)

            .. math::
                amax = int(a\_max * iscale)

            where 'a_min' and 'a\_max' are attributes of op, 
            and 'iscale' is the input scale of the input.

            .. math::
                op = clip(x, amin, amax)
        """
        precs, scales = kwargs['precs'], kwargs['scales']
        th_dict = kwargs['th_dict']
        X = op.get_children()[0]
        name, X_name = op.attr('name'), X.attr('name')
        attrs = op.list_attr()

        # `a_max`, `a_min` and precision should be align with CVM-Runtime
        scales[name] = iscale = scales[X.attr('name')]
        a_min = int(sutils.get_attr(attrs, "a_min") * iscale)
        a_max = int(sutils.get_attr(attrs, "a_max") * iscale)
        precs[name][OUT_KEY] = get_bit(max(abs(a_min), a_max))
        return mx.sym.clip(X, a_min=a_min, a_max=a_max, name=name)

    def compile(self, op, **kwargs):
        childs = kwargs['childs']
        attrs = kwargs['attr']
        op_name, new_attrs = 'clip', {}
        new_attrs['a_min'] = get_attr(attrs, 'a_min')
        new_attrs['a_max'] = get_attr(attrs, 'a_max')
        return get_nnvm_op(op_name)(*childs, **new_attrs)


@register_pass("prepare_for_compile")
@register_transformer("_minimum")
class Minimum(Transformer):
    def compile(self, op, **kwargs):
        childs = kwargs['childs']
        attrs = kwargs['attr']
        op_name = 'broadcast_min'
        return get_nnvm_op(op_name)(*childs,
                                    name=N.n('_minimum'), **attrs)


@register_pass("prepare_for_compile")
@register_transformer("_maximum")
class Maximum(Transformer):
    def compile(self, op, **kwargs):
        childs = kwargs['childs']
        attrs = kwargs['attr']
        op_name = 'broadcast_max'
        return get_nnvm_op(op_name)(*childs,
                                    name=N.n('_maximum'), **attrs)


@register_pass("validate")
@register_pass("fuse_transpose")
@register_pass("rewrite")
@register_pass("quantize")
@register_pass("calculate_ops")
@register_pass("prepare_for_compile")
@register_pass("compile")
@register_transformer("max")
class Max(Transformer):
    pass


@register_pass("prepare_for_compile")
@register_pass("compile")
@register_transformer("min")
class Min(Transformer):
    pass

@register_pass("fuse_transpose")
@register_pass("rewrite")
@register_pass("quantize")
@register_pass("prepare_for_compile")
@register_transformer("argmax")
class Argmax(Transformer):
    def compile(self, op, **kwargs):
        childs = kwargs['childs']
        attrs = kwargs['attr']
        op_name = 'argmax'
        new_attrs = {}
        new_attrs['axis'] = get_attr(attrs, 'axis', 0)
        new_attrs['keepdims'] = get_attr(attrs, 'keepdims', False)
        return get_nnvm_op(op_name)(*childs,
                                    name=N.n('_argmax'), **new_attrs)


@register_pass("prepare_for_compile")
@register_transformer("argmin")
class Argmin(Transformer):
    def compile(self, op, **kwargs):
        childs = kwargs['childs']
        attrs = kwargs['attr']
        op_name = 'argmin'
        new_attrs = {}
        new_attrs['axis'] = get_attr(attrs, 'axis', 0)
        new_attrs['keepdims'] = get_attr(attrs, 'keepdims', False)
        return get_nnvm_op(op_name)(*childs,
                                    name=N.n('_argmin'), **new_attrs)


@register_pass("prepare_for_compile")
@register_pass("compile")
@register_transformer("abs")
class Abs(Transformer):
    pass


@register_pass("compile")
@register_pass("validate")
@register_pass("calculate_ops")
@register_pass("rewrite")
@register_pass("prepare_for_compile")
@register_transformer("elemwise_add")
class ElemwiseAdd(Transformer):
    def fuse_transpose(self, op, **kwargs):
        """ Customized fuse_transpose pass Introduction.

            See :func:`mrt.tfm_ops._quantize_scale <._quantize_scale>` for reference.
        """
        return _ft_multi_input(op)

    def quantize(self, op, **kwargs):
        """ Customized quantize pass Introduction.

            See :func:`mrt.tfm_ops._quantize_scale <._quantize_scale>` for reference.
        """
        return _quantize_scale(op, **kwargs)


@register_pass("compile")
@register_pass("validate")
@register_pass("calculate_ops")
@register_pass("rewrite")
@register_pass("prepare_for_compile")
@register_transformer("elemwise_sub")
class ElemwiseSub(Transformer):
    def fuse_transpose(self, op, **kwargs):
        """ Customized fuse_transpose pass Introduction.

            See :func:`mrt.tfm_ops._quantize_scale <._quantize_scale>` for reference.
        """
        return _ft_multi_input(op)

    def quantize(self, op, **kwargs):
        """ Customized quantize pass Introduction.

            See :func:`mrt.tfm_ops._quantize_scale <._quantize_scale>` for reference.
        """
        return _quantize_scale(op, **kwargs)


# @register_pass("compile")
# @register_pass("prepare_for_compile")
# @register_pass("rewrite")
@register_transformer("elemwise_mul")
class ElemwiseMul(Transformer):
    def fuse_transpose(self, op, **kwargs):
        return _ft_multi_input(op)

    def rewrite(self, op, **kwargs):
        """ validate the infer_shapes of lhs and rhs must be the same
            thus this op could be rewrite into broadcast_mul
            corresponding cvm op would be optimized at compile time
        """
        name, op_name = op.attr('name'), op.attr('op_name')
        childs = sym_iter(op.get_children())

        ln, rn = [c.attr('name') for c in childs]
        infer_shapes = kwargs['infer_shapes']
        lshp, rshp = infer_shapes[ln], infer_shapes[rn]
        assert lshp == rshp, \
            "lhs infer_shape: {}, rhs infer_shape: {}".format(lshp, rshp)

        lhs, rhs = childs
        op = mx.sym.broadcast_mul(lhs, rhs, name=name)
        return op

    # def quantize(self, op, **kwargs):
        # precs, scales = kwargs['precs'], kwargs['scales']
        # name, op_name = op.attr('name'), op.attr('op_name')
        # childs, attr = sym_iter(op.get_children()), op.list_attr()
        # cns = [c.attr('name') for c in childs] if childs else []

        # oprec = kwargs['op_input_precs'][op_name]
        # X, xprec, xs = requant(childs[0], oprec, oname=name, **kwargs)
        # W, wprec, ws = requant(childs[1], oprec, oname=name, **kwargs)
        # scales[name] = ws * xs
        # op = get_mxnet_op(op_name)(X, W, **attr, name=name)

        # infer_prec = xprec + wprec
        # kwargs['precs'][name][OUT_KEY] = infer_prec

        # logger = logging.getLogger('log.mrt.realize')
        # logger.debug(
            # "operator  %-20s name=%-40s oscale=%s, iscale=%s",
            # op_name, name, scales[name], cns)
        # return op


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


@register_pass("validate")
@register_pass("calculate_ops")
@register_pass("rewrite")
@register_pass("quantize")
@register_transformer("_arange")
class Arange(Transformer):
    pass


@register_pass("validate")
@register_pass("calculate_ops")
@register_pass("fuse_transpose")
@register_pass("rewrite")
@register_pass("quantize")
@register_pass("prepare_for_compile")
@register_pass("compile")
@register_transformer("tile")
class Tile(Transformer):
    pass

@register_pass("validate")
@register_pass("calculate_ops")
@register_pass("fuse_transpose")
@register_pass("rewrite")
@register_pass("quantize")
@register_pass("prepare_for_compile")
@register_pass("compile")
@register_transformer("negative")
class Negative(Transformer):
    pass


@register_pass("validate")
@register_pass("calculate_ops")
@register_pass("fuse_transpose")
@register_transformer("SwapAxis")
class SwapAxis(Transformer):
    def rewrite(self, op, **kwargs):
        """ Customized rewrite pass Introduction.

            .. math::
                ndims = len(ishp)

            where 'ishp' is the infer shape of the input.

            .. math::
                new_axis = range(ndims)

            .. math:
                new_axis[dim1] = dim2

            .. math:
                new_axis[dim2] = dim1

            where 'dim1' and 'dim2' is the attributes of op.
        """
        name = op.attr('name')
        attr, childs = op.list_attr(), sym_iter(op.get_children())

        dim1, dim2 = get_attr(attr, 'dim1', 0), get_attr(attr, 'dim2', 0)
        if dim1 == dim2:
            return childs[0]
        ndims = len(kwargs['infer_shapes'][name][get_entry_id(childs[0])])
        new_axes = [i for i in range(ndims)]
        new_axes[dim1], new_axes[dim2] = dim2, dim1
        return mx.sym.transpose(childs[0], tuple(new_axes), name=name)


@register_pass("validate")
@register_pass("calculate_ops")
@register_pass("fuse_transpose")
@register_transformer("_plus_scalar")
class PlusScalar(Transformer):
    def rewrite(self, op, **kwargs):
        """ Customized rewrite pass Introduction.

            Transform into broadcast_add.
        """
        graph, params = kwargs['graph'], kwargs['params']
        name = op.attr('name')
        attr, childs = op.list_attr(), sym_iter(op.get_children())

        scalar = get_attr(attr, 'scalar')
        if scalar == 0:
            return childs[0]
        offset = nd_const(scalar, graph, params)
        return mx.sym.broadcast_add(childs[0], offset, name=name)


@register_pass("validate")
@register_pass("calculate_ops")
@register_pass("fuse_transpose")
@register_transformer("zeros_like")
class ZerosLike(Transformer):
    def rewrite(self, op, **kwargs):
        """ Customized quantize pass Introduction.

            Make constant zeros with fixed shape.
        """
        graph, params = kwargs['graph'], kwargs['params']
        name = op.attr('name')
        childs = sym_iter(op.get_children())

        mul_zero = nd_const(0, graph, params)
        op = mx.sym.broadcast_mul(childs[0], mul_zero, name=name)
        return op



@register_pass("validate")
@register_pass("calculate_ops")
@register_pass("fuse_transpose")
@register_transformer("ones_like")
class OnesLike(Transformer):
    def rewrite(self, op, **kwargs):
        """ Customized quantize pass Introduction.

            Make constant ones with fixed shape.
        """
        graph, params = kwargs['graph'], kwargs['params']
        name = op.attr('name')
        childs = sym_iter(op.get_children())

        mul_zero = nd_const(0, graph, params)
        op = mx.symbol.broadcast_mul(childs[0], mul_zero)
        add_one = nd_const(1, graph, params)
        op = mx.sym.broadcast_add(op, add_one, name=name)
        return op


@register_pass("calculate_ops")
@register_pass("fuse_transpose")
@register_pass("rewrite")
@register_pass("prepare_for_compile")
@register_transformer("_greater_scalar")
class GreaterScalar(Transformer):
    def validate(self, op, **kwargs):
        """ Customized validate pass Introduction.

            Only support integer scalar.
        """
        attr = op.list_attr()

        scalar = int(get_attr(attr, 'scalar', None))
        assert int(scalar) == scalar
        return op

    def compile(self, op, **kwargs):
        childs, attr = kwargs['childs'], kwargs['attr']

        scalar = int(attr['scalar'])
        prec = get_bit(scalar)
        var = cvm.sym.var(N.n('greater_scalar'), shape=(1,),
                                precision=str(prec))
        kwargs['params'][var.attr('name')] = nd_array([scalar])
        op = cvm.symbol.broadcast_greater(childs[0], var, name=N.n())
        return op


@register_pass("calculate_ops")
@register_pass("validate")
@register_pass("rewrite")
@register_pass("fuse_transpose")
@register_pass("prepare_for_compile")
@register_pass("compile")
@register_transformer("where")
class Where(Transformer):
    pass


@register_pass("validate")
@register_pass("calculate_ops")
@register_pass("fuse_transpose")
@register_pass("rewrite")
@register_pass("quantize")
@register_pass("prepare_for_compile")
@register_pass("compile")
@register_transformer("squeeze")
class Squeeze(Transformer):
    pass


# @register_pass("fuse_transpose")
# @register_pass("rewrite")
# # @register_pass("prepare_for_compile") # only for restore
# @register_transformer("L2Normalization")
# class L2Normalization(Transformer):
#     def quantize(self, op, **kwargs):
#         """ Customized quantize pass Introduction.

#             **Step 1. Requant Input**

#             .. math::
#                 Xq, xprec, xs = requant(X, oprec)

#             where 'oprec' is default quantize precision for 'L2Normalization'.

#             See :func:`mrt.tfm_utils.requant <.requant>` for reference.

#             **Step 2. Equivalent Transformation**

#             .. math::
#                 product = Xq * Xq

#             .. math::
#                 scale\_product = xs * xs

#             Consider the attribute 'mode', if it's "channel":

#             .. math::
#                 axis = [1]

#             If 'mode' is "instance":

#             .. math::
#                 axis = [1,2,3]

#             If 'mode' is "spatial":

#             .. math::
#                 axis = [2,3]

#             .. math::
#                 sum\_reduce = sum(product, axis)

#             .. math::
#                 epsilon = int(eps * scale\_product)

#             where 'eps' is the attribute, which is a small constant for numerical stability.

#             .. math::
#                 add_eps = sum\_reduce + eps

#             .. math::
#                 r = sqrt(add\_eps)

#             .. code-block:: python

#                 reps = tuple(
#                     [shp if i in axis else 1
#                     for i, shp in enumerate(list(shape))])

#             where 'shape' is the infer shape of the input.

#             Then, exert tile operation on 'r':

#             .. math::
#                 tile\_r = rile(r, reps)

#             .. math::
#                 op = Xq / tile\_r
#         """
#         scales = kwargs['scales']
#         name, op_name = op.attr('name'), op.attr('op_name')
#         attrs, childs = op.list_attr(), sym_iter(op.get_children())
#         cns = [c.attr('name') for c in childs]
#         X, xname = childs[0], cns[0]

#         # broadcast_mul
#         oprec = kwargs['op_input_precs'][op_name]
#         X, xprec, xs = requant(X, oprec, oname=name, **kwargs)
#         product = mx.sym.broadcast_mul(X, X, name=N.n('L2norm_mul'))
#         scale_product = xs*xs

#         # sum
#         # TODO(ryt): precision check align with runtime infer precision
#         mode = attrs.get('mode', 'instance')
#         if mode == "channel":
#             axis = [1]
#         elif mode == "instance":
#             axis = [1,2,3]
#         elif mode == "spatial":
#             axis = [2,3]
#         else:
#             assert "not valid `mode` type: %s" % mode
#         shape = kwargs['infer_shapes'][xname][get_entry_id(X)]
#         reps = tuple([shp if i in axis else 1 for i, shp in enumerate(list(shape))])

#         sum_reduce = mx.sym.sum(product, axis=axis, name=N.n('l2norm_sum'), keepdims=True)

#         # broadcast_add eps
#         eps_val = int(eval(attrs.get('eps', '1e-10')) * scale_product)
#         eps = nd_const(eps_val, kwargs['graph'], kwargs['params'])
#         add_eps = mx.sym.broadcast_add(sum_reduce, eps, N.n('l2norm_add'))

#         # get root
#         op = mx.sym.sqrt(add_eps, N.n('l2norm_root'))

#         # exert `tile` on `op`
#         # to get the same shape as 'X'
#         op = mx.sym.tile(op, reps=reps)

#         # since `op` and `X`
#         op = mx.sym.broadcast_div(X, op, name=name)
#         scales[name] = 1
#         prec = kwargs['precs'][name][OUT_KEY] = get_bit(kwargs['th_dict'][name])

#         logger = logging.getLogger('log.mrt.realize')
#         logger.debug("operator  %-20s name=%-40s oscale=%s, iscale=%s",
#                      op_name, name, scales[name], cns)
#         return op


#  @register_pass("prepare_for_compile")
#  @register_pass("compile")
#  @register_transformer("sqrt")
#  class Sqrt(Transformer):
#      def quantize(self, op, **kwargs):
#          """ Customized quantize pass Introduction.

#              The quantized scale equals to the square root of input scale.
#          """
#          precs, scales = kwargs["precs"], kwargs["scales"]
#          th_dict = kwargs["th_dict"]
#          name, op_name = op.attr("name"), op.attr("op_name")
#          X = op.get_children()[0]
#          xs = scales[X.attr("name")]

#          oscale = scales[name] = math.sqrt(xs)
#          precs[name][OUT_KEY] = get_bit(th_dict[name]*oscale)

#          op = mx.sym.sqrt(X, name=name)
#          return op


#  @register_pass("fuse_transpose")
#  @register_transformer("InstanceNorm")
#  class InstanceNorm(Transformer):
#      def rewrite(self, op, **kwargs):
#          """ Customized quantize pass Introduction.

#              Dynamic shape fusion verison of InstanceNorm operator equivalent transform  function.

#              .. code-block:: python

#                  axis = [i for i in range(len(xshp)) if i != 1]

#              where xshp is the infer shape of the input operator, which is of layout 'NCH...'.

#              The mean of X is equivalently calculated as:

#              .. math::
#                  sum\_x = sum(X, axis=axis, keepdims=True)

#              .. math::
#                  mul = 1/product(xshp[2:])

#              .. math::
#                  mena = sum\_x * mul.

#              The variance can be calculated as follows:

#              .. math::
#                  dev = X - mean

#              .. math::
#                  dev\_mul = dev * dev

#              .. math::
#                  var = sum(dev_mul, axis=axis, keepdims=True) * mul

#              The standard deviation can be calculated as follows:

#              .. math::
#                  std = sqrt(var) + eps

#              where eps is the attribute to prevent zero division.

#              The equivalent operator is finally calculated:

#              .. math::
#                  frac = dev / std

#              .. math::
#                  op = frac * gamma + beta

#          """
#          infer_shapes = kwargs["infer_shapes"]
#          name, op_name = op.attr("name"), op.attr("op_name")
#          attrs, childs = op.list_attr(), sym_iter(op.get_children())
#          cns = [c.attr('name') for c in childs]
#          X, gamma, beta = childs

#          xshp = infer_shapes[cns[0]][get_entry_id(X)]
#          assert len(xshp) >= 3
#          gshp = infer_shapes[cns[1]][get_entry_id(gamma)]
#          bshp = infer_shapes[cns[2]][get_entry_id(beta)]
#          assert len(gshp) == len(bshp) == 1 and \
#              gshp[0] == bshp[0] == xshp[1]

#          axis = [i for i in range(len(xshp)) if i != 1]
#          for i in axis:
#              gamma = mx.sym.expand_dims(
#                  gamma, axis=i, name=N.n("expand_dims"))
#              beta = mx.sym.expand_dims(
#                  beta, axis=i, name=N.n("expand_dims"))

#          # TODO(ryt_tune): get batch_axes from **kwargs
#          batch_axes = [0]
#          assert len(batch_axes) == 1 and batch_axes[0] == 0, \
#              "Only NCH... format is supported for op: (%s). " + \
#              "name: (%s), batch_axes: (%s)." % \
#              (op_name, name, batch_axes)

#          sum_x = mx.sym.sum(
#              X, axis=axis, keepdims=True, name=N.n("sum"))
#          mul = nd_const(
#              1/np.product(xshp[2:]),
#              kwargs["graph"], kwargs["params"])
#          mean = mx.sym.broadcast_mul(
#              sum_x, mul, name=N.n("broadcast_mul"))
#          dev = mx.sym.broadcast_sub(X, mean, name=N.n("broadcast_sub"))
#          dev_mul = mx.sym.broadcast_mul(
#              dev, dev, name=N.n("broadcast_mul"))
#          var = mx.sym.broadcast_mul(
#              mx.sym.sum(dev_mul, axis=axis, keepdims=True, name=N.n("sum")),
#              mul, name=N.n("broadcast_mul"))
#          eps = nd_const(
#              get_attr(attrs, "eps", 0.00100000005),
#              kwargs["graph"], kwargs["params"])
#          std = mx.sym.broadcast_add(
#              mx.sym.sqrt(var, name=N.n("sqrt")),
#              eps, name=N.n("broadcast_add"))
#          frac = mx.sym.broadcast_div(
#              dev, std, name=N.n("broadcast_div"))
#          frac_mul = mx.sym.broadcast_mul(
#              frac, gamma, name=N.n("broadcast_mul"))
#          op = mx.sym.broadcast_add(frac_mul, beta, name=name)
#          return op


@register_pass("fuse_transpose")
@register_transformer("batch_dot")
class BatchDot(Transformer):
    def rewrite(self, op, **kwargs):
        """ Customized rewrite pass Introduction.

            Using matrix decomposition to avoid overflow.

            .. math::
                Y = A \cdot B
                  = A_1 \cdot B_1 + A_2 \cdot B_2 + ...

            where

            .. math::
                A_i\\text{.shape} = (batch, M, step)

            .. math::
                B_i\\text{.shape} = (batch, step, N)
        """
        infer_shapes = kwargs["infer_shapes"]

        name = op.attr("name")
        attrs, childs = op.list_attr(), sym_iter(op.get_children())
        A, B = childs
        an, bn = [c.attr("name") for c in childs]

        transpose_a = get_attr(attrs, "transpose_a", 0)
        transpose_b = get_attr(attrs, "transpose_b", 0)
        ashp = infer_shapes[an][get_entry_id(A)]
        bshp = infer_shapes[bn][get_entry_id(B)]

        andims, bndims = len(ashp), len(bshp)
        assert andims == 3 and bndims == 3, \
            "batch_dot currently only support 3D*3D array." + \
            "name: (%s), op_name: (%s)" % (name, op_name)

        if transpose_a:
            ashp = ashp[:-2] + (ashp[-1], ashp[-2])
            axes = tuple(range(andims-2)) + (andims-1, andims-2)
            A = mx.sym.transpose(
                A, axes=axes, name=N.n("transpose_a"))

        if transpose_b:
            bshp = bshp[:-2] + (bshp[-1], bshp[-2])
            bndims = len(bshp)
            axes = tuple(range(bndims-2)) + (bndims-1, bndims-2)
            B = mx.sym.transpose(
                B, axes=axes, name=N.n("transpose_b"))

        assert ashp[-1] == bshp[1]
        C, MATRIX_MAXIMUM_SIZE = ashp[-1], 65536
        if ashp[-1] <= MATRIX_MAXIMUM_SIZE:
            return mx.sym.batch_dot(A, B, name=name)

        C, nodes, step, start = \
            ashp[-1], [], MATRIX_MAXIMUM_SIZE, 0

        while start < C:
            stop = min(start+step, C)

            begin, end = (0,0,start), (ashp[0],ashp[1],stop)
            Ak = mx.sym.slice(
                A, begin=begin, end=end, name=N.n("slice_a"))

            begin, end = (0,start,0), (bshp[0],stop,bshp[2])
            Bk = mx.sym.slice(
                B, begin=begin, end=end, name=N.n("slice_b"))

            tmp = mx.sym.batch_dot(
                Ak, Bk, name=N.n("batch_dot"))
            nodes.append(tmp)
            start += step

        while len(nodes) > 1:
            A, B = nodes.pop(0), nodes.pop(0)
            nname = N.n("elemwise_add") if nodes else name
            tmp = mx.sym.elemwise_add(A, B, name=nname)
            nodes.append(tmp)

        op = nodes[0]
        return op

    def quantize(self, op, **kwargs):
        """ Customized quantize pass Introduction.

            The inputs are quantized into the same precision level.
        """
        precs, scales = kwargs["precs"], kwargs["scales"]
        th_dict = kwargs["th_dict"]

        name, op_name = op.attr("name"), op.attr("op_name")
        childs, attr = sym_iter(op.get_children()), op.list_attr()
        cns = [c.attr("name") for c in childs]

        oprec = kwargs["op_input_precs"][op_name]
        A, aprec, ascale = requant(
            childs[0], oprec, oname=name, **kwargs)
        B, bprec, bscale = requant(
            childs[1], oprec, oname=name, **kwargs)

        op = get_mxnet_op(op_name)(A, B, **attr, name=name)
        scales[name] = ascale * bscale
        precs[name][OUT_KEY] = aprec + bprec

        logger = logging.getLogger('log.mrt.realize')
        logger.debug("operator  %-20s name=%-40s oscale=%s, iscale=%s",
                     op_name, name, scales[name], cns)
        return op


@register_pass("fuse_transpose")
@register_transformer("broadcast_like")
class BroadcastLike(Transformer):
    def _broadcast_like(self, X, W, name, **kwargs):
        graph, params = kwargs["graph"], kwargs["params"]
        one = nd_const(1, graph, params)
        zero = nd_const(0, graph, params)
        nW = mx.sym.broadcast_add(
            mx.sym.broadcast_mul(W, zero, name=N.n("broadcast_mul")),
            one, name=N.n("broadcast_add"))
        op = mx.sym.broadcast_mul(X, nW, name=name)
        return op

    def rewrite(self, op, **kwargs):
        """ Customized rewrite pass Introduction.

            The original operator:

            .. math::
                op = broadcast\_like(X, W, lhs\_axes, rhs\_axes)

            can be transformed in threee conditions.

            *Case 1. Null Attributes*

            .. math::
                mul = broadcast\_mul(W, 0)

            .. math::
                add = broadcast\_add(mul, 1)

            .. math::
                op = broadcast\_mul(X, add)

            *Case 2. Batch Axis not in Attributes*

            Calculate attributes for tile as follows:

            .. code-block:: python

                cnts = {v: wshp[rhs_axes[i]] for i, v in enumerate(lhs_axes)}
                reps = tuple([cnts[v] if v in lhs_axes else 1 for v in range(xndims)])

            where wshp is the shape of input weight.

            .. math::
                op = tile(X, reps=reps)

            *Case 3. Other Cases*

            In this case, we only support injection of  batchaxis from lhs_axes to rhs_axes.

            After transformation of weight (see source code for ref), result was calculated as in case 1.

        """
        infer_shapes = kwargs["infer_shapes"]
        name, op_name = op.attr("name"), op.attr("op_name")
        attrs, childs = op.list_attr(), sym_iter(op.get_children())
        cns = [c.attr("name") for c in childs]
        X, W = childs

        xshp = infer_shapes[cns[0]][get_entry_id(X)]
        xndims = len(xshp)
        wshp = infer_shapes[cns[1]][get_entry_id(W)]
        wndims = len(wshp)
        lhs_axes = get_attr(attrs, "lhs_axes", None)
        rhs_axes = get_attr(attrs, "rhs_axes", None)

        if lhs_axes is None or rhs_axes is None:
            assert xndims == wndims and lhs_axes is None \
                and rhs_axes is None
            return self._broadcast_like(X, W, name, **kwargs)

        lhs_axes, lndims = list(lhs_axes), len(lhs_axes)
        rhs_axes, rndims = list(rhs_axes), len(rhs_axes)
        assert lndims == rndims > 0

        lhs_axes = tuple([v+xndims if v<0 else v for v in lhs_axes])
        assert all([0<=v<xndims for v in list(lhs_axes)])

        rhs_axes = tuple([v+wndims if v<0 else v for v in rhs_axes])
        assert all([0<=v<wndims for v in list(rhs_axes)])

        assert all([xshp[lhs_axes[i]] == 1 for i in range(lndims)])

        # TODO(ryt_tune): get batch_axes from **kwargs
        batch_axes = [0]
        flg = all([batch_axis not in rhs_axes \
            for batch_axis in batch_axes])
        if flg:
            cnts = {v: wshp[rhs_axes[i]] \
                for i, v in enumerate(lhs_axes)}
            reps = tuple([cnts[v] if v in lhs_axes else 1 \
                for v in range(xndims)])
            return mx.sym.tile(X, reps=reps, name=name)

        axis_map = {}
        for i, v in enumerate(lhs_axes):
            axis_map[v] = rhs_axes[i]
        for batch_axis in batch_axes:
            assert sum([1 if v == batch_axis else 0 \
                for k, v in axis_map.items()]) <= 1, \
                "multiple broadcast on batch_axis: (%s), " + \
                "which is not support by dynamic shape fusion. " + \
                "name: (%s), op_name: (%s)" % (batch_axis, name, op_name)
        assert wndims < 6, "slice can manipulate at most 5d, " + \
            "name: (%s), op_name: (%s)" % (name, op_name)

        # reduce shape to 1 for non-broadcast dimensions
        begin = tuple([0]*wndims)
        end = tuple([wshp[v] if v in axis_map.values() else 1 \
            for v in range(wndims)])
        W = mx.sym.slice(
            W, begin=begin, end=end, name=N.n("slice"))

        # decompose k1->v, k2->v into k1->v, k2->v2
        # which make axis
        while True:
            vs, flag, paxis_map = set(), True, axis_map
            for pk, pv in paxis_map.items():
                if pv not in vs:
                    vs.add(pv)
                    continue
                flag = False
                axis_map = {k: (v+1 if v>pv or k==pk else v) \
                    for k, v in axis_map.items()}
                W = mx.sym.expand_dims(
                    W, axis=pv, name=N.n("expand_dims"))
                W = mx.sym.repeat(
                    W, axis=pv, repeats=wshp[pv], name=N.n("repeat"))
                wshp = wshp[:pv] + (wshp[pv],) + wshp[pv:]
                break
            if flag:
                break
        wndims = len(wshp)

        # trim wndims if not equal to xndims
        v = 0
        while wndims > xndims:
            while v in axis_map.values():
                v += 1
            W = mx.sym.squeeze(W, axis=v, name=N.n("squeeze"))
            wndims -= 1
            axis_map = {k: (nv-1 if nv > v else nv) \
                for k, nv in axis_map.items()}
        while wndims < xndims:
            W = mx.sym.expand_dims(
                W, axis=wndims, name=N.n("expand_dims"))
            wndims += 1
        axes = list(range(wndims))
        while True:
            dels = [k for k, v in axis_map.items() if k==v]
            for k in dels:
                del axis_map[k]
            if not axis_map:
                break
            keys = list(axis_map.keys())
            k, v = keys[0], axis_map[keys[0]]
            axes[k], axes[v] = axes[v], axes[k]
            for nk in keys:
                nv = axis_map[nk]
                if nv == k:
                    axis_map[nk] = v
                elif nv == v:
                    axis_map[nk] = k
        axes = tuple(axes)
        if axes != tuple(range(wndims)):
            assert wndims < 7, \
                "slice can manipulate at most 6d"
            W = mx.sym.transpose(W, axes=axes, name=N.n("transpose"))

        op = self._broadcast_like(X, W, name, **kwargs)
        return op


@register_pass("fuse_transpose")
@register_transformer("reshape_like")
class ReshapeLike(Transformer):
    def rewrite(self, op, **kwargs):
        """ Customized rewrite pass Introduction.

            Equivalently transformed into reshape. 
            Since dynamic shape fusion is supported, only single 
            batch axis is supported rather than multiple ones.
        """
        infer_shapes = kwargs["infer_shapes"]
        name, op_name = op.attr("name"), op.attr("op_name")
        attrs, childs = op.list_attr(), sym_iter(op.get_children())
        cns = [c.attr("name") for c in childs]
        X, W = childs

        xshp = infer_shapes[cns[0]][get_entry_id(X)]
        xndims = len(xshp)
        wshp = infer_shapes[cns[1]][get_entry_id(W)]
        wndims = len(wshp)

        lhs_begin = get_attr(attrs, "lhs_begin", 0)
        lhs_end = get_attr(attrs, "lhs_end", xndims)
        rhs_begin = get_attr(attrs, "rhs_begin", 0)
        rhs_end = get_attr(attrs, "rhs_end", wndims)

        lhs_begin = lhs_begin+xndims if lhs_begin < 0 else lhs_begin
        lhs_end = lhs_end+xndims if lhs_end< 0 else lhs_end
        assert 0 <= lhs_begin < lhs_end <= xndims

        rhs_begin = rhs_begin+wndims if rhs_begin < 0 else rhs_begin
        rhs_end = rhs_end+wndims if rhs_end< 0 else rhs_end
        assert 0 <= rhs_begin < rhs_end <= wndims

        rshp = xshp[:lhs_begin] + \
            wshp[rhs_begin:rhs_end] + xshp[lhs_end:]

        # TODO(ryt_tune): get batch_axes from **kwargs
        batch_axes = [0]
        assert len(batch_axes) == 1, \
            "Dynamic batch shape fusion of (%s) only support" + \
            "single dimension of batch. Providied: (%s)" % \
            (op_name, batch_axes)
        batch_axis = batch_axes[0]
        rshp = rshp[:batch_axis] + (-1,) + rshp[batch_axis+1:]

        op = mx.sym.reshape(X, shape=rshp, name=name)
        return op

def _ft_multi_input(op):
    name, childs = op.attr('name'), sym_iter(op.get_children())
    # Assert all the inputs are transpose
    if any([c.attr('op_name') != Transpose.op_name for c in childs]):
        return op
    # Check all the inputs shapes are consistent
    axeses = [tuple(get_attr(c.list_attr(), 'axes')) for c in childs]
    axeses = set([axes for axes in axeses])
    # Fuse transpose
    if len(axeses) == 1:
        axes = get_attr(childs[0].list_attr(), 'axes')
        Xs = [X.get_children()[0] for X in childs]
        opname = op.attr('op_name')
        op = get_mxnet_op(opname)(*Xs, name=name)
        op = mx.sym.transpose(op, axes=axes, name=N.n('fuse_transpose'))
    return op

def _quantize_scale(op, **kwargs):
    """ quantization function with the inputs form of:

        .. math::
            Y = f(node1, node2, node3, ...)

        where node1, node2, node3, ... ought to be quantized to the same scale.

        The infer precision depends on the input with the maximum quantized tight precision.
    """
    scales = kwargs['scales']
    th_dict, precs = kwargs['th_dict'], kwargs['precs']
    name, op_name = op.attr('name'), op.attr('op_name')
    attr, childs = op.list_attr(), sym_iter(op.get_children())
    cns = [c.attr('name') for c in childs] if childs else []

    oprec = kwargs['op_input_precs'][op_name]
    in_th = max([th_dict[n] for n in cns])
    oscale = scales[name] = scale(in_th, oprec)
    new_childs = []
    cprecs = []
    for c in childs:
        c, cprec, _ = requant(c, oprec, oscale=oscale, oname=name, **kwargs)
        cprecs.append(cprec)
        new_childs.append(c)
    op = get_mxnet_op(op_name)(*new_childs, **attr, name=name)
    infer_prec = max(cprecs) if op_name in ['Concat'] else max(cprecs)+1
    precs[name][OUT_KEY] = infer_prec

    logger = logging.getLogger('log.mrt.realize')
    logger.debug("operator  %-20s name=%-40s oscale=%s, iscale=%s",
                 op_name, name, scales[name], cns)
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

disabled_restore_ops = {"null"}

def _restore(op, **kwargs):
    th_dict, precs, scales = kwargs['th_dict'], kwargs['precs'], kwargs['scales']
    name, op_name = op.attr('name'), op.attr('op_name')
    childs, attr = sym_iter(op.get_children()), op.list_attr()

    childs = [] if childs is None else childs

    new_childs = [c / scales[c.attr('name')] \
        if scales.get(c.attr('name'), 1) != 1 else c \
                 for c in childs]

    out = get_mxnet_op(op_name)(*new_childs, **attr, name=name)
    precs[name][OUT_KEY] = get_bit(th_dict[name])
    scales[name] = 1

    return out

def _quantize_table(op, **kwargs):
    """ quantization function with the inputs form of:

        .. math::
            Y = f(X) = g(exp(X))

        **Step 1. Requant Input**

        .. math::
            Xq, xprec, xs = requant\_operator(X, iprec, oscale)

        where 'Xq' is quantized symbol, 
        'xprec' and 'xs' stand for quantized precision and scale, 
        for reference of 'requant_operator', 
        see :func:`mrt.tfm_utils.requant_operator <.requant_operator>`.

        .. math::
            offset = BroadcastAdd(Xq, alpha)

        where alpha is the threshold of 'Xq'

        **Step 2. Create Lookup Table**

        .. math::
            r = range(-alpha, alpha+1) / xs

        where 'r' stands for the table range of the unscaled input.

        .. math::
            out = f(r)

        .. math::
            oscale = scale(||out||_2, xprec)


        for reference of 'scale', 
        see :func:`mrt.tfm_utils.scale <.scale>`.

        .. math::
            in\_dim = 2*alpha+1

        .. math::
            out\_q = round(out * oscale)

        Util now, the float-simulating-int process has been accomplished.

        .. math::
            W = reshape(round(out_q * oscale), in\_dim)

        Util now, the lookup table has been created.

        **Step 3. Get Output**

        The cvm customized operator 'cvm_lut' has been adopted.

        .. math::
            op = cvm\_lut(X, W, in\_dim)
    """
    params, graph = kwargs['params'], kwargs['graph']
    th_dict, precs, scales = kwargs['th_dict'], kwargs['precs'], kwargs['scales']
    name, op_name = op.attr('name'), op.attr('op_name')
    childs = sym_iter(op.get_children())
    cns = [c.attr('name') for c in childs] if childs else []

    iprec = kwargs['op_input_precs'][op_name]
    xs = scale(th_dict[cns[0]], iprec)
    # xs= scales[cns[0]]
    X, xprec, xs = requant_operator(childs[0], iprec, \
            oscale=xs, oname=name, **kwargs)
    alpha = get_range(xprec)
    var = nd_const(alpha, graph, params)
    X = mx.sym.broadcast_add(X, var, name=N.n(op_name+'_offset'))

    out = sutils.get_nd_op(op_name)(sutils.nd_arange(-alpha, alpha+1) / xs)
    oprec = precs[name].get(OUT_KEY, 16)
    oscale = scales[name] = scale(out.abs().max().asscalar(), oprec)

    W_name = N.n("cvm_lut_weight")
    params[W_name] = weight = (out * oscale).round().reshape(2*alpha+1, 1)
    wattr = {'precision': str(oprec)}
    W = graph[W_name] = mx.sym.var(W_name, shape=weight.shape, attr=wattr)
    op = mx.sym.Custom(X, W, in_dim=2*alpha+1, name=name, op_type='cvm_lut')
    precs[name][OUT_KEY] = oprec

    logger = logging.getLogger('log.mrt.realize')
    logger.debug("operator  %-20s name=%-40s oscale=%s, iscale=%s",
                 op_name, name, scales[name], cns)
    return op

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

def reverse_sequence(op):
    """
        Reverse the symbol sequenze may leads to
        error of the different result, due to the graph
        unequaivent transformer.

        **Example**

        .. code-block:: none

            A ->  B -> C
              |-> D -> E

        after reverse sequence is

        .. code-block:: none

            B -> A ->  C
                   |-> D -> E

        which is invalid.

        Notice:
            The fuse_transpose pass have the same hidden problems.
    """
    name, op_name = op.attr('name'), op.attr('op_name')
    childs, attrs = sutils.sym_iter(op.get_children()), op.list_attr()
    assert len(childs) == 1
    X = childs[0]
    t_name, t_attr = X.attr('name'), X.list_attr()
    t_opname = X.attr('op_name')
    assert len(X.get_children()) == 1
    X = X.get_children()[0]
    op = get_mxnet_op(op_name)(X, **attrs, name=name)
    op = get_mxnet_op(t_opname)(op, name=t_name, **t_attr)
    return op

def is_fusable_cvm_precision(op):
    return op.attr('op_name') == 'Custom' and \
        op.list_attr()['op_type'] in [
            'cvm_clip', 'cvm_right_shift']

def fusable_cvm_precision_attr(op):
    assert is_fusable_cvm_precision(op)
    attr = op.list_attr()
    return get_attr(attr, 'precision'), get_attr(attr, 'shift_bit', 0)
