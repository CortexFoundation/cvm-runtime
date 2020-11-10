import logging
import math
import numpy as np

from mxnet import ndarray as nd
import mxnet as mx
import cvm

from mrt.sym_utils import get_attr, sym_iter, is_params, is_inputs, \
                          nd_array, get_mxnet_op, get_nnvm_op, \
                          nd_const, get_entry_id, nd_full
from mrt.tfm_utils import realize
from mrt.tfm_base import N, MAX_BIT
from mrt.tfm_pass import OUT_KEY
from .tfm_base import register_pass, register_transformer, Transformer
from .tfm_types import get_quantizer, USQuantizer, UAQuantizer, \
                       FT_TYPE_EXP, AFeature, SBuffer, LAYER_WISE_TYPE, \
                       US_QUANT_TYPE, AFeature, MMFeature
from .tfm_utils import scale_exp, get_buffer_exp, get_bit_exp, \
                       get_range_exp, get_bit_cnt_exp
from .sym_utils import nd_full_const

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
class Flatten(tops.Flatten, Transformer):
    pass


@register_pass("validate")
@register_pass("rewrite")
@register_pass("fuse_transpose")
@register_pass("prepare_for_compile")
@register_transformer("null")
class Null(tops.Null, Transformer):
    def quantize(self, op, **kwargs):
        if is_inputs(op, kwargs['params']):
            name, attr = op.attr('name'), op.list_attr()
            prec = kwargs['precs'][name][OUT_KEY]
            ft = kwargs['features'][name]
            assert ft.name == FT_TYPE_EXP
            kwargs['buffers'][name] = get_buffer_exp(
                scale_exp(ft.get(), prec))
            extra_attr = {'precision': str(prec)}
            return mx.sym.var(name, **attr, attr=extra_attr)
        return op


@register_pass("fuse_transpose")
@register_pass("rewrite")
@register_pass("quantize")
@register_pass("prepare_for_compile")
@register_transformer("Pooling")
class Pooling(tops.Pooling, Transformer):
    pass


@register_pass("validate")
@register_pass("calculate_ops")
@register_pass("rewrite")
@register_pass("quantize")
@register_pass("prepare_for_compile")
@register_transformer("Dropout")
class Dropout(tops.Dropout, Transformer):
    pass


@register_pass("rewrite")
@register_pass("prepare_for_compile")
@register_pass("calculate_ops")
@register_pass("compile")
@register_pass("fuse_transpose")
@register_pass("validate")
@register_pass("quantize")
@register_transformer("Activation")
class Activation(tops.Activation, Transformer):
    pass


@register_pass("validate")
@register_pass("rewrite")
@register_pass("fuse_transpose")
@register_pass("prepare_for_compile")
@register_transformer("FullyConnected")
class FullyConnected(tops.FullyConnected, Transformer):
    def quantize(self, op, **kwargs):
        features, buffers = kwargs['features'], kwargs['buffers']
        precs = kwargs['precs']
        cfg_dict, params = kwargs['cfg_dict'], kwargs['params']
        name, op_name = op.attr('name'), op.attr('op_name')
        childs, attr = sym_iter(op.get_children()), op.list_attr()
        cns = [c.attr('name') for c in childs] if childs else []

        assert len(childs) == 2
        xquant_type = cfg_dict[cns[0]]['quant_type']
        wquant_type = cfg_dict[cns[1]]['quant_type']

        if xquant_type == wquant_type == USQuantizer.name:
            op = _quantize_xw(op, **kwargs)
        else:
            raise NotImplementedError(
                "Quantization type not implementated," + \
                " op: %20s, Xquant: %20s, Wquant: %20s",
                op_name, [xquant_type, wquant_type])

        logger = logging.getLogger('log.mrt.realize')
        logger.debug(
            "operator  %-20s name=%-40s oscale=%s, iscale=%s",
            op_name, name, buffers[name].serialize(), cns)
        return op


@register_pass("fuse_transpose")
@register_pass("prepare_for_compile")
@register_pass("rewrite")
@register_transformer("Convolution")
class Convolution(tops.Convolution, Transformer):
    def slice_channel(self, op, **kwargs):
        name, op_name = op.attr('name'), op.attr('op_name')
        attr, childs = op.list_attr(), sym_iter(op.get_children())
        cns = [c.attr('name') for c in childs]
        cfg_dict = kwargs['cfg_dict']
        infer_shapes = kwargs['infer_shapes']

        gn_info = cfg_dict[name]['gn_info']
        ichannel, step = gn_info['ichannel'], gn_info['step']
        assert ichannel == 1

        assert len(childs) == 2
        X, W = childs
        xshp = infer_shapes[cns[0]][get_entry_id(childs[0])]
        wshp = infer_shapes[cns[1]][get_entry_id(childs[1])]
        oshp = infer_shapes[name][get_entry_id(op)]
        assert len(xshp) == len(wshp) == 4 and xshp[1]%step == 0

        xi_cfg_info, wi_cfg_info = cfg_dict[cns[0]], cfg_dict[cns[1]]
        xi_cfg_info['gn_info'] = {'gn_type': LAYER_WISE_TYPE}
        wi_cfg_info['gn_info'] = {'gn_type': LAYER_WISE_TYPE}
        yi_cfg_info = {
            'gn_info': {'gn_type': LAYER_WISE_TYPE},
            'quant_type': US_QUANT_TYPE,
            'opt_info': cfg_dict[name]['opt_info'],
        }
        num_group = eval(attr['num_group'])
        C, IC, OC = xshp[1], wshp[1], wshp[0]
        assert num_group * IC == C and OC >= num_group and OC % num_group == 0
        if num_group == 1:
            # xs = sym_slice(X, ichannel, step, **kwargs)
            # ws = sym_slice(W, ichannel, step, **kwargs)
            # nodes = []
            # j = 0
            # for i in range(0, C, step):
                # suffix = '_' + str(i)+'-'+str(i+step)
                # xni = xs[j].attr('name')
                # cfg_dict[xni] = xi_cfg_info
                # wni = ws[j].attr('name')
                # cfg_dict[wni] = wi_cfg_info
                # yni = N.n(name+suffix)
                # Yi = get_mxnet_op(op_name)(xs[j], ws[j], **attr, name=yni)
                # cfg_dict[yni] = yi_cfg_info
                # nodes.append(Yi)
                # j += 1
            # assert len(nodes) > 1
            # op = mx.sym.add_n(*nodes, name=name)
            # transpose and reshape weight
            Wt = mx.sym.transpose(W, axes=(1,0,2,3), name=N.n('transpose'))
            rshp = (OC*IC,1,) + wshp[2:]
            wrn = N.n('reshape')
            cfg_dict[wrn] = wi_cfg_info
            Wr = mx.sym.reshape(Wt, shape=rshp, name=wrn)
            # groupwise convolution
            nattr = attr.copy()
            nattr['num_group'] = IC
            nattr['num_filter'] = IC * OC
            conv_name = N.n('groupwise_convolution')
            cfg_dict[conv_name] = yi_cfg_info
            op = mx.sym.Convolution(X, Wr, **nattr, name=conv_name)
            # reshape output
            rname = N.n('reshape')
            cfg_dict[rname] = yi_cfg_info
            rshp = (-1, IC, OC,) + oshp[2:]
            op = mx.sym.reshape(op, shape=rshp, name=rname)
            # sum
            sum_name = N.n('sum')
            cfg_dict[sum_name] = yi_cfg_info
            op = mx.sym.sum(op, axis=1, keepdims=False, name=sum_name)
        else:
            assert step == 1
            xs = sym_slice(X, ichannel, step, **kwargs)
            ws = kernel_slice_2d(W, **kwargs)
            OPG = OC // num_group
            nattr = attr.copy()
            nattr['num_group'] = '1'
            nattr['num_filter'] = '1'
            nodes = []
            for o in range(OC):
                nnodes = []
                j = int(o/OPG)*IC
                for i in range(IC):
                    suffix = '_' + str(o)+'-'+str(i)
                    k = i+j
                    xk, woi = xs[k], ws[o][i]
                    xnk, wnoi = xk.attr('name'), woi.attr('name')
                    cfg_dict[xnk] = xi_cfg_info
                    cfg_dict[wnoi] = wi_cfg_info
                    ynoi = N.n(name+suffix)
                    yoi = mx.sym.Convolution(xk, woi, **nattr, name=ynoi)
                    cfg_dict[ynoi] = yi_cfg_info
                    nnodes.append(yoi)
                if len(nnodes) > 1:
                    zni = N.n(name+'_add_n_'+str(o))
                    zi = mx.sym.add_n(*nnodes, name=zni)
                    cfg_dict[zni] = yi_cfg_info
                else:
                    zi = nnodes[0]
                nodes.append(zi)
            assert len(nodes) > 1
            op = mx.sym.concat(*nodes, dim=1, name=name)
        return op

    def quantize(self, op, **kwargs):
        features, buffers = kwargs['features'], kwargs['buffers']
        precs, graph = kwargs['precs'], kwargs['graph']
        cfg_dict, params = kwargs['cfg_dict'], kwargs['params']
        name, op_name = op.attr('name'), op.attr('op_name')
        childs, attr = sym_iter(op.get_children()), op.list_attr()
        cns = [c.attr('name') for c in childs] if childs else []

        assert len(childs) == 2 and 'pad' not in attr
        xquant_type = cfg_dict[cns[0]]['quant_type']
        wquant_type = cfg_dict[cns[1]]['quant_type']
        X, W = childs
        xquant, wquant = \
            get_quantizer(xquant_type), get_quantizer(wquant_type)
        oprec = kwargs['op_input_precs'][op_name]

        if xquant_type == wquant_type == USQuantizer.name:
            op = _quantize_xw(op, **kwargs)
        elif xquant_type == USQuantizer.name and \
            wquant_type == UAQuantizer.name:
            Xq, xprec, xscale = xquant.quantize(
                X, oprec, oname=name, **kwargs)
            Wq, wprec, wscale, wzpoint = wquant.quantize(
                W, oprec, oname=name, **kwargs)
            buffers[name] = get_buffer_exp(xscale*wscale)

            Ye1 = mx.sym.Convolution(Xq, Wq, **attr, name=N.n('Convolution'))
            wshp = params[cns[1]].shape
            pd = int(np.product(wshp[1:]))
            infer_prec1 = get_bit_cnt_exp(pd) + xprec + wprec

            W1 = nd_full_const(1, wshp, graph, params)
            Ye2 = mx.sym.Convolution(Xq, W1, **attr, name=N.n('Convolution'))
            wzint = round(wzpoint*wscale)
            Wz = nd_const(wzint, graph, params)
            Ye2 = mx.sym.broadcast_mul(Wz, Ye2, name=N.n('broadcast_mul'))
            infer_prec2 = get_bit_cnt_exp(pd) + xprec + get_bit_exp(wzint)

            op = mx.sym.elemwise_add(Ye1, Ye2, name=name)
            precs[name][OUT_KEY] = max(infer_prec1, infer_prec2) + 1
            buffers[name] = get_buffer_exp(xscale*wscale)
        elif xquant_type == UAQuantizer.name and \
            wquant_type == USQuantizer.name:
            Xq, xprec, xscale, Xzp = xquant.quantize(
                X, oprec, oname=name, **kwargs)
            Wq, wprec, wscale = wquant.quantize(
                W, oprec, oname=name, **kwargs)
            buffers[name] = get_buffer_exp(xscale*wscale)

            Y1 = mx.sym.Convolution(Xq, Wq, **attr, name=N.n('Convolution'))
            wshp = params[cns[1]].shape
            pd = np.product(wshp[1:])
            infer_prec1 = get_bit_cnt_exp(pd) + xprec + wprec + 1

            xshp = params[cns[0]].shape
            X1 = nd_full(1, xshp, graph, params)
            Y2 = mx.sym.Convolution(X1, Wq, **attr, name=N.n('Convolution'))
            xzp = params[Xzp.attr('name')].asscalar()
            infer_prec2 = get_bit_cnt_exp(abs(xzp)*pd) + wprec

            op = mx.sym.elemwise_add(Y1, Y2, name=N.n('elemwise_add'))
            infer_prec = max(infer_prec1, infer_prec2) + 1
            precs[name][OUT_KEY] = infer_prec
        elif xquant_type == wquant_type == UAQuantizer.name:
            Xq, xprec, xscale, Xzp = xquant.quantize(
                X, oprec, oname=name, **kwargs)
            Wq, wprec, wscale, Wzp = wquant.quantize(
                W, oprec, oname=name, **kwargs)
            buffers[name] = get_buffer_exp(xscale*wscale)

            nodes, infer_precs = [], []

            Y1 = mx.sym.Convolution(Xq, Wq, **attr, name=N.n('Convolution'))
            nodes.append(Y1)
            wshp = params[cns[1]].shape
            pd = np.product(wshp[1:])
            infer_prec1 = get_bit_cnt_exp(pd) + xprec + wprec + 2
            infer_precs.append(infer_prec1)

            W1 = nd_full_const(1, wshp, graph, params)
            Y2 = mx.sym.Convolution(Xq, W1, **attr, name=N.n('Convolution'))
            Y2 = mx.sym.broadcast_mul(Wzp, Y2, name=N.n('broadcast_mul'))
            nodes.append(Y2)
            wzp = params[Wzp.attr('name')].asscalar()
            infer_prec2 = get_bit_cnt_exp(abs(wzp)*pd) + xprec + 1
            infer_precs.append(infer_prec2)

            xshp = params[cns[0]].shape
            X1 = nd_full_const(1, xshp, graph, params)
            Y3 = mx.sym.Convolution(X1, Wq, graph, params)
            Y3 = mx.sym.broadcast_mul(Xzp, Y3, name=N.n('broadcast_mul'))
            nodes.append(Y3)
            xzp = params[Xzp.attr('name')].asscalar()
            infer_prec3 = get_bit_cnt_exp(abs(xzp)*pd) + wprec + 1
            infer_precs.append(infer_prec3)

            val = pd*abs(xzp)*abs(wzp)
            Y4 = nd_const(val, graph, params)
            nodes.append(Y4)
            infer_prec4 = get_bit_cnt_exp(val)
            infer_precs.append(infer_prec4)

            while len(nodes) > 1:
                a, b = nodes.pop(), nodes.pop()
                node = mx.sym.broadcast_add(a, b, name=N.n('broadcast_add'))
                nodes.append(node)
            op = nodes[0]
            infer_prec = max(infer_precs) + 2
            precs[name][OUT_KEY] = infer_prec
        else:
            raise NotImplementedError(
                "Quantization type not implementated," + \
                " op: %20s, Xquant: %20s, Wquant: %20s",
                op_name, [xquant_type, wquant_type])

        logger = logging.getLogger('log.mrt.realize')
        logger.debug(
            "operator  %-20s name=%-40s oscale=%s, iscale=%s",
            op_name, name, buffers[name].serialize(), cns)
        return op


@register_pass("validate")
@register_pass("calculate_ops")
@register_pass("fuse_transpose")
@register_pass("rewrite")
@register_pass("quantize")
@register_pass("prepare_for_compile")
@register_transformer('Pad')
class Pad(tops.Pad, Transformer):
    pass


@register_pass("validate")
@register_pass("calculate_ops")
@register_pass("fuse_transpose")
@register_pass("rewrite")
@register_pass("prepare_for_compile")
@register_pass("compile")
@register_transformer("broadcast_add")
class BroadcastAdd(tops.BroadcastAdd, Transformer):
    def quantize(self, op, **kwargs):
        name, op_name = op.attr('name'), op.attr('op_name')
        attr, childs = op.list_attr(), sym_iter(op.get_children())
        cns = [c.attr('name') for c in childs]
        cfg_dict = kwargs['cfg_dict']
        graph, params = kwargs['graph'], kwargs['params']

        aquant_type = cfg_dict[cns[0]]['quant_type']
        aquant = get_quantizer(aquant_type)
        bquant_type = cfg_dict[cns[1]]['quant_type']
        bquant = get_quantizer(bquant_type)
        A, B = childs
        oprec = kwargs['op_input_precs'][op_name]

        if aquant_type == bquant_type == USQuantizer.name:
            op = _quantize_broadcast(op, **kwargs)
        elif aquant_type == UAQuantizer.name or \
            bquant_type == UAQuantizer.name:
            op = _quantize_scale_zp(op, **kwargs)
        else:
            raise NotImplementedError(
                "Quantization type not implementated," + \
                " op: %20s, Aquant: %20s, Bquant: %20s",
                op_name, [aquant_type, bquant_type])
        return op


@register_pass("validate")
@register_pass("calculate_ops")
@register_pass("fuse_transpose")
@register_pass("rewrite")
@register_pass("prepare_for_compile")
@register_pass("compile")
@register_transformer("broadcast_sub")
class BroadcastSub(tops.BroadcastSub, Transformer):
    def quantize(self, op, **kwargs):
        return _quantize_broadcast(op, **kwargs)


@register_pass("rewrite")
@register_pass("validate")
@register_pass("calculate_ops")
@register_pass("fuse_transpose")
@register_pass("prepare_for_compile")
@register_pass("compile")
@register_transformer("Concat")
class Concat(tops.Concat, Transformer):
    def quantize(self, op, **kwargs):
        return _quantize_scale(op, **kwargs)


@register_pass("calculate_ops")
@register_pass("fuse_transpose")
@register_pass("validate")
@register_pass("rewrite")
@register_pass("quantize")
@register_pass("prepare_for_compile")
@register_pass("compile")
@register_transformer("slice")
class Slice(tops.Slice, Transformer):
    pass


@register_pass("validate")
@register_pass("fuse_transpose")
@register_pass("rewrite")
@register_pass("prepare_for_compile")
@register_pass("calculate_ops")
@register_transformer("BatchNorm")
class BatchNorm(tops.BatchNorm, Transformer):
    pass


# @register_pass("prepare_for_compile") # only for restore
@register_transformer("add_n")
class AddN(Transformer):
    def quantize(self, op, **kwargs):
        return _quantize_scale(op, **kwargs)


@register_pass("rewrite")
@register_pass("validate")
@register_pass("fuse_transpose")
@register_pass("calculate_ops")
@register_pass("prepare_for_compile")
@register_pass("compile")
@register_transformer("clip")
class Clip(tops.Clip, Transformer):
    def quantize(self, op, **kwargs):
        precs, buffers = kwargs['precs'], kwargs['buffers']
        features = kwargs['features']
        X = op.get_children()[0]
        name, X_name = op.attr('name'), X.attr('name')
        attrs = op.list_attr()

        # `a_max`, `a_min` and precision should be align with CVM-Runtime
        iscale = buffers[X.attr('name')].get()
        buffers[name] = SBuffer(iscale)
        a_min = int(get_attr(attrs, "a_min") * iscale)
        a_max = int(get_attr(attrs, "a_max") * iscale)
        precs[name][OUT_KEY] = get_bit_exp(max(abs(a_min), a_max))
        return mx.sym.clip(X, a_min=a_min, a_max=a_max, name=name)


@register_pass("rewrite")
@register_pass("validate")
@register_pass("quantize")
@register_pass("calculate_ops")
@register_pass("fuse_transpose")
@register_pass("prepare_for_compile")
@register_pass('compile')
@register_transformer("transpose")
class Transpose(tops.Transpose, Transformer):
    pass


@register_pass("validate")
@register_pass("calculate_ops")
@register_pass("fuse_transpose")
@register_pass("rewrite")
@register_pass("quantize")
@register_pass("prepare_for_compile")
@register_pass("compile")
@register_transformer("squeeze")
class Squeeze(tops.Squeeze, Transformer):
    pass


@register_pass("validate")
@register_pass("calculate_ops")
@register_pass("fuse_transpose")
@register_pass("rewrite")
@register_pass("quantize")
@register_pass("compile")
@register_pass("prepare_for_compile")
@register_transformer("Reshape")
class Reshape(tops.Reshape, Transformer):
    pass


@register_pass("validate")
@register_pass("rewrite")
@register_pass("calculate_ops")
@register_pass("fuse_transpose")
# @register_pass("prepare_for_compile") # only for restore
@register_transformer("softmax")
class Softmax(tops.Softmax, Transformer):
    def quantize(self, op, **kwargs):
        params, graph = kwargs['params'], kwargs['graph']
        buffers, precs = kwargs['buffers'], kwargs['precs']
        features, cfg_dict = kwargs['features'], kwargs['cfg_dict']
        name, op_name = op.attr('name'), op.attr('op_name')
        childs, attr = sym_iter(op.get_children()), op.list_attr()
        cns = [c.attr('name') for c in childs] if childs else []

        oprec = kwargs['op_input_precs'][op_name]
        th = features[cns[0]].get()
        xs = scale_exp(th, oprec)
        quant_type = cfg_dict[cns[0]]['quant_type']
        assert quant_type == USQuantizer.name
        quant = get_quantizer(quant_type)
        X, xprec, xs = quant.quantize(
            childs[0], oprec, oscale=xs, oname=name, **kwargs)
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

        tprec = get_bit_exp(math.exp(lambd))
        table = nd.clip(table, a_min=0, a_max=get_range_exp(tprec))
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
        oscale = get_range_exp(oprec)
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
        buffers[name] = SBuffer(oscale)

        logger = logging.getLogger('log.mrt.realize')
        logger.debug("operator  %-20s name=%-40s oscale=%s, iscale=%s",
                     op_name, name, buffers[name].serialize(), cns)
        return op


@register_pass("calculate_ops")
@register_pass("validate")
@register_pass("rewrite")
@register_pass("fuse_transpose")
@register_transformer("slice_axis")
class SliceAxis(tops.SliceAxis, Transformer):
    pass


@register_pass("validate")
@register_pass("calculate_ops")
@register_pass("rewrite")
@register_pass("fuse_transpose")
@register_pass("prepare_for_compile") # only for restore
@register_transformer("_div_scalar")
class DivScalar(tops.DivScalar, Transformer):
    pass


@register_pass("validate")
@register_pass("calculate_ops")
@register_pass("fuse_transpose")
@register_pass("rewrite")
@register_pass("prepare_for_compile")
@register_pass("compile")
@register_transformer("broadcast_mul")
class BroadcastMul(tops.BroadcastMul, Transformer):
    def quantize(self, op, **kwargs):
        precs, buffers = kwargs['precs'], kwargs['buffers']
        name, op_name = op.attr('name'), op.attr('op_name')
        cfg_dict = kwargs['cfg_dict']
        childs, attr = sym_iter(op.get_children()), op.list_attr()
        cns = [c.attr('name') for c in childs] if childs else []

        oprec = kwargs['op_input_precs'][op_name]
        xquant_type, bquant_type = \
            cfg_dict[cns[0]]['quant_type'], cfg_dict[cns[1]]['quant_type']
        xquant, bquant = \
            get_quantizer(xquant_type), get_quantizer(bquant_type)
        if xquant_type == bquant_type == USQuantizer.name:
            X, xprec, xs = xquant.quantize(
                childs[0], oprec, oname=name, **kwargs)
            B, bprec, bs = bquant.quantize(
                childs[1], oprec, oname=name, **kwargs)

            op = get_mxnet_op(op_name)(X, B, **attr, name=name)

            if bprec == 1 and bs == 1:
                # special case: childs[1] is 0
                buffers[name] = SBuffer(1)
                precs[name][OUT_KEY] = 1
            else:
                buffers[name] = SBuffer(xs * bs)
                infer_prec = xprec + bprec
                precs[name][OUT_KEY] = infer_prec
        else:
            raise NotImplementedError(
                "Quantization type not implementated," + \
                " op: %20s, Xquant: %20s, Wquant: %20s",
                op_name, [xquant_type, bquant_type])

        logger = logging.getLogger('log.mrt.realize')
        logger.debug(
            "operator  %-20s name=%-40s oscale=%s, iscale=%s",
            op_name, name, buffers[name].serialize(), cns)
        return op


@register_pass("validate")
@register_pass("calculate_ops")
@register_pass("prepare_for_compile")
@register_pass("compile")
@register_transformer("Custom")
class Custom(tops.Custom, Transformer):
    pass



@register_pass("validate")
@register_pass("fuse_transpose")
@register_pass("rewrite")
@register_pass("quantize")
@register_pass("calculate_ops")
@register_pass("prepare_for_compile")
@register_pass("compile")
@register_transformer("max")
class Max(tops.Max, Transformer):
    pass


@register_pass("validate")
@register_pass("rewrite")
@register_pass("fuse_transpose")
@register_pass("calculate_ops")
@register_pass("quantize")
@register_pass("prepare_for_compile")
@register_pass("compile")
@register_transformer("relu")
class Relu(Transformer):
    pass


@register_pass("validate")
@register_pass("fuse_transpose")
@register_pass("prepare_for_compile")
@register_pass("calculate_ops")
@register_pass('compile')
@register_pass("rewrite")
@register_transformer("sum")
class Sum(tops.Sum, Transformer):
    def quantize(self, op, **kwargs):
        infer_shapes = kwargs['infer_shapes']
        buffers = kwargs['buffers']
        cfg_dict = kwargs['cfg_dict']
        name, op_name = op.attr('name'), op.attr('op_name')
        childs, attr = sym_iter(op.get_children()), op.list_attr()
        cns = [c.attr('name') for c in childs] if childs else []
        oshp = infer_shapes[name][get_entry_id(op)]

        quant_type = cfg_dict[cns[0]]['quant_type']
        assert quant_type == USQuantizer.name, (quant_type, name, op_name)
        quant = get_quantizer(quant_type)
        oprec = kwargs['op_input_precs'][op_name]
        X, xprec, xs = quant.quantize(
            childs[0], oprec, oname=name, **kwargs)
        buffers[name] = SBuffer(xs)
        op = get_mxnet_op(op_name)(X, **attr, name=name)

        ishp = infer_shapes[cns[0]][get_entry_id(childs[0])]
        k = int(nd.prod(nd_array(ishp)).asscalar() / \
            nd.prod(nd_array(oshp)).asscalar())
        kprec = get_bit_cnt_exp(k)
        infer_prec = kprec + xprec
        kwargs['precs'][name][OUT_KEY] = infer_prec

        logger = logging.getLogger('log.mrt.realize')
        logger.debug(
            "operator  %-20s name=%-40s oscale=%s, iscale=%s",
            op_name, name, buffers[name].serialize(), cns)
        return op


@register_pass("calculate_ops")
@register_pass("prepare_for_compile")
@register_pass("compile")
@register_transformer("Cast")
class Cast(tops.Cast, Transformer):
    pass


@register_pass("compile")
@register_pass("fuse_transpose")
@register_pass("validate")
@register_pass("calculate_ops")
@register_pass("rewrite")
@register_pass("prepare_for_compile")
@register_transformer("elemwise_add")
class ElemwiseAdd(tops.ElemwiseAdd, Transformer):
    def quantize(self, op, **kwargs):
        return _quantize_scale(op, **kwargs)


@register_pass("calculate_ops")
@register_pass("fuse_transpose")
@register_pass("rewrite")
@register_pass("quantize")
@register_pass("prepare_for_compile")
@register_pass("compile")
@register_transformer("broadcast_div")
class BroadcastDiv(Transformer):
    pass


@register_pass("validate")
@register_pass("calculate_ops")
@register_pass("fuse_transpose")
@register_pass("rewrite")
@register_transformer("SwapAxis")
class SwapAxis(Transformer, tops.SwapAxis):
    pass


@register_pass("validate")
@register_pass("calculate_ops")
@register_pass("fuse_transpose")
@register_pass("rewrite")
@register_pass("compile")
@register_pass("prepare_for_compile")
@register_transformer('Embedding')
class Embedding(Transformer):
    def quantize(self, op, **kwargs):
        features, buffers = kwargs['features'], kwargs['buffers']
        cfg_dict = kwargs['cfg_dict']
        name, op_name = op.attr('name'), op.attr('op_name')
        attr, childs = op.list_attr(), sym_iter(op.get_children())
        cns = [c.attr('name') for c in childs]

        xquant_type = cfg_dict[cns[0]]['quant_type']
        wquant_type = cfg_dict[cns[1]]['quant_type']
        xquant, wquant = \
            get_quantizer(xquant_type), get_quantizer(wquant_type)
        oprec = kwargs['op_input_precs'][op_name]

        if xquant_type == wquant_type == USQuantizer.name:
            X, xs = childs[0], buffers[cns[0]]
            if xs != 1:
                X, _, _ = xquant.quantize(X, 32, oscale=1, oname=name, **kwargs)
            W, _, ws = wquant.quantize(childs[1], oprec, oname=name, **kwargs)
            features[name] = features[cns[1]]
            buffers[name] = SBuffer(ws)
            kwargs['precs'][name][OUT_KEY] = get_bit_exp(features[name].get()*ws)
            op = get_mxnet_op(op_name)(X, W, **attr, name=name)
        else:
            raise NotImplementedError(
                "Quantization type not implementated," + \
                " op: %20s, Xquant: %20s, Wquant: %20s",
                op_name, [xquant_type, wquant_type])

        return op


@register_pass("validate")
@register_pass("calculate_ops")
@register_pass("fuse_transpose")
@register_pass("rewrite")
@register_pass("quantize")
@register_pass("prepare_for_compile")
@register_pass("compile")
@register_transformer('expand_dims')
class ExpandDims(Transformer, tops.ExpandDims):
    pass


@register_pass("calculate_ops")
@register_pass("validate")
@register_pass("fuse_transpose")
@register_pass("rewrite")
@register_transformer("LeakyReLU")
class LeakyReLU(Transformer, tops.LeakyReLU):
    pass


@register_pass("validate")
@register_pass("calculate_ops")
@register_pass("fuse_transpose")
@register_pass("rewrite")
@register_transformer("_mul_scalar")
class MulScalar(Transformer, tops.MulScalar):
    pass


@register_pass("validate")
@register_pass("rewrite")
@register_pass("calculate_ops")
@register_pass("fuse_transpose")
@register_pass("quantize")
@register_pass("prepare_for_compile")
@register_pass("compile")
@register_transformer("repeat")
class Repeat(Transformer, tops.Repeat):
    pass


@register_pass("validate")
@register_pass("rewrite")
@register_pass("calculate_ops")
@register_pass("fuse_transpose")
@register_pass("prepare_for_compile")
@register_pass("compile")
@register_transformer("slice_like")
class SliceLike(Transformer, tops.SliceLike):
    def quantize(self, op, **kwargs):
        return _quantize_scale(op, **kwargs)


@register_pass("calculate_ops")
@register_pass("validate")
@register_pass("rewrite")
@register_pass("fuse_transpose")
# @register_pass("prepare_for_compile") # only for restore
@register_transformer("sigmoid")
class Sigmoid(Transformer, tops.Sigmoid):
    def quantize(self, op, **kwargs):
        return _quantize_table(op, **kwargs)


@register_pass("calculate_ops")
@register_pass("validate")
@register_pass("rewrite")
@register_pass("fuse_transpose")
# @register_pass("prepare_for_compile") # only for restore
@register_transformer("exp")
class Exp(Transformer, tops.Exp):
    def quantize(self, op, **kwargs):
        return _quantize_table(op, **kwargs)


@register_pass("compile")
@register_pass("validate")
@register_pass("fuse_transpose")
@register_pass("calculate_ops")
@register_pass("rewrite")
@register_pass("prepare_for_compile")
@register_transformer("elemwise_sub")
class ElemwiseSub(Transformer, tops.ElemwiseSub):
    def quantize(self, op, **kwargs):
        return _quantize_scale(op, **kwargs)


@register_pass("validate")
@register_pass("calculate_ops")
@register_pass("fuse_transpose")
@register_pass("rewrite")
@register_pass("quantize")
@register_pass("prepare_for_compile")
@register_pass("compile")
@register_transformer("tile")
class Tile(Transformer, tops.Tile):
    pass


@register_pass("calculate_ops")
@register_pass("validate")
@register_pass("fuse_transpose")
@register_pass("rewrite")
@register_pass("prepare_for_compile")
@register_pass("compile")
@register_transformer('_contrib_box_nms')
class BoxNms(Transformer, tops.BoxNms):
    pass


@register_pass("validate")
@register_pass("calculate_ops")
@register_pass("fuse_transpose")
@register_pass("rewrite")
@register_pass("quantize")
@register_pass("prepare_for_compile")
@register_pass("compile")
@register_transformer("negative")
class Negative(Transformer, tops.Negative):
    pass


@register_pass("validate")
@register_pass("calculate_ops")
@register_pass("rewrite")
@register_pass("fuse_transpose")
@register_transformer("_plus_scalar")
class PlusScalar(Transformer, tops.PlusScalar):
    pass


@register_pass("validate")
@register_pass("calculate_ops")
@register_pass("rewrite")
@register_pass("fuse_transpose")
@register_transformer("zeros_like")
class ZerosLike(Transformer, tops.ZerosLike):
    pass


@register_pass("calculate_ops")
@register_pass("fuse_transpose")
@register_pass("rewrite")
@register_pass("validate")
@register_pass("prepare_for_compile")
@register_pass("compile")
@register_transformer("_greater_scalar")
class GreaterScalar(Transformer, tops.GreaterScalar):
    pass


@register_pass("calculate_ops")
@register_pass("validate")
@register_pass("rewrite")
@register_pass("fuse_transpose")
@register_pass("prepare_for_compile")
@register_pass("compile")
@register_transformer("where")
class Where(Transformer, tops.Where):
    pass


@register_pass("validate")
@register_pass("calculate_ops")
@register_pass("fuse_transpose")
@register_pass("rewrite")
@register_transformer("ones_like")
class OnesLike(Transformer, tops.OnesLike):
    pass

def _quantize_table(op, **kwargs):
    params, graph = kwargs['params'], kwargs['graph']
    features, precs, buffers = \
        kwargs['features'], kwargs['precs'], kwargs['buffers']
    cfg_dict = kwargs['cfg_dict']
    name, op_name = op.attr('name'), op.attr('op_name')
    childs = sym_iter(op.get_children())
    cns = [c.attr('name') for c in childs] if childs else []

    xquant_type = cfg_dict[cns[0]]['quant_type']
    xquant = get_quantizer(xquant_type)

    iprec = kwargs['op_input_precs'][op_name]
    xs = scale_exp(features[cns[0]].get(), iprec)
    X, xprec, xs = xquant.quantize(
        childs[0], iprec, oscale=xs, oname=name, **kwargs)
    alpha = get_range_exp(xprec)
    var = nd_const(alpha, graph, params)
    X = mx.sym.broadcast_add(X, var, name=N.n(op_name+'_offset'))

    out = sutils.get_nd_op(op_name)(sutils.nd_arange(-alpha, alpha+1) / xs)
    oprec = precs[name].get(OUT_KEY, 16)
    oscale = scale_exp(out.abs().max().asscalar(), oprec)
    buffers[name] = SBuffer(oscale)

    W_name = N.n("cvm_lut_weight")
    params[W_name] = weight = (out * oscale).round().reshape(2*alpha+1, 1)
    wattr = {'precision': str(oprec)}
    W = graph[W_name] = mx.sym.var(W_name, shape=weight.shape, attr=wattr)
    op = mx.sym.Custom(X, W, in_dim=2*alpha+1, name=name, op_type='cvm_lut')
    precs[name][OUT_KEY] = oprec

    logger = logging.getLogger('log.mrt.realize')
    logger.debug("operator  %-20s name=%-40s oscale=%s, iscale=%s",
                 op_name, name, buffers[name].serialize(), cns)
    return op

def _quantize_scale_zp(op, **kwargs):
    features, precs = kwargs['features'], kwargs['precs']
    buffers, cfg_dict = kwargs['buffers'], kwargs['cfg_dict']
    graph, params = kwargs['graph'], kwargs['params']
    name, op_name = op.attr('name'), op.attr('op_name')
    attr, childs = op.list_attr(), sym_iter(op.get_children())
    cns = [c.attr('name') for c in childs] if childs else []

    oprec = kwargs['op_input_precs'][op_name]
    oscales = []
    for c in childs:
        cquant_type = cfg_dict[c.attr('name')]['quant_type']
        cquant = get_quantizer(cquant_type)
        ft = features[c.attr('name')]
        oscale = cquant.get_scale(oprec, ft)
        oscales.append(oscale)
    oscale = min(oscales)
    buffers[name] = SBuffer(oscale)
    nodes, cprecs = [], []

    for c in childs:
        cquant_type = cfg_dict[c.attr('name')]['quant_type']
        cquant = get_quantizer(cquant_type)
        if cquant.name == USQuantizer.name:
            c, cprec, _ = cquant.quantize(
                c, oprec, oscale=oscale, oname=name, **kwargs)
        elif cquant.name == UAQuantizer.name:
            c, cprec, cscale, czpoint = cquant.quantize(
                c, oprec, oscale=oscale, oname=name, **kwargs)
            czint = round(czpoint*cscale)
            Cz = nd_const(czint, graph, params)
            nodes.append(Cz)
            cprecs.append(get_bit_exp(czint))
        cprecs.append(cprec)
        nodes.append(c)

    if op_name in [Concat.op_name]:
        op = get_mxnet_op(op_name)(*nodes, **attr, name=name)
        infer_prec = max(cprecs)
    elif op_name in [BroadcastAdd.op_name]:
        while len(nodes) > 1:
            tname = N.n('broadcast_add') if len(nodes) > 2 else name
            a, b = nodes.pop(0), nodes.pop(0)
            tmp = mx.sym.broadcast_add(a, b, name=tname)
            nodes.append(tmp)
        kprec = get_bit_cnt_exp(len(nodes))
        infer_prec = max(cprecs) + kprec
        op = nodes[0]
    elif op_name in [AddN.op_name]:
        while len(nodes) > 1:
            tname = N.n('elemwise_add') if len(nodes) > 2 else name
            a, b = nodes.pop(0), nodes.pop(0)
            tmp = mx.sym.elemwise_add(a, b, name=tname)
            nodes.append(tmp)
        kprec = get_bit_cnt_exp(len(nodes))
        infer_prec = max(cprecs) + kprec
        op = nodes[0]
    else:
        raise NotADirectoryError(
            "symbol merge function of op_name: %s has not been " + \
            "implemented, name: %s", op_name, name)
    precs[name][OUT_KEY] = infer_prec

    logger = logging.getLogger('log.mrt.realize')
    logger.debug("operator  %-20s name=%-40s oscale=%s, iscale=%s",
                 op_name, name, buffers[name].serialize(), cns)
    return op

def _quantize_scale(op, **kwargs):
    features, precs = kwargs['features'], kwargs['precs']
    buffers, cfg_dict = kwargs['buffers'], kwargs['cfg_dict']
    name, op_name = op.attr('name'), op.attr('op_name')
    attr, childs = op.list_attr(), sym_iter(op.get_children())
    cns = [c.attr('name') for c in childs] if childs else []

    assert all([features[cn].name == FT_TYPE_EXP for cn in cns])
    absmax = max([features[cn].get() for cn in cns])
    oprec = kwargs['op_input_precs'][op_name]
    oscale = scale_exp(absmax, oprec)
    buffers[name] = SBuffer(oscale)
    nodes, cprecs = [], []

    assert all([cfg_dict[cn]['quant_type'] == \
        USQuantizer.name for cn in cns])
    quant = get_quantizer(USQuantizer.name)

    for c in childs:
        c, cprec, _ = quant.quantize(
            c, oprec, oscale=oscale, oname=name, **kwargs)
        cprecs.append(cprec)
        nodes.append(c)

    if op_name in [Concat.op_name, BroadcastAdd.op_name,
        ElemwiseAdd.op_name, ElemwiseSub.op_name, SliceLike.op_name]:
        op = get_mxnet_op(op_name)(*nodes, **attr, name=name)
        infer_prec = max(cprecs) if op_name == Concat.op_name \
            else max(cprecs)+1
    elif op_name == AddN.op_name:
        while len(nodes) > 1:
            tname = N.n('elemwise_add') if len(nodes) > 2 else name
            a, b = nodes.pop(0), nodes.pop(0)
            tmp = mx.sym.elemwise_add(a, b, name=tname)
            nodes.append(tmp)
        kprec = get_bit_cnt_exp(len(nodes))
        infer_prec = max(cprecs) + kprec
        op = nodes[0]
    else:
        raise NotImplementedError(
            "symbol merge function of op_name: %s has not been " + \
            "implemented, name: %s", op_name, name)
    precs[name][OUT_KEY] = infer_prec

    logger = logging.getLogger('log.mrt.realize')
    logger.debug("operator  %-20s name=%-40s oscale=%s, iscale=%s",
                 op_name, name, buffers[name].serialize(), cns)
    return op

def _quantize_xw(op, **kwargs):
    features, buffers = kwargs['features'], kwargs['buffers']
    precs = kwargs['precs']
    cfg_dict, params = kwargs['cfg_dict'], kwargs['params']
    name, op_name = op.attr('name'), op.attr('op_name')
    childs, attr = sym_iter(op.get_children()), op.list_attr()
    cns = [c.attr('name') for c in childs] if childs else []

    X, W = childs
    Xquant = get_quantizer(cfg_dict[cns[0]]['quant_type'])
    Wquant = get_quantizer(cfg_dict[cns[1]]['quant_type'])

    oprec = kwargs['op_input_precs'][op_name]
    Xq, xprec, xscale = Xquant.quantize(X, oprec, oname=name, **kwargs)
    Wq, wprec, wscale = Wquant.quantize(W, oprec, oname=name, **kwargs)
    buffers[name] = get_buffer_exp(xscale*wscale)
    op = get_mxnet_op(op_name)(Xq, Wq, **attr, name=name)

    shp = params[cns[1]].shape
    k = int(nd.prod(nd_array(shp[1:])).asscalar())
    kprec = get_bit_cnt_exp(k)
    infer_prec = kprec + xprec + wprec
    precs[name][OUT_KEY] = infer_prec
    return op

def sym_slice(X, ichannel, step, **kwargs):
    name = X.attr('name')
    shp = kwargs['infer_shapes'][name][get_entry_id(X)]
    ndims = len(shp)
    nodes = []
    rchannel = ndims-ichannel-1
    for i in range(0, shp[ichannel], step):
        suffix = '_' + str(i)+'-'+str(i+step)
        Xi = mx.sym.slice(
            X, begin=(None,)*ichannel+(i,)+(None,)*rchannel,
            end=(None,)*ichannel+(i+step,)+(None,)*rchannel,
            name=N.n(name+suffix))
        nodes.append(Xi)
    return nodes

def kernel_slice_2d(W, **kwargs):
    name = W.attr('name')
    shp = kwargs['infer_shapes'][name][get_entry_id(W)]
    OC, IC = shp[:2]
    nodes = []
    for o in range(OC):
        Wo = mx.sym.slice(W, begin=(o,None,None,None), end=(o+1,None,None,None))
        nnodes = []
        for i in range(IC):
            suffix = '_' + str(o)+'-'+str(i)
            Woi = mx.sym.slice(
                Wo, begin=(None,i,None,None), end=(None,i+1,None,None),
                name=N.n(name+suffix))
            nnodes.append(Woi)
        nodes.append(nnodes[:])
    return nodes

def sym_merge(op, nodes, **kwargs):
    name, op_name = op.attr('name'), op.attr('op_name')
    attr = op.list_attr()
    return op

def _quantize_broadcast(op, **kwargs):
    params = kwargs['params']
    features = kwargs['features']
    precs = kwargs['precs']
    buffers = kwargs['buffers']

    name = op.attr('name')
    childs = sym_iter(op.get_children())
    cns = [c.attr('name') for c in childs]
    assert all([features[cn].name == FT_TYPE_EXP for cn in cns])
    cfts = [features[cn].get() for cn in cns]

    if cfts[0] == 0 or cfts[1] == 0:
        if cfts[0] == 0 and cfts[1] == 0:
            features[name] = AFeature(0)
            precs[name] = {OUT_KEY: 1}
            buffers[name] = SBuffer(1)
            return op
        cn = cns[1] if cfts[0] == 0 else cns[0]
        bit = get_bit_exp(params[cn]) if cn in params \
            else precs[cn][OUT_KEY]
        buffers[name] = SBuffer(1) if cn in params else buffers[cn]
        precs[name] = {OUT_KEY: bit}
        features[name] = features[cn]
        return op

    return _quantize_scale(op, **kwargs)
