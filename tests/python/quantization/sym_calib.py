import logging
import os
import numpy as np

import mxnet as mx
from mxnet.gluon import nn, SymbolBlock
from mxnet import ndarray as nd
import nnvm as nnvm
import tvm

from sym_utils import *
from sym_pass import *
from quant_utils import *
from utils import *
import sim_quant_helper as sim
import cvm_op as cvm

from scipy import stats

default_target_bit = 8 # INT8
bias_target_bit = default_target_bit * 4 - 1
disable_requant_ops = [
    'Activation', 'Pooling', 'Flatten',
    'slice', 'clip',
]

def _collect_symbol_ext(sym, params, graph, inputs_ext, scale_shapes):
    name = sym.attr('name')
    op_name = sym.attr('op_name')
    attr = sym.list_attr()
    childs = sym_iter(sym.get_children())

    scale_shapes[name] = (1,)
    # if op_name == 'Convolution' and attr['num_group'] == attr['num_filter']:
    #     X, W = childs[0], childs[1]
    #     channel = int(attr['num_filter'])
    #     scale_shapes[W.attr('name')] = (channel, 1, 1, 1)
    #     scale_shapes[X.attr('name')] = (1, channel, 1, 1)
    #     if attr['no_bias'] == 'False':
    #         B_name = childs[2].attr('name')
    #         scale_shapes[B_name] = (channel,)
    return sym, params

def _smooth_distribution(p, eps=0.0001):
    """Given a discrete distribution (may have not been normalized to 1),
    smooth it by replacing zeros with eps multiplied by a scaling factor and taking the
    corresponding amount off the non-zero values.
    Ref: http://web.engr.illinois.edu/~hanj/cs412/bk3/KL-divergence.pdf
    """
    is_zeros = (p == 0).astype(np.float32)
    is_nonzeros = (p != 0).astype(np.float32)
    n_zeros = is_zeros.sum()
    n_nonzeros = p.size - n_zeros
    if not n_nonzeros:
        raise ValueError('The discrete probability distribution is malformed. All entries are 0.')
    eps1 = eps * float(n_zeros) / float(n_nonzeros)
    assert eps1 < 1.0, 'n_zeros=%d, n_nonzeros=%d, eps1=%f' % (n_zeros, n_nonzeros, eps1)
    hist = p.astype(np.float32)
    hist += eps * is_zeros + (-eps1) * is_nonzeros
    assert (hist <= 0).sum() == 0
    return hist
def _get_optimal_threshold(arr, num_bins=8001, num_quantized_bins=255):
    """Given a dataset, find the optimal threshold for quantizing it.
    Ref: http://on-demand.gputechconf.com/gtc/2017/presentation/s7310-8-bit-inference-with-tensorrt.pdf
    """
    if isinstance(arr, nd.NDArray):
        arr = arr.asnumpy()
    elif isinstance(arr, list):
        assert len(arr) != 0
        for i, nd_arr in enumerate(arr):
            if isinstance(nd_arr, nd.NDArray):
                arr[i] = nd_arr.asnumpy()
            elif not isinstance(nd_arr, np.ndarray):
                raise TypeError('get_optimal_threshold only supports input type of NDArray,'
                                ' list of np.ndarrays or NDArrays, and np.ndarray,'
                                ' while received type=%s' % (str(type(nd_arr))))
        arr = np.concatenate(arr)
    elif not isinstance(arr, np.ndarray):
        raise TypeError('get_optimal_threshold only supports input type of NDArray,'
                        ' list of NDArrays and np.ndarray,'
                        ' while received type=%s' % (str(type(arr))))
    min_val = np.min(arr)
    max_val = np.max(arr)
    th = max(abs(min_val), abs(max_val))

    hist, hist_edges = np.histogram(arr, bins=num_bins, range=(-th, th))
    zero_bin_idx = num_bins // 2
    num_half_quantized_bins = num_quantized_bins // 2
    assert np.allclose(hist_edges[zero_bin_idx] + hist_edges[zero_bin_idx + 1],
                       0, rtol=1e-5, atol=1e-7), "val: %s, %s" \
                    % (hist_edges[zero_bin_idx], hist_edges[zero_bin_idx + 1])

    thresholds = np.zeros(num_bins // 2 + 1 - num_quantized_bins // 2)
    divergence = np.zeros_like(thresholds)

    quantized_bins = np.zeros(num_quantized_bins, dtype=np.int32)
    # i means the number of bins on half axis excluding the zero bin.
    for i in range(num_quantized_bins // 2,
                   num_bins // 2 + 1):
        p_bin_idx_start = zero_bin_idx - i
        p_bin_idx_stop = zero_bin_idx + i + 1
        thresholds[i - num_half_quantized_bins] = hist_edges[p_bin_idx_stop]
        sliced_nd_hist = hist[p_bin_idx_start:p_bin_idx_stop]

        # generate reference distribution p
        p = sliced_nd_hist.copy()
        assert p.size % 2 == 1
        assert p.size >= num_quantized_bins
        # put left outlier count in p[0]
        left_outlier_count = np.sum(hist[0:p_bin_idx_start])
        p[0] += left_outlier_count
        # put right outlier count in p[-1]
        right_outlier_count = np.sum(hist[p_bin_idx_stop:])
        p[-1] += right_outlier_count
        # is_nonzeros[k] indicates whether hist[k] is nonzero
        is_nonzeros = (sliced_nd_hist != 0).astype(np.int32)

        # calculate how many bins should be merged to generate quantized distribution q
        num_merged_bins = p.size // num_quantized_bins
        # merge hist into num_quantized_bins bins
        for j in range(num_quantized_bins):
            start = j * num_merged_bins
            stop = start + num_merged_bins
            quantized_bins[j] = sliced_nd_hist[start:stop].sum()
        quantized_bins[-1] += sliced_nd_hist[num_quantized_bins * num_merged_bins:].sum()
        # expand quantized_bins into p.size bins
        q = np.zeros(p.size, dtype=np.float32)
        for j in range(num_quantized_bins):
            start = j * num_merged_bins
            if j == num_quantized_bins - 1:
                stop = -1
            else:
                stop = start + num_merged_bins
            norm = is_nonzeros[start:stop].sum()
            if norm != 0:
                q[start:stop] = float(quantized_bins[j]) / float(norm)
        q[sliced_nd_hist == 0] = 0
        p = _smooth_distribution(p)
        # There is a chance that q is an invalid probability distribution.
        try:
            q = _smooth_distribution(q)
        except ValueError:
            divergence[i - num_half_quantized_bins] = float("inf")
        divergence[i - num_half_quantized_bins] = stats.entropy(p, q)
        quantized_bins[:] = 0

    min_divergence_idx = np.argmin(divergence)
    min_divergence = divergence[min_divergence_idx]
    opt_th = thresholds[min_divergence_idx]
    return min_val, max_val, min_divergence, opt_th
def _get_thresholds(output, calib_mode='naive'):
    if calib_mode == 'naive':
        min_range = output.min().asscalar()
        max_range = output.max().asscalar()
    elif calib_mode == 'entropy':
        min_val, max_val, min_divergence, opt_th = \
            _get_optimal_threshold(output, num_bins=8001, num_quantized_bins=255)
        min_range = -opt_th if min_val < 0 else 0
        max_range = opt_th
    return (min_range, max_range)
def _calib_sym_collect_thresholds(sym, params, graph, inputs_ext,
        scale_shapes, th_dict, calib_data, last_out=[None],
        calib_mode='naive', ctx=mx.gpu()):
    logger = logging.getLogger('log.calib.sym.collect.thresholds')

    name = sym.attr('name')
    op_name = sym.attr('op_name')
    childs = sym_iter(sym.get_children())
    attr = sym.list_attr()

    if op_name == 'null':
        if name in inputs_ext:
            last_out[0] = output = calib_data
        else:
            last_out[0] = output = params[name]
    elif op_name in disable_requant_ops:
        assert len(childs) == 1
        output = last_out[0]
    else:
        args = sym.list_inputs()
        inputs = [mx.sym.var(n) for n in inputs_ext if n in args]
        graph = SymbolBlock(sym, inputs)
        load_parameters(graph, params, ctx=ctx)
        last_out[0] = output = graph.forward(calib_data.as_in_context(ctx))

    slices = [output]
    shape = scale_shapes[name]
    for idx, s in enumerate(shape):
        if s == 1 :
            continue
        begin, end = [None]*len(shape), [None]*len(shape)
        tmp_slices = []
        for sli in slices:
            for start in range(s):
                begin[idx], end[idx] = start, start+1
                tmp = sli.slice(begin=begin, end=end)
                tmp_slices.append(tmp)
        slices = tmp_slices

    th_dict[name] = nd.zeros((len(slices), 2))
    for idx, out in enumerate(slices):
        th_dict[name][idx] = _get_thresholds(out, calib_mode)
    logger.debug("collect symbol %-30s output shape %-20s vs. %-20s th_dict: (%s, %s)",
            name, output.shape, shape,
            th_dict[name].min().asscalar(), th_dict[name].max().asscalar())
    return sym, params

def _calib_sym_zero_symmetric(sym, params, graph, inputs_ext,
        th_dict, in_zeros, out_zeros):
    logger = logging.getLogger('log.calib.sym.requantize.params')

    name = sym.attr('name')
    op_name = sym.attr('op_name')
    childs = sym_iter(sym.get_children())
    attr = sym.list_attr()
    cpu = mx.cpu()

    # calculate input zero symmetric
    if childs is not None:
        childs_name = [c.attr('name') for c in childs]
        in_zeros[name] = [out_zeros[n] for n in childs_name]

    # calculate output zero symmetric
    if op_name == 'null':
        out_zeros[name] = 0
        if name in inputs_ext:
            out_zeros[name] = inputs_ext[name]['zero_point']
            # out_zeros[name] = sim.get_zero_symmetric(th_dict[name])
    elif op_name in ['Pooling', 'Flatten', 'slice']:
        assert len(in_zeros[name]) == 1
        out_zeros[name] = in_zeros[name][0]
    else:
        out_zeros[name] = sim.get_zero_symmetric(th_dict[name])

    return sym, params
def _calib_sym_zero_rewrite(sym, params, graph, inputs_ext,
        in_zeros, out_zeros, infer_shapes, idxs):
    logger = logging.getLogger('log.calib.sym.rewrite')

    name = sym.attr('name')
    op_name = sym.attr('op_name')
    attr = sym.list_attr()
    childs = sym_iter(sym.get_children())

    if op_name == 'null':
        return sym, params

    assert childs is not None
    node = sym
    new_childs = []
    out_z, in_z = out_zeros[name], in_zeros[name]

    index = idxs['index']
    if op_name in ['FullyConnected', 'Convolution']:
        X, W = childs[0], childs[1]
        X_shape = infer_shapes[X.attr('name')]
        W_shape = infer_shapes[W.attr('name')]
        Y_shape = infer_shapes[name]
        # logger.debug("%s out_shape: %s, in_shape: %s, weight_shape: %s",
                # index, infer_shapes[name], X_shape, W_shape)

        bias_name = name + '_offset_bias'
        assert bias_name not in graph

        Y_z, X_z = out_zeros[name], in_zeros[name][0]

        data_shape = [1 if i==0 else s for i,s in enumerate(X_shape)]
        data = nd.full(data_shape, -X_z)
        weight = params[W.attr('name')]
        bias = nd.full((W_shape[0]), Y_z)
        if attr['no_bias'] == 'False':
            bias += params[childs[2].attr('name')]
        params[bias_name] = get_nd_op(op_name)(data, weight, bias, **attr)

        attr['no_bias'] = 'True'
        B = graph[bias_name] = mx.sym.var(bias_name,
                shape=params[bias_name].shape)
        node = get_mxnet_op(op_name)(X, W, **attr, name=name)
        node = mx.sym.broadcast_add(node, B)
    elif op_name in ['broadcast_mul']:
        X, W = childs[0], childs[1]
        assert W.attr('op_name') == 'null'

        X_shape = infer_shapes[X.attr('name')]
        W_shape = infer_shapes[W.attr('name')]
        Y_shape = infer_shapes[name]

        bias_name = name + '_offset_bias'
        assert bias_name not in graph

        Y_z, X_z = out_zeros[name], in_zeros[name][0]

        data_shape = [1 if i==0 else s for i,s in enumerate(X_shape)]
        data = nd.full(data_shape, -X_z)
        weight = params[W.attr('name')]
        params[bias_name] = get_nd_op(op_name)(data, weight, **attr)
        params[bias_name] += Y_z

        if np.any(params[bias_name].asnumpy() != 0):
            B = graph[bias_name] = mx.sym.var(bias_name,
                    shape=params[bias_name].shape)
            node = get_mxnet_op(op_name)(X, W, **attr, name=name)
            node = mx.sym.broadcast_add(node, B)
    elif op_name in ['elemwise_add', 'broadcast_add']:
        X, A = childs[0], childs[1]
        Y_shape = infer_shapes[name]
        B_shape = [1 for _ in Y_shape]

        bias_name = name + '_offset_bias'
        assert bias_name not in graph

        Y_z, X_z = out_zeros[name], in_zeros[name]
        params[bias_name] = nd.array([Y_z - X_z[0] - X_z[1]]).reshape(B_shape)

        B = graph[bias_name] = mx.sym.var(bias_name, shape=B_shape)
        node = get_mxnet_op(op_name)(X, A, **attr, name=name)
        node = mx.sym.broadcast_add(node, B)
    elif op_name in ['Pooling', 'Flatten', 'slice']:
        Y_z, X_z = out_zeros[name], in_zeros[name]
        for x_z in X_z:
            assert x_z == Y_z
    elif op_name == 'sum':
        X = childs[0]
        X_shape = infer_shapes[X.attr('name')]
        bias_name = name + '_offset_bias'
        assert bias_name not in graph

        Y_z, X_z = out_zeros[name], in_zeros[name][0]
        data_shape = [1 if i==0 else s for i,s in enumerate(X_shape)]
        data = nd.full(data_shape, -X_z)
        params[bias_name] = get_nd_op(op_name)(data, **attr)
        params[bias_name] += Y_z

        B = graph[bias_name] = mx.sym.var(bias_name,
                shape=params[bias_name].shape)
        node = get_mxnet_op(op_name)(X, **attr, name=name)
        node = mx.sym.broadcast_add(node, B)
    else:
        logger.info("symbol %-40s processed symmertric by default", name)
        for idx, child in enumerate(childs):
            if in_z[idx] != 0:
                cname = child.attr('name')
                cshape = [1 for _ in infer_shapes[cname]]
                restore_name = cname + '_restore'
                if restore_name in graph:
                    logger.warn("symbol %s has childs(%s)[%s] restore in graph",
                            name, [c.attr('name') for c in childs], idx)
                    restore = graph[restore_name]
                else:
                    restore = mx.sym.var(restore_name, shape=cshape)
                graph[restore_name] = restore

                tmp = mx.sym.broadcast_sub(child, restore)
                new_childs.append(tmp)
                params[restore_name] = nd.array([in_z[idx]]).reshape(cshape)
            else:
                new_childs.append(child)

        node = get_mxnet_op(op_name)(*new_childs, **attr)

        offset_shape = [1 for _ in infer_shapes[name]]
        offset_name = name + '_offset'
        assert offset_name not in graph
        offset = mx.sym.var(offset_name, shape=offset_shape)
        node = mx.sym.broadcast_add(node, offset)
        params[offset_name] = nd.array([out_z]).reshape(offset_shape)

    logger.debug("rewrite symbol %-40s -> %-40s with zeros %-50s -> %s",
            name, node.attr('name'), in_z, out_z)

    infer_shapes[node.attr('name')] = infer_shapes[name]
    idxs['index'] = index + 1
    return node, params

def _sim_requantize_op(sym, scale, params, graph):
    name = sym.attr('name')
    scale_name = name + '_requant_scale'
    assert scale_name not in graph, "scale name %s has existed in graph" \
            % (scale_name)
    scale_sym = graph[scale_name] = mx.sym.var(scale_name, shape=scale.shape)
    params[scale_name] = scale

    requant_op_name = name + '_requant_op'
    assert requant_op_name not in graph
    node = mx.sym.broadcast_mul(sym, scale_sym, name=requant_op_name)
    graph[requant_op_name] = node
    return node
def _is_sim_requantize_op(sym):
    name = sym.attr('name')
    return True if name.endswith('_requant_op') else False
def _realize_sim_requant_op(sym, sb, params, graph, target_bit=8):
    """Requantize Op:
        out = round(sym >> sb)  if sb >  0
        out = round(sym)        if sb == 0
        out = round(sym << -sb) if sb <  0

        round(sym >> sb) = int((int(sym >> (sb - 1)) + 1) >> 1)

        out = clip_int8(out)
    """
    name = sym.attr('name')
    sb_name = name + '_shift_bit'
    assert sb_name not in graph
    sb_sym = mx.sym.var(sb_name, shape=(1,))
    clip_range = 2 ** (target_bit - 1) - 1

    if sb == 0:
        out = mx.sym.clip(sym, a_min=-clip_range, a_max=clip_range)
        return out
    elif sb < 0:
        params[sb_name] = nd.array([2 ** (-sb)])
        out = mx.sym.broadcast_mul(sym, sb_sym)
        out = mx.sym.clip(sym, a_min=-clip_range, a_max=clip_range)
        return out

    params[sb_name] = nd.array([2 ** (sb - 1)])
    n1, n2 = "const_var_1", 'const_var_2'
    var1 = graph[n1] if n1 in graph else mx.sym.var(n1, shape=(1,))
    var2 = graph[n2] if n2 in graph else mx.sym.var(n2, shape=(1,))
    graph[n1], graph[n2] = var1, var2
    params[n1], params[n2] = nd.array([1]), nd.array([2])

    out = sym
    if sb > 1:
        out = mx.sym.broadcast_div(out, sb_sym)
        out = mx.sym.floor(out)
    out = mx.sym.broadcast_add(out, var1)
    out = mx.sym.broadcast_div(out, var2)
    out = mx.sym.floor(out)
    out = mx.sym.clip(out, a_min=-clip_range, a_max=clip_range)
    return out
def _realize_cvm_requant_op(sym, sb, params, graph, target_bit=8):
    name = sym.attr('name')
    requant_op = name + '_cvm_shift'
    assert requant_op not in graph
    if sb == 0:
        return mx.sym.Custom(sym, precision=target_bit,
                name=requant_op, op_type='cvm_clip')
    elif sb < 0:
        return mx.sym.Custom(sym, shift_bit=-sb, precision=target_bit,
                name=requant_op, op_type='cvm_left_shift')
    else:
        return mx.sym.Custom(sym, shift_bit=sb, precision=target_bit,
                name=requant_op, op_type='cvm_right_shift')

def _annotate_symbol(sym, params, graph, inputs_ext,
        scale_shapes, th_dict, scale_helper, target_bits, get_scale):
    logger = logging.getLogger('log.calib.sym.sim.rewrite')
    name = sym.attr('name')
    op_name = sym.attr('op_name')
    childs = sym_iter(sym.get_children())
    attr = sym.list_attr()

    node = sym
    scale = get_scale(th_dict[name], default_target_bit)
    scale_helper[name] = scale.reshape(scale_shapes[name])
    target_bits[name] = default_target_bit
    if op_name == 'null':
        if name in inputs_ext:
            inputs_ext[name]['target_bit'] = target_bits[name]
        return node, params
    elif op_name in disable_requant_ops:
        X_name = childs[0].name
        requant_scale = scale_helper[name] / scale_helper[X_name]
    elif op_name in ['Convolution']:
        X, W = childs[0], childs[1]
        X_name, W_name = X.attr('name'), W.attr('name')
        X_scale = scale_helper[X_name]
        W_scale = scale_helper[W_name]
        in_scale = X_scale * W_scale.reshape(scale_shapes[X_name])
        if attr['no_bias'] == 'False':
            B_name = childs[2].attr('name')
            scale_helper[B_name] = in_scale.reshape(scale_shapes[B_name])
            target_bits[B_name] = bias_target_bit
        requant_scale = scale_helper[name] / in_scale
    elif op_name in ['FullyConnected']:
        X, W = childs[0], childs[1]
        X_name, W_name = X.attr('name'), W.attr('name')
        if attr['no_bias'] == 'False':
            B_name = childs[2].attr('name')
            scale_helper[B_name] = scale_helper[X_name] * scale_helper[W_name]
            target_bits[B_name] = bias_target_bit

        in_scale = scale_helper[X_name] * scale_helper[W_name]
        out_scale = scale_helper[name]
        requant_scale = out_scale / in_scale
    elif op_name == 'broadcast_mul':
        X, W = childs[0], childs[1]
        X_name, W_name = X.attr('name'), W.attr('name')
        in_scale = scale_helper[X_name] * scale_helper[W_name]
        out_scale = scale_helper[name]
        requant_scale = out_scale / in_scale
    elif op_name in ['elemwise_add', 'broadcast_add', 'broadcast_sub']:
        A, B = childs[0], childs[1]
        A_name, B_name = A.attr('name'), B.attr('name')

        A_scale, B_scale = scale_helper[A_name], scale_helper[B_name]
        in_scale = A_scale
        if A_scale > B_scale:
            in_scale, offset = B_scale, B_scale / A_scale
            A = _sim_requantize_op(A, offset, params, graph)
            logger.debug("symbol %s requant scale=%s out=%s",
                    A_name, offset.min().asscalar(),
                    scale_helper[A_name].min().asscalar())
        elif A_scale < B_scale:
            in_scale, offset = A_scale, A_scale / B_scale
            B = _sim_requantize_op(B, offset, params, graph)
            logger.debug("symbol %s requant scale=%s out=%s",
                    B_name, offset.min().asscalar(),
                    scale_helper[A_name].min().asscalar())

        requant_scale = scale_helper[name] / in_scale
        node = get_mxnet_op(op_name)(A, B, **attr, name=name)
    elif op_name in ['sum']:
        X_name = childs[0].attr('name')
        requant_scale = scale_helper[name] / scale_helper[X_name]
    else:
        logger.critical('Unrecognized op:%s(%s) . attrs(%s)', op_name, name, attr)

    if (requant_scale.asnumpy() != 1).any():
        node = _sim_requantize_op(node, requant_scale, params, graph)
        logger.debug("symbol %-40s requant scale=%-20s out=%-20s in=%s",
                name, requant_scale.shape, scale_helper[name].shape,
                [scale_helper[c.attr('name')].shape for c in childs] if childs else [])
    scale_helper[node.attr('name')] = scale_helper[name]
    scale_shapes[node.attr('name')] = scale_shapes[name]
    return node, params
def _annotate_parameters(sym, params, graph, inputs_ext,
        scale_helper, target_bits):
    logger = logging.getLogger('log.annotate.parameters')
    if sym.attr('op_name') != 'null':
        return sym, params
    name = sym.attr('name')
    if name in inputs_ext:
        inputs_ext[name]['scale'] = scale_helper[name]
    elif name in scale_helper:
        params[name] = params[name] * scale_helper[name]
    return sym, params
def _annotate_requantize_op(sym, params, graph, inputs_ext):
    logger = logging.getLogger('log.calib.sim.post.scale')
    if not _is_sim_requantize_op(sym):
        return sym, params
    childs = sym_iter(sym.get_children())
    A, B = childs[0], childs[1]
    scales = params[B.attr('name')]
    scale_shape = scales.shape
    size = np.product(scale_shape)
    if size == 1:
        return sym, params

    slices = [A]
    for idx, s in enumerate(scale_shape):
        if s == 1:
            continue
        begin, end = [None] * len(scale_shape), [None] * len(scale_shape)
        tmp_slices = []
        for sli in slices:
            for start in range(s):
                begin[idx], end[idx] = start, start+1
                tmp = mx.sym.slice(sli, begin=begin, end=end)
                tmp_slices.append(tmp)
        slices = tmp_slices

    logger.info("requantize op %s split into %s slices matching with shape(%s)",
            sym.attr('name'), size, scale_shape)
    nodes = []
    scales = scales.asnumpy().flatten()
    for i in range(size):
        tmp = _sim_requantize_op(slices[i], nd.array([scales[i]]), params, graph)
        nodes.append(tmp)

    for idx, s in enumerate(reversed(scale_shape)):
        if s == 1:
            continue
        assert len(nodes) % s == 0, "nodes: %s, shape: %s" \
                % (len(nodes), scale_shape)
        tmp_nodes = []
        for begin in range(0, len(nodes), s):
            dim = len(scale_shape)-1-idx
            tmp = mx.sym.concat(*nodes[begin:begin+s], dim=dim)
            tmp_nodes.append(tmp)
        nodes = tmp_nodes

    assert len(nodes) == 1
    node = nodes[0]
    return node, params
def _realize_symbol(sym, params, graph, inputs_ext):
    logger = logging.getLogger('log.calib.realize')
    if not _is_sim_requantize_op(sym):
        return sym, params

    childs = sym_iter(sym.get_children())
    X, B = childs[0], childs[1]
    X_name, B_name = X.attr('name'), B.attr('name')

    scale = params[B_name].asscalar()
    if scale == 1:
        logger.debug("layer %s skip realize requant with one", sym.attr('name'))
        return X, params
    frac, sb = sim.extract_float(scale)
    Y_range = 2 ** (default_target_bit - 1) - 1
    A_range = Y_range / scale

    if frac == 0:
        var0, _ = op_const(0, graph, var=mx.sym.var)
        params[var0.attr('name')] = nd.array([0])
        node = mx.sym.broadcast_mul(X, var0)
        logger.debug("layer %s skip realize requant with zero", sym.attr('name'))
        return node, params

    # Y = Z * frac * (2 ** sb) <=>
    # Y = (Z -> Int16) * (frac -> Int16) * (2 ** sb_)
    A_target_bit, B_target_bit = 16, 16
    A_bit = math.ceil(math.log2(A_range)) + 1
    B_bit = math.ceil(math.log2(frac)) + 1
    A_target_bit = min(A_bit, A_target_bit)
    B_target_bit = min(B_bit, B_target_bit)
    A_target_bit = 32 - B_target_bit if B_target_bit < 16 else A_target_bit
    B_target_bit = 32 - A_target_bit if A_target_bit < 16 else B_target_bit
    A_target_bit = min(A_bit, A_target_bit)
    B_target_bit = min(B_bit, B_target_bit)
    A_sb, B_sb = A_bit - A_target_bit, B_bit - B_target_bit
    Y_sb = (-sb) - A_sb - B_sb

    X = _realize_cvm_requant_op(X, A_sb, params, graph, A_target_bit)
    params[B_name] = nd.array([round(frac / (2 ** B_sb))])
    B_range = 2 ** (B_target_bit - 1) - 1
    params[B_name] = nd.clip(params[B_name],
            a_min=-B_range, a_max=B_range)
    attr = { 'precision': str(B_target_bit) }
    B = mx.sym.var(B_name, shape=(1,), attr=attr)
    node = mx.sym.broadcast_mul(X, B)
    node = _realize_cvm_requant_op(node, Y_sb, params, graph, target_bit=8)
    logger.debug("layer %s Y(%s >> %s) X(%s|%s >> %s) B(%s|%s vs. %s %s >> %s)",
            sym.attr('name'), Y_range, Y_sb, A_range, A_bit, A_sb, B_range,
            B_bit, frac, sb, B_sb)
    return node, params
def _realize_parameters(sym, params, graph, inputs_ext,
        target_bits={}, params_sim={}):
    logger = logging.getLogger('log.calib.realize.parameters')
    name = sym.attr('name')
    attr = sym.list_attr()
    if 'precision' not in attr or name in inputs_ext:
        return sym, params
    target_bit = int(attr['precision'])
    data = params[name]
    params[name] = sim.int_realize(data, target_bit, logger=logger)
    # calculate error
    error = params[name].astype('float32') - data
    error_rate = error / data
    rate = nd.norm(error_rate).asscalar() / np.product(data.shape)
    if rate > 0.001:
        logger.warn("realize parameter %-60s average rate=%10.9f shape=%s",
                name, rate, data.shape)
    else:
        logger.debug("realize parameter %-60s average rate=%10.9f shape=%s",
                name, rate, data.shape)
    return sym, params

def _sim_scale(sym, params, graph, inputs_ext,
        th_dict, scale_helper, target_bits):
    name = sym.attr('name')
    op_name = sym.attr('op_name')
    childs = sym_iter(sym.get_children())
    attr = sym.list_attr()

    # calculate simulate scale
    target_bits[name] = default_target_bit
    scale = sim.get_sim_scale(th_dict[name], default_target_bit)
    scale_helper[name] = nd.array([scale])
    if op_name in ['Convolution', 'FullyConnected']:
        X_name, W_name = childs[0].attr('name'), childs[1].attr('name')
        # if 'num_group' in attr and attr['num_group'] == attr['num_filter']:
        #     channel = int(attr['num_filter'])
        #     weight = params[W_name]
        #     weight_shape = weight.shape
        #     assert weight_shape[0] == channel
        #     weight_scales = [None] * channel
        #     for idx in range(channel):
        #         threshold = _get_thresholds(weight[idx,:])
        #         weight_scales[idx] = sim.get_sim_scale(threshold, default_target_bit)
        #     scale_helper[W_name] = nd.array(weight_scales).reshape(channel, 1, 1, 1)

        if attr['no_bias'] == 'False':
            X_name, W_name = childs[0].attr('name'), childs[1].attr('name')
            B_name = childs[2].attr('name')

            w_size = np.product(scale_helper[W_name].shape)
            flat_w_scale = scale_helper[W_name].reshape((w_size))
            scale_helper[B_name] = scale_helper[X_name] * flat_w_scale
            target_bits[B_name] = bias_target_bit
    return sym, params
def _sim_rewrite(sym, params, graph, inputs_ext, scale_helper, params_sim):
    logger = logging.getLogger('log.calib.sym.sim.rewrite')

    name = sym.attr('name')
    op_name = sym.attr('op_name')
    childs = sym_iter(sym.get_children())
    attr = sym.list_attr()

    if op_name == 'null':
        if name in inputs_ext:
            inputs_ext[name]['scale'] = scale_helper[name]
            sim.save_data_scale(name, scale_helper[name], params)
        else:
            params[name] = params[name] * scale_helper[name]
            params_sim.append(name)
        return sym, params
    elif op_name in disable_requant_ops:
        return sym, params

    node = sym
    if op_name in ['Convolution']:
        X, W = childs[0], childs[1]
        X_name, W_name = X.attr('name'), W.attr('name')
        if False and attr['num_group'] == attr['num_filter']:
            channel = int(attr['num_filter'])
            flat_w_scale = scale_helper[W_name].reshape(1, channel, 1, 1)
            in_scale = scale_helper[X_name] * flat_w_scale
        else:
            in_scale = scale_helper[X_name] * scale_helper[W_name]
        requant_scale = scale_helper[name] / in_scale
    elif op_name in ['FullyConnected', 'broadcast_mul']:
        X, W = childs[0], childs[1]
        X_name, W_name = X.attr('name'), W.attr('name')
        in_scale = scale_helper[X_name] * scale_helper[W_name]
        out_scale = scale_helper[name]
        requant_scale = out_scale / in_scale
    elif op_name in ['elemwise_add', 'broadcast_add', 'broadcast_sub']:
        A, B = childs[0], childs[1]
        A_name, B_name = A.attr('name'), B.attr('name')

        A_scale, B_scale = scale_helper[A_name], scale_helper[B_name]
        if A_scale > B_scale:
            in_scale, offset = B_scale, B_scale / A_scale
            A = _sim_requantize_op(A, offset, params, graph)
            logger.debug("symbol %s requant scale=%s out=%s",
                    A_name, offset.min().asscalar(),
                    scale_helper[A_name].min().asscalar())
        elif A_scale < B_scale:
            in_scale, offset = A_scale, A_scale / B_scale
            B = _sim_requantize_op(B, offset, params, graph)
            logger.debug("symbol %s requant scale=%s out=%s",
                    B_name, offset.min().asscalar(),
                    scale_helper[A_name].min().asscalar())
        else:
            in_scale = A_scale

        requant_scale = scale_helper[name] / in_scale
        node = get_mxnet_op(op_name)(A, B, **attr, name=name)
    elif op_name in ['sum']:
        X_name = childs[0].attr('name')
        requant_scale = scale_helper[name] / scale_helper[X_name]
    else:
        logger.critical('Unrecognized op:%s(%s)', op_name, name)
        new_childs = []
        for child in childs:
            child_name = child.attr('name')
            scale_name = child_name + '_restore'
            if scale_name in graph:
                scale = graph[scale_name]
            else:
                scale = graph[scale_name] = mx.sym.var(scale_name, shape=(1,))

            restore = scale_helper[child_name]
            params[scale_name] = nd.array([restore])
            tmp = mx.sym.broadcast_div(child, scale)
            new_childs.append(tmp)

        node = get_mxnet_op(op_name)(*new_childs, **attr, name=name)
        requant_scale = scale_helper[name]

    if (requant_scale.asnumpy() != 1).any():
        node = _sim_requantize_op(node, requant_scale, params, graph)
        logger.debug("symbol %s requant scale=%s out=%s in=%s",
                name, requant_scale.min().asscalar(),
                scale_helper[name].min().asscalar(),
                [scale_helper[c.attr('name')].min().asscalar() for c in childs])

    scale_helper[node.attr('name')] = scale_helper[name]
    return node, params
def _sim_post_scale(sym, params, graph, inputs_ext):
    logger = logging.getLogger('log.calib.sim.post.scale')
    if not _is_sim_requantize_op(sym):
        return sym, params

    name = sym.attr('name')
    childs = sym_iter(sym.get_children())

    A, B = childs[0], childs[1]
    A_attr = A.list_attr()
    A_name, B_name = A.attr('name'), B.attr('name')
    B_size = np.product(params[B_name].shape)
    if B_size == 1:
        return sym, params

    assert A.attr('op_name') == 'Convolution' and \
            A_attr['num_group'] == A_attr['num_filter']
    channel = int(A_attr['num_filter'])
    scales = params[B_name].asnumpy().flatten()
    logger.info("layer %s split into %s slices matching with shape(%s)",
            sym.attr('name'), B_size, params[B_name].shape)
    nodes = []
    begin, end = [None]*4, [None]*4
    for i in range(channel):
        begin[1], end[1] = i, i+1
        tmp = mx.sym.slice(A, begin=begin, end=end)
        tmp = _sim_requantize_op(tmp, nd.array([scales[i]]), params, graph)
        nodes.append(tmp)
    node = mx.sym.concat(*nodes, dim=1)
    return node, params
def _sim_realize(sym, params, graph, inputs_ext):
    logger = logging.getLogger('log.calib.realize')
    if not _is_sim_requantize_op(sym):
        return sym, params

    childs = sym_iter(sym.get_children())
    X, B = childs[0], childs[1]
    X_name, B_name = X.attr('name'), B.attr('name')

    scale = params[B_name].asscalar()
    if scale == 1:
        logger.debug("layer %s skip realize requant with one", sym.attr('name'))
        return X, params
    frac, sb = sim.extract_float(scale)
    Y_range = 2 ** (default_target_bit - 1) - 1
    A_range = Y_range / scale
    if frac == 0:
        var0, _ = op_const(0, graph, var=mx.sym.var)
        params[var0.attr('name')] = nd.array([0])
        node = mx.sym.broadcast_mul(X, var0)
        logger.debug("layer %s skip realize requant with zero", sym.attr('name'))
        return node, params

    # Y = Z * frac * (2 ** sb) <=>
    # Y = (Z -> Int16) * (frac -> Int16) * (2 ** sb_)
    A_target_bit, B_target_bit = 16, 16
    A_bit = math.ceil(math.log2(A_range)) + 1
    B_bit = math.ceil(math.log2(frac)) + 1
    A_target_bit = min(A_bit, A_target_bit)
    B_target_bit = min(B_bit, B_target_bit)
    A_target_bit = 32 - B_target_bit if B_target_bit < 16 else A_target_bit
    B_target_bit = 32 - A_target_bit if A_target_bit < 16 else B_target_bit
    A_target_bit = min(A_bit, A_target_bit)
    B_target_bit = min(B_bit, B_target_bit)
    A_sb, B_sb = A_bit - A_target_bit, B_bit - B_target_bit
    Y_sb = (-sb) - A_sb - B_sb

    X = _realize_sim_requant_op(X, A_sb, params, graph, A_target_bit)
    params[B_name] = nd.array([round(frac / (2 ** B_sb))])
    B_range = 2 ** (B_target_bit - 1) - 1
    params[B_name] = nd.clip(params[B_name],
            a_min=-B_range, a_max=B_range)
    node = mx.sym.broadcast_mul(X, B)
    node = _realize_sim_requant_op(node, Y_sb, params, graph, target_bit=8)
    logger.debug("layer %s Y(%s >> %s) X(%s|%s >> %s) B(%s|%s vs. %s %s >> %s)",
            sym.attr('name'), Y_range, Y_sb, A_range, A_bit, A_sb, B_range,
            B_bit, frac, sb, B_sb)
    return node, params
def _sim_realize_requantize_op(sym, params, graph, inputs_ext):
    logger = logging.getLogger('log.calib.realize')
    if not _is_sim_requantize_op(sym):
        return sym, params

    childs = sym_iter(sym.get_children())
    node = sym
    A, B = childs[0], childs[1]
    A_name, B_name = A.attr('name'), B.attr('name')
    scale_shape = params[B_name].shape
    slices = [A]
    for idx, s in enumerate(scale_shape):
        if s == 1:
            continue
        begin, end = [None] * len(scale_shape), [None] * len(scale_shape)
        tmp_slices = []
        for start in range(s):
            begin[idx], end[idx] = start, start+1
            for sli in slices:
                tmp = mx.sym.slice(sli, begin=begin, end=end)
                tmp_slices.append(tmp)
        slices = tmp_slices

    size = np.product(scale_shape)
    assert len(slices) == size
    logger.info("layer %s split into %s slices matching with shape(%s)",
            sym.attr('name'), size, scale_shape)
    scales = params[B_name].asnumpy().flatten()
    nodes = []
    for idx in range(size):
        data, scale = slices[idx], scales[idx]
        requant_name = B.attr('name') + '_' + str(idx)
        frac, sb = sim.extract_float(scale)
        Y_range = 2 ** (default_target_bit - 1) - 1
        A_range = Y_range / scale

        if frac == 0:
            var0, _ = op_const(0, graph, var=mx.sym.var)
            params[var0.attr('name')] = nd.array([0])
            tmp = mx.sym.broadcast_mul(data, var0)
            nodes.append(tmp)
            logger.debug("layer %s skip realize requant with zero", sym.attr('name'))
            continue

        # Y = Z * frac * (2 ** sb) <=>
        # Y = (Z -> Int16) * (frac -> Int16) * (2 ** sb_)
        A_target_bit, B_target_bit = 16, 16
        A_bit = math.ceil(math.log2(A_range)) + 1
        B_bit = math.ceil(math.log2(frac)) + 1
        A_target_bit = min(A_bit, A_target_bit)
        B_target_bit = min(B_bit, B_target_bit)
        A_target_bit = 32 - B_target_bit if B_target_bit < 16 else A_target_bit
        B_target_bit = 32 - A_target_bit if A_target_bit < 16 else B_target_bit
        A_target_bit = min(A_bit, A_target_bit)
        B_target_bit = min(B_bit, B_target_bit)
        A_sb, B_sb = A_bit - A_target_bit, B_bit - B_target_bit
        Y_sb = (-sb) - A_sb - B_sb

        data = _realize_sim_requant_op(data, A_sb, params, graph, A_target_bit)
        params[requant_name] = nd.array([round(frac / (2 ** B_sb))])
        B_range = 2 ** (B_target_bit - 1) - 1
        params[requant_name] = nd.clip(params[requant_name],
                a_min=-B_range, a_max=B_range)
        assert requant_name not in graph
        requant_sym = mx.sym.var(requant_name, shape=(1,))
        tmp = mx.sym.broadcast_mul(data, requant_sym)
        tmp = _realize_sim_requant_op(tmp, Y_sb, params, graph, target_bit=8)
        nodes.append(tmp)

    for idx, s in enumerate(reversed(scale_shape)):
        if s == 1:
            continue
        assert len(nodes) % s == 0, "nodes: %s, shape: %s" \
                % (len(nodes), scale_shape)
        tmp_nodes = []
        for begin in range(0, len(nodes), s):
            dim = len(scale_shape)-1-idx
            tmp = mx.sym.concat(*nodes[begin:begin+s], dim=dim)
            tmp_nodes.append(tmp)
        nodes = tmp_nodes

    assert len(nodes) == 1
    node = nodes[0]
    return node, params

def _simple_sim_scale(sym, params, graph, inputs_ext,
        th_dict, scale_helper, target_bits):
    logger = logging.getLogger('log.calib.sym.out_shift_bits')
    name = sym.attr('name')
    op_name = sym.attr('op_name')
    childs = sym_iter(sym.get_children())
    attr = sym.list_attr()

    # update params bias shift_bits
    if op_name in ['Convolution', 'FullyConnected']:
        if attr['no_bias'] == 'False':
            X_name = childs[0].attr('name')
            W_name = childs[1].attr('name')
            B_name = childs[2].attr('name')
            scale_helper[B_name] = scale_helper[X_name] * scale_helper[W_name]
            target_bits[B_name] = bias_target_bit
    # calculate output shift_bits
    scale_helper[name] = sim.get_simple_sim_scale(th_dict[name],
            default_target_bit)
    target_bits[name] = default_target_bit
    return sym, params
def _simple_sim_realize_requantize_op(sym, params, graph, inputs_ext):
    logger = logging.getLogger('log.calib.realize')
    childs = sym_iter(sym.get_children())
    node = sym
    if _is_sim_requantize_op(sym):
        A, B = childs[0], childs[1]
        A_name, B_name = A.attr('name'), B.attr('name')

        scale = params[B_name].asscalar()
        frac, sb = sim.extract_float(scale)
        assert frac == 1, \
            "extract parameter:%s float:%s fraction:%s shift bit:%s" \
                % (sym.attr('name'), scale, frac, sb)

        sb = - int(sb)
        node = _realize_sim_requant_op(A, sb, params, graph)
        logger.debug("realize requant operator %-60s scale=%-20s fraction=%s shift bit=%s",
                sym.attr('name'), scale, frac, sb)
    return node, params

# interface API
def sym_simulate(symbol, params, inputs_ext, calib_data, ctx):
    logger = logging.getLogger('log.simulate')
    scale_shapes = {}
    topo_visit(symbol, params, get_op=get_mxnet_op,
            logger=logger, inputs_ext=inputs_ext,
            callback=_collect_symbol_ext, scale_shapes=scale_shapes)
    th_dict = {}
    topo_visit(symbol, params, get_op=get_mxnet_op,
            logger=logger, inputs_ext=inputs_ext,
            callback=_calib_sym_collect_thresholds, scale_shapes=scale_shapes,
            th_dict=th_dict, calib_data=calib_data,
            calib_mode='naive', ctx=ctx)
    scale_helper, target_bits = {}, {}
    symbol, params = topo_visit(symbol, params, get_op=get_mxnet_op,
            logger=logger, inputs_ext=inputs_ext,
            callback=_annotate_symbol, th_dict=th_dict, scale_shapes=scale_shapes,
            scale_helper=scale_helper, target_bits=target_bits,
            get_scale=sim.get_sim_scale)
    _, params = topo_visit(symbol, params, get_op=get_mxnet_op,
            logger=logger, inputs_ext=inputs_ext,
            callback=_annotate_parameters,
            scale_helper=scale_helper, target_bits=target_bits)
    symbol, params = topo_visit(symbol, params, get_op=get_mxnet_op,
            logger=logger, inputs_ext=inputs_ext,
            callback=_annotate_requantize_op)
    symbol, params = sym_attach_attrs(symbol, params, inputs_ext,
            precision=target_bits)
    return symbol, params

def sym_realize(symbol, params, inputs_ext):
    logger = logging.getLogger('log.realize')
    _, params = topo_visit(symbol, params, get_op=get_mxnet_op,
           logger=logger, inputs_ext=inputs_ext,
           callback=_realize_parameters)
    symbol, params = topo_visit(symbol, params, get_op=get_mxnet_op,
           logger=logger, inputs_ext=inputs_ext,
           callback=_realize_symbol)

    def _check_int_params(params, arg):
       param = params[arg]
       msg = "key:%s value:%s"%(arg, param)
       flat = param.asnumpy().flatten()
       assert all(flat >= INT32_MIN) and all(flat <= INT32_MAX), msg
       assert all(flat.astype('int32').astype(flat.dtype) == flat), msg

    params = examine_parameters(symbol, params, inputs_ext,
          callback=_check_int_params)
    return symbol, params

def sym_calib_sim_quant(symbol, params, inputs_ext, calib_data, ctx):
    logger = logging.getLogger('log.simulate')

    th_dict, target_bits, scale_helper, params_sim = {}, {}, {}, []
    # simulate
    topo_visit(symbol, params, get_op=get_mxnet_op,
            logger=logger, inputs_ext=inputs_ext,
            callback=_calib_sym_collect_thresholds,
            th_dict=th_dict, calib_data=calib_data,
            calib_mode='naive', ctx=ctx)
    topo_visit(symbol, params, get_op=get_mxnet_op,
            logger=logger, inputs_ext=inputs_ext,
            callback=_sim_scale, th_dict=th_dict,
            scale_helper=scale_helper, target_bits=target_bits)
    sym, params = topo_visit(symbol, params, get_op=get_mxnet_op,
            logger=logger, inputs_ext=inputs_ext,
            callback=_sim_rewrite,
            scale_helper=scale_helper, params_sim=params_sim)
    sym, params = topo_visit(sym, params, get_op=get_mxnet_op,
            logger=logger, inputs_ext=inputs_ext,
            callback=_sim_post_scale)

    # realize
    _, params = topo_visit(sym, params, get_op=get_mxnet_op,
           logger=logger, inputs_ext=inputs_ext,
           callback=_realize_parameters,
           target_bits=target_bits, params_sim=params_sim)
    sym, params = topo_visit(sym, params, get_op=get_mxnet_op,
           logger=logger, inputs_ext=inputs_ext,
           callback=_sim_realize)

    def _check_int_params(params, arg):
        param = params[arg]
        msg = "key:%s value:%s"%(arg, param)
        flat = param.asnumpy().flatten()
        assert all(flat >= INT32_MIN) and all(flat <= INT32_MAX), msg
        assert all(flat.astype('int32').astype(flat.dtype) == flat), msg

    params = examine_parameters(sym, params, inputs_ext,
           allows=['data_scale'], callback=_check_int_params)

    return sym, params, th_dict

def sym_calib_simple_sim_quant(symbol, params, inputs_ext,
        calib_data=None, th_dict={}, ctx=mx.cpu()):
    logger = logging.getLogger("log.calib.sym")
    if not th_dict:
        topo_visit(symbol, params, get_op=get_mxnet_op,
                logger=logger, inputs_ext=inputs_ext,
                callback=_calib_sym_collect_thresholds,
                th_dict=th_dict, calib_data=calib_data, ctx=ctx)

    scale_helper, target_bits, params_sim = {}, {}, []
    # simulate
    topo_visit(symbol, params, get_op=get_mxnet_op,
            logger=logger, inputs_ext=inputs_ext,
            callback=_simple_sim_scale, th_dict=th_dict,
            scale_helper=scale_helper, target_bits=target_bits)
    sym, params = topo_visit(symbol, params, get_op=get_mxnet_op,
            logger=logger, inputs_ext=inputs_ext,
            callback=_sim_rewrite,
            scale_helper=scale_helper, params_sim=params_sim)

    # realize
    _, params = topo_visit(sym, params, get_op=get_mxnet_op,
            logger=logger, inputs_ext=inputs_ext,
            callback=_realize_parameters,
            target_bits=target_bits, params_sim=params_sim)
    sym, params = topo_visit(sym, params, get_op=get_mxnet_op,
            logger=logger, inputs_ext=inputs_ext,
            callback=_simple_sim_realize_requantize_op)

    def _check_int_params(params, arg):
       param = params[arg]
       msg = "key:%s value:%s"%(arg, param)
       flat = param.asnumpy().flatten()
       assert all(flat >= INT32_MIN) and all(flat <= INT32_MAX), msg
       assert all(flat.astype('int32').astype(flat.dtype) == flat), msg

    params = examine_parameters(sym, params, inputs_ext,
            allows=['data_scale'], callback=_check_int_params)

    return sym, params

def sym_calib_quantize(symbol, params, inputs_ext, calib_data, ctx):
    logger = logging.getLogger("log.calib.quantize")

    ops = sym_collect_attr(symbol)
    print (ops)

    th_dict, in_zeros, out_zeros= {}, {}, {}
    topo_visit(symbol, params, get_op=get_mxnet_op,
            logger=logger, inputs_ext=inputs_ext,
            callback=_calib_sym_collect_thresholds,
            th_dict=th_dict, calib_data=calib_data, ctx=ctx)

    topo_visit(symbol, params, get_op=get_mxnet_op,
            logger=logger, inputs_ext=inputs_ext,
            callback=_calib_sym_zero_symmetric,
            th_dict=th_dict, in_zeros=in_zeros, out_zeros=out_zeros)

    infer_shapes = sym_infer_shape(symbol, params, inputs_ext)
    indexes = {'index': 0}
    sym, params = topo_visit(symbol, params, get_op=get_mxnet_op,
            logger=logger, inputs_ext=inputs_ext,
            callback=_calib_sym_zero_rewrite,
            in_zeros=in_zeros, out_zeros=out_zeros,
            infer_shapes=infer_shapes, idxs=indexes)

    return sym, params
