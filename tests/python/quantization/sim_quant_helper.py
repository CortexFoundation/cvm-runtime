import logging
import math

import mxnet as mx
from mxnet import ndarray as nd

def get_zero_symmetric(threshold):
    min_range, max_range = threshold

    if max_range == min_range:
        return 0
    else:
        return (max_range + min_range) / 2

def save_ins_ext(params, inputs_ext):
    for name in inputs_ext:
        ext = inputs_ext[name]
        scale, target_bit = ext['scale'], ext['target_bit']
        params[name+'_scale'] = ext['scale']
        params[name+'_target_bit'] = nd.array([ext['target_bit']])
def load_ins_ext(params, inputs_ext):
    for name in inputs_ext:
        ext = inputs_ext[name]
        ext['scale'] = params[name+'_scale']
        ext['target_bit'] = params[name+'_target_bit']
def load_sim_data(data, name, inputs_ext):
    return data * inputs_ext[name]['scale']
def load_real_data(data, name, inputs_ext):
    logger = logging.getLogger('log.data.load')
    data = load_sim_data(data, name, inputs_ext)
    return int_realize(data, inputs_ext[name]['target_bit'].asscalar(),
            logger=logger)

def save_data_scale(name, scale, params):
    params[name+'_scale'] = scale

def load_quant_data(data, name, params):
    data_name = name + '_scale'
    assert data_name in params, "data scale %s not in params dict %s" \
            % (data_name, params.keys())
    return data*params[data_name]

def get_simple_sim_scale(threshold, target_bit):
    min_range, max_range = threshold
    alpha = max(abs(min_range), abs(max_range))

    bit = math.ceil(math.log2(alpha))
    shift_bit = target_bit - 1 - bit
    return 2 ** shift_bit

def get_sim_scale(thresholds, target_bit):
    assert len(thresholds.shape) == 2
    size = thresholds.shape[0]
    assert thresholds.shape[1] == 2
    scales = []
    for value in thresholds:
        alpha = value.abs().max().asscalar()
        sim_max = 2 ** (target_bit - 1) - 1
        scales.append(sim_max / alpha)
    return nd.array(scales)

def int_realize(data, target_bit, logger=logging):
    out = data.round()
    clip_range = 2 ** (target_bit - 1) - 1
    if logger and out.abs().max() > clip_range:
        logger.warn("quant out of range int%d with data=<%s,%s,%s>",
                target_bit,
                out.asnumpy().flatten()[0],
                out.max().asnumpy(),
                out.min().asnumpy())

    out = out.clip(a_min=-clip_range, a_max=clip_range)
    return out

def parse_nd_float(array):
    shape= array.shape
    array = array.asnumpy().flatten()
    size = len(array)
    fracs, sbs = [None] * size, [None] * size
    for idx in range(size):
        fracs[idx], sbs[idx] = extract_float(array[idx])
    return nd.array(fracs).reshape(shape), nd.array(sbs).reshape(shape)

def extract_float(number):
    sign, binary = float_bin(number, 24)
    dot_idx = binary.find('.')
    binary = binary.replace('.', '')
    use_idx = binary.rfind('1') + 1
    if use_idx == 0:
        return 0, 0
    sb = dot_idx - use_idx
    frac = sign * int(binary[:use_idx], 2)
    return frac, sb

def float_bin(number, places = 24):
    """Single precision float convert into binary
    """
    sign = 1 if number >= 0 else -1
    number = abs(number)
    whole, dec = int(number), number - int(number)
    res = bin(whole).lstrip('0b') + '.'
    if len(res) > places:
        return res
    for x in range(places - len(res) + 1):
        dec *= 2
        whole, dec = int(dec), dec - int(dec)
        res += str(whole)
    return sign, res

def nd_quant(data, shift_bits=None, target_bit=8,
        logger=logging):
    real_bits = target_bit - 1 # real bits is decreased for int type
    if shift_bits is None:
        assert isinstance(data, nd.NDArray)
        shift_bits, _ = _nd_quant_params(data, real_bits)

    out = (data / (2 ** (shift_bits))).round()

    clip_range = 2 ** real_bits - 1
    if logger and out.abs().max() > clip_range:
        logger.warn("quant out of range int%d with data=<%s,%s,%s>, sb=%s",
                target_bit,
                out.asnumpy().flatten()[0],
                out.max().asnumpy(),
                out.min().asnumpy(),
                shift_bits.asnumpy())
    elif logger:
        logger.debug("quant into int%d with data=<%s,%s,%s>, sb=%s",
                target_bit,
                out.asnumpy().flatten()[0],
                out.max().asnumpy(),
                out.min().asnumpy(),
                shift_bits.asnumpy())

    out = out.clip(a_min=-clip_range, a_max=clip_range)
    return out, shift_bits


def sym_quant(sym, params, graph, shift_bits, target_bit=8):
    scale_name = sym.attr('name') + '_requant_scale'
    assert scale_name not in graph
    scale_sym = mx.sym.var(scale_name, shape=(1,))
    graph[scale_name] = scale_sym

    n1, n2 = "const_var_1", 'const_var_2'
    sym_1 = graph[n1] if n1 in graph else mx.sym.var(n1, shape=(1,))
    sym_2 = graph[n2] if n2 in graph else mx.sym.var(n2, shape=(1,))
    graph[n1], graph[n2] = sym_1, sym_2

    assert shift_bits.shape == (1,)
    if shift_bits < 0:
        scale = 2 ** (-shift_bits)
        out = mx.sym.broadcast_mul(sym, scale_sym)
    elif shift_bits == 0:
        out, scale = sym, nd.zeros((1,))
    else:
        scale = 2 ** (shift_bits - 1)
        out = mx.sym.broadcast_div(sym, scale_sym)
        out = mx.sym.floor(out)
        out = mx.sym.broadcast_add(out, sym_1)
        out = mx.sym.broadcast_div(out, sym_2)
        out = mx.sym.floor(out)

    params[n1] = nd.array([1])
    params[n2] = nd.array([2])
    params[scale_name] = scale

    clip_range = 2 ** (target_bit - 1) -1
    out = mx.sym.clip(out, a_min=-clip_range, a_max=clip_range)
    return out, params


def _nd_quant_params(data, real_bits, use_asymmetric=True,
        eliminate_outlier=False):
    if eliminate_outlier:
        mean = data.mean()
        var = ((data - mean) * (data - mean)).mean()
        std = var.sqrt()
        norm_data = ((data - mean) / std).clip(a_min=-4, a_max=4) * std + data.mean()
    else:
        norm_data = data

    if use_asymmetric:
        alpha = norm_data.abs().max()
        offset = nd.zeros((1,))
    else:
        min_v = norm_data.min()
        max_v = norm_data.max()
        alpha = (max_v - min_v) / 2
        offset = (alpha - max_v).floor()

    assert any(alpha != 0)

    bits = alpha.log2().ceil()
    shift_bits = bits - real_bits

    return shift_bits, offset
