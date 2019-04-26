import logging
from enum import Enum
from mxnet import symbol, sym
from mxnet import ndarray as nd

sym.contrib.cond

class CalibMode(Enum):
    NONE = 0
    NAIVE = 1
    CALIBRATION = 2


class QuantFlag():
    def __init__(self, is_fuse_bn=True, calib_mode=CalibMode.NONE,
            allowed_layers=[], disabled_layers=[],
            log_level=logging.INFO, matrix_decomposition=False,
            use_scalar=False):
        self.is_fuse_bn = is_fuse_bn
        assert isinstance(calib_mode, CalibMode)
        self.calib_mode = calib_mode
        self.log_level = log_level
        self.allowed_layers = allowed_layers
        self.disabled_layers = disabled_layers

        self.matrix_decomposition = matrix_decomposition
        self.use_scalar = use_scalar

DEFAULT_TARGET_BITS = 7
BIAS_TARGET_BITS= (DEFAULT_TARGET_BITS+1)*4-1

def quant_helper(data, shift_bits=None, target_bits=DEFAULT_TARGET_BITS,
        logger=None, msg="", F=nd, **kwargs):
    if shift_bits is None:
        shift_bits, _ = calib_quant_params(data, target_bits, **kwargs)

    out = shift_round(F, data, shift_bits)
    clip_range = 2 ** target_bits - 1
    if isinstance(data, nd.NDArray):
        if logger and out.abs().max() > clip_range:
            logger.warn("quant %s out of range int%d with data=<%s,%s,%s>, sb=%s",
                    msg,
                    target_bits+1,
                    out.asnumpy().flatten()[0],
                    out.max().asnumpy(),
                    out.min().asnumpy(),
                    shift_bits.asnumpy())
        elif logger:
            logger.debug("quant %s into int%d with data=<%s,%s,%s>, sb=%s",
                    msg,
                    target_bits+1,
                    out.asnumpy().flatten()[0],
                    out.max().asnumpy(),
                    out.min().asnumpy(),
                    shift_bits.asnumpy())

    out = out.clip(a_min=-clip_range, a_max=clip_range)
    return out, shift_bits

def calib_quant_params(data, target_bits, use_asymmetric=True,
        eliminate_outlier=False):
    """ Used in calibration pass
    """
    assert isinstance(data, nd.NDArray)

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
    shift_bits = bits - target_bits

    return shift_bits, offset

def shift_round(F, data, shift_bits):
    """ Use round(x) instead of floor(x), which can lead to large accuracy drop
        in inference.
    """
    if isinstance(data, nd.NDArray):
        power = F.power(2, shift_bits-1)
        out = F.floor(F.broadcast_div(data, power) + 1)
        out = F.floor(out / 2)
        return out

    assert isinstance(data, sym.Symbol)

    def left_shift():
        power = F.pow(2, -shift_bits)
        out = F.floor(F.broadcast_mul(data, power))
        return out

    def right_shift():
        power = F.pow(2, shift_bits-1)
        out = F.floor(F.broadcast_div(data, power) + 1)
        out = F.floor(out / 2)
        return out

    out = F.contrib.cond(shift_bits < 1, left_shift, right_shift)

    return out
