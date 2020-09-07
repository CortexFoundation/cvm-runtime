from .tfm_types import get_buffer, BUF_TYPE_EXP, QUANT_TYPE_EXP, \
                       get_quantizer

from mrt import tfm_utils as tutils

def get_bit_exp(opt):
    return tutils.get_bit(opt)

def scale_exp(th, prec):
    return tutils.scale(th, prec)

def get_buffer_exp(scale):
    return get_buffer(BUF_TYPE_EXP, scale)

def get_quantizer_exp():
    return get_quantizer(QUANT_TYPE_EXP)

def get_range_exp(prec):
    return tutils.get_range(prec)

