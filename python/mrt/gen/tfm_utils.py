from .tfm_types import get_buffer, SC_TYPE_EXP

from mrt import tfm_utils as tutils

def get_bit_exp(opt):
    return tutils.get_bit(opt)

def scale_exp(th, prec):
    return tutils.scale(th, prec)

def get_buffer_exp(scale):
    return get_buffer(SC_TYPE_EXP, scale)

