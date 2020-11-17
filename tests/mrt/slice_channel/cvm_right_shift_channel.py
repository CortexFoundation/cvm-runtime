import numpy as np
import mxnet as mx
from mxnet import ndarray as nd
from mrt.V2 import cvm_op
from mrt import cvm_op
import slice_utils as utils

def test_symbol(precs=None, sbs=None):
    assert precs is not None and sbs is not None, \
        "please specify attrs for precs, sbs"
    X = mx.sym.var("var")
    op = mx.sym.Custom(
        X, precs=precs, sbs=sbs, name="out",
        op_type="cvm_right_shift_channel")

def forward_right_shift_channel(X, precs=None, sbs=None):
    assert precs is not None and sbs is not None, \
        "please specify attrs for precs, sbs"
    num_channel = X.shape[1]
    out = nd.Custom(
        X, precs=precs, sbs=sbs,
        op_type="cvm_right_shift_channel")
    return out

def forward_slice_right_shift(X, precs=None, sbs=None):
    assert precs is not None and sbs is not None, \
        "please specify attrs for precs, sbs"
    precs = [eval(prec) for prec in precs.split(',')]
    sbs = [eval(sb) for sb in sbs.split(',')]
    xshp = X.shape
    ndim = len(xshp)
    assert xshp[1] % len(precs) == 0, \
        "division error, xshp[1]: {}, len(precs): {}".format( \
        (xshp[1], len(precs)))
    inc = xshp[1] // len(precs)
    nodes = []
    for i in range(0, xshp[1], inc):
        begin = (None,i,) + (None,)*(ndim-2)
        end = (None,i+inc) + (None,)*(ndim-2)
        node = nd.slice(X, begin=begin, end=end)
        j = i // inc
        node = nd.Custom(
            node, precision=precs[j], shift_bit=sbs[j],
            op_type="cvm_right_shift")
        nodes.append(node)
    out = nd.concat(*nodes, dim=1)
    return out

if __name__ == "__main__":
    test_symbol(precs="8,8,8", sbs="8,8,8")

    # case 1
    shps = [(16,3,224,224)]
    attrs = {
        "precs": "8,8,8",
        "sbs": "8,8,8",
    }
    utils.test_nd(
        shps, attrs, forward_right_shift_channel,
        forward_slice_right_shift)

    # case 2
    shps = [(16,128,32,32)]
    attrs = {
        "precs": "8,7,8,8",
        "sbs": "8,8,8,8",
    }
    utils.test_nd(
        shps, attrs, forward_right_shift_channel,
        forward_slice_right_shift)

    # case 2
    shps = [(2,4,1,4)]
    attrs = {
        "precs": "8,7,8,8",
        "sbs": "8,8,8,8",
    }
    utils.test_nd(
        shps, attrs, forward_right_shift_channel,
        forward_slice_right_shift, th=1e-6)
