import numpy as np
import mxnet as mx
from mxnet import ndarray as nd
import slice_utils as utils

def fwd_conv2d(*data, **attrs):
    assert len(data) == 2
    out = nd.Convolution(*data, **attrs)
    return out

def fwd_conv2d_groupwise(*data, **attrs):
    assert len(data) == 2
    num_group = attrs['num_group']
    assert num_group == 1, \
        "currently only support num_group = 1, provided {}".format(num_group)
    X, W = data
    # reshape weight
    OC, IC = W.shape[:2]
    W = nd.transpose(W, axes=(1,0,2,3))
    rshp = (OC*IC,1,) + W.shape[2:]
    W = W.reshape(rshp)
    # convert to groupwise conv
    attrs['num_group'] = IC
    attrs['num_filter'] = IC * OC
    out = nd.Convolution(X, W, **attrs)
    # sum axis
    YH, YW = out.shape[-2:]
    N = X.shape[0]
    rshp = (N, IC, OC, YH, YW)
    out = out.reshape(rshp)
    out = nd.sum(out, axis=1)
    return out

if __name__ == '__main__':
    shps = [(16, 32, 28, 28), (4, 32, 1, 1)]
    attrs = {
        'no_bias': True,
        'num_group': 1,
        'kernel': (1,1),
        'num_filter': 4,
    }
    utils.test_nd(shps, attrs, fwd_conv2d, fwd_conv2d_groupwise)
    shps = [(1, 3, 225, 225), (8, 3, 3, 3)]
    attrs = {
        'no_bias': True,
        'num_group': 1,
        'kernel': (3,3),
        'num_filter': 8,
    }
    utils.test_nd(shps, attrs, fwd_conv2d, fwd_conv2d_groupwise)
