import numpy as np

import mxnet as mx
from mxnet import ndarray as nd

def get_norm(data):
    data = data.asnumpy()
    ndims = np.product(data.shape)
    data = np.reshape(data, (ndims,))
    norm = np.linalg.norm(data)
    return norm, data

def verify_reshape_like(
    xshp, wshp, lhs_begin, lhs_end, rhs_begin, rhs_end):
    x_np = np.random.uniform(size=xshp)
    w_np = np.random.uniform(size=wshp)
    x = nd.array(x_np)
    w = nd.array(w_np)

    # org op
    y = nd.reshape_like(
        x, w, lhs_begin=lhs_begin, lhs_end=lhs_end,
        rhs_begin=rhs_begin, rhs_end=rhs_end)

    # rewrite op
    xndims = len(xshp)
    lhs_begin = lhs_begin+xndims if lhs_begin < 0 else lhs_begin
    lhs_end = lhs_end+xndims if lhs_end< 0 else lhs_end
    assert 0 <= lhs_begin < lhs_end <= xndims

    wndims = len(wshp)
    rhs_begin = rhs_begin+wndims if rhs_begin < 0 else rhs_begin
    rhs_end = rhs_end+wndims if rhs_end< 0 else rhs_end
    assert 0 <= rhs_begin < rhs_end <= wndims

    rshp = tuple(xshp[:lhs_begin] + \
        wshp[rhs_begin:rhs_end] + xshp[lhs_end:])
    z = nd.reshape(x, rshp)

    # compare
    assert z.shape == y.shape
    zn, zp = get_norm(z)
    yn, yp = get_norm(y)
    rn = np.linalg.norm(zp-yp)
    print(zn, yn, rn)

if __name__ == '__main__':
    verify_reshape_like((30,12), (4,2,2,3), -1, 2, 1, 4)
    # test()
