import numpy as np

import mxnet as mx
from mxnet import ndarray as nd

def get_norm(data):
    data = data.asnumpy()
    ndims = np.product(data.shape)
    data = np.reshape(data, (ndims,))
    norm = np.linalg.norm(data)
    return norm, data

def test():
    x_np = np.random.uniform(size=(1,1,1,3))
    x = nd.array(x_np)
    w_np = np.random.uniform(size=(3,1,3))
    w = nd.array(w_np)
    print(x.shape, w.shape)

    y = nd.broadcast_like(x, w, lhs_axes=(1), rhs_axes=(1))
    print(y)

    z = nd.tile(x, reps=(1,1,1,1))
    print(z)

    assert z.shape == y.shape
    zn, zp = get_norm(z)
    yn, yp = get_norm(y)
    rn = np.linalg.norm(zp-yp)
    print(zn, yn, rn)


def verify_broadcast_like(xshp, wshp, lhs_axes, rhs_axes):
    x_np = np.random.uniform(size=xshp)
    w_np = np.random.uniform(size=wshp)
    x = nd.array(x_np)
    w = nd.array(w_np)

    # org op
    y = nd.broadcast_like(x, w, lhs_axes, rhs_axes)

    # rewrite op
    lndims = len(lhs_axes)
    rndims = len(rhs_axes)
    assert lndims == rndims

    xndims = len(xshp)
    lhs_axes = tuple([v+xndims if v<0 else v for v in lhs_axes])
    assert all([0<=v<xndims for v in list(lhs_axes)])

    wndims = len(wshp)
    rhs_axes = tuple([v+wndims if v<0 else v for v in rhs_axes])
    assert all([0<=v<wndims for v in list(rhs_axes)])

    assert all([xshp[lhs_axes[i]]==1 for i in range(lndims)])

    cnts = {v: wshp[rhs_axes[i]] for i, v in enumerate(lhs_axes)}
    reps = tuple([cnts[v] if v in lhs_axes else 1 for v in range(xndims)])
    z = nd.tile(x, reps=reps)

    # compare
    assert z.shape == y.shape
    zn, zp = get_norm(z)
    yn, yp = get_norm(y)
    rn = np.linalg.norm(zp-yp)
    print(zn, yn, rn)

if __name__ == '__main__':
    verify_broadcast_like((3,2,1,3,1), (2,8,1,9), (2,4), (3,-3))
    # test()
