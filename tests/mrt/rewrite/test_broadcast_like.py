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
    w_np = np.random.uniform(size=(3,1,4))
    w = nd.array(w_np)

    y = nd.broadcast_like(x, w, lhs_axes=(1,1), rhs_axes=(2,0))
    print(y.shape)

    z = nd.tile(x, reps=(1,3,1,1))
    print(z.shape)

    assert z.shape == y.shape
    zn, zp = get_norm(z)
    yn, yp = get_norm(y)
    rn = np.linalg.norm(zp-yp)
    print(zn, yn, rn)


def verify_broadcast_like_fixed(xshp, wshp, lhs_axes, rhs_axes):
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

def _broadcast_like(x, w):
    zero = nd.zeros(shape=(1,))
    one = nd.ones(shape=(1,))
    nw = nd.broadcast_add(nd.broadcast_mul(w, zero), one)
    z = nd.broadcast_mul(x, nw)
    return z

def verify_broadcast_like_dynamic(xshp, wshp, lhs_axes, rhs_axes):
    x_np = np.random.uniform(size=xshp)
    w_np = np.random.uniform(size=wshp)
    x = nd.array(x_np)
    w = nd.array(w_np)

    # org op
    y = nd.broadcast_like(x, w,
        lhs_axes=lhs_axes, rhs_axes=rhs_axes)
    print(y.shape)

    # rewrite op
    xndims, wndims = len(xshp), len(wshp)
    if lhs_axes is None or rhs_axes is None:
        assert xndims == wndims and lhs_axes is None \
            and rhs_axes is None
        z = _broadcast_like(x, w)
    else:
        lhs_axes, lndims = list(lhs_axes), len(lhs_axes)
        rhs_axes, rndims = list(rhs_axes), len(rhs_axes)
        assert lndims == rndims > 0

        lhs_axes = tuple([v+xndims if v<0 else v for v in lhs_axes])
        assert all([0<=v<xndims for v in list(lhs_axes)])

        rhs_axes = tuple([v+wndims if v<0 else v for v in rhs_axes])
        assert all([0<=v<wndims for v in list(rhs_axes)])

        assert all([xshp[lhs_axes[i]] == 1 for i in range(lndims)])

        batch_axes = [0]
        flg = all([batch_axis not in rhs_axes \
            for batch_axis in batch_axes])
        if flg:
            cnts = {v: wshp[rhs_axes[i]] \
                for i, v in enumerate(lhs_axes)}
            reps = tuple([cnts[v] if v in lhs_axes else 1 \
                for v in range(xndims)])
            z = nd.tile(x, reps=reps)
        else:
            axis_map = {}
            for i, v in enumerate(lhs_axes):
                axis_map[v] = rhs_axes[i]
            for batch_axis in batch_axes:
                assert sum([1 if v == batch_axis else 0 \
                    for k, v in axis_map.items()]) <= 1, \
                    "multiple broadcast on batch_axis: %s, " + \
                    "which is not support by dynamic shape fusion." % \
                    batch_axis
            assert wndims < 6, \
                "slice can manipulate at most 5d"

            # reduce shape to 1 for non-broadcast dimensions
            begin = tuple([0]*wndims)
            end = tuple([wshp[v] if v in axis_map.values() else 1 \
                for v in range(wndims)])
            w = nd.slice(w, begin=begin, end=end)

            # decompose k1->v, k2->v into k1->v, k2->v2
            # which make axis
            while True:
                vs, flag, paxis_map = set(), True, axis_map
                for pk, pv in paxis_map.items():
                    if pv not in vs:
                        vs.add(pv)
                        continue
                    flag = False
                    axis_map = {k: (v+1 if v>pv or k==pk else v) \
                        for k, v in axis_map.items()}
                    w = nd.expand_dims(w, axis=pv)
                    w = nd.repeat(w, axis=pv, repeats=wshp[pv])
                    wshp = wshp[:pv] + (wshp[pv],) + wshp[pv:]
                    break
                if flag:
                    break
            wndims = len(wshp)

            # trim wndims if not equal to xndims
            v = 0
            while wndims > xndims:
                while v in axis_map.values():
                    v += 1
                w = nd.squeeze(w, axis=v)
                wndims -= 1
                axis_map = {k: (nv-1 if nv > v else nv) \
                    for k, nv in axis_map.items()}
            while wndims < xndims:
                w = nd.expand_dims(w, axis=wndims)
                wndims += 1
            axes = list(range(wndims))
            while True:
                dels = [k for k, v in axis_map.items() if k==v]
                for k in dels:
                    del axis_map[k]
                if not axis_map:
                    break
                keys = list(axis_map.keys())
                k, v = keys[0], axis_map[keys[0]]
                axes[k], axes[v] = axes[v], axes[k]
                for nk in keys:
                    nv = axis_map[nk]
                    if nv == k:
                        axis_map[nk] = v
                    elif nv == v:
                        axis_map[nk] = k
            axes = tuple(axes)
            if axes != tuple(range(wndims)):
                assert wndims < 7, \
                    "slice can manipulate at most 6d"
                w = nd.transpose(w, axes=axes)
            z = _broadcast_like(x, w)
    print(z.shape)

    # compare
    assert z.shape == y.shape
    zn, zp = get_norm(z)
    yn, yp = get_norm(y)
    rn = np.linalg.norm(zp-yp)
    print(zn, yn, rn)

if __name__ == '__main__':
    verify_broadcast_like_fixed(
        (3,2,1,3,1), (2,8,1,9), (2,4), (3,-3))
    verify_broadcast_like_dynamic(
        (7,1,1,5,1,3),
        (8,9,10,1),
        (2,1,4),
        (1,1,0))
    verify_broadcast_like_dynamic(
        (1,4,1,1),
        (10,6,5,9,1),
        (2,2,0,3),
        (2,1,0,1))
    test()
