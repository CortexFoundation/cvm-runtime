import numpy as np

import mxnet as mx
from mxnet import ndarray as nd

def std(X, W, attr):
    return nd.Convolution(X, W, **attr)

def sc(X, W, attr, ichannel, step):
    xshp = X.shape
    xs = sym_slice(X, ichannel, step)
    ws = sym_slice(W, ichannel, step)
    nodes = []
    for i in range(0, xshp[1], step):
        yi = nd.Convolution(xs[i], ws[i], **attr)
        nodes.append(yi)
    return nd.add_n(*nodes)

def sym_slice(op, ichannel, step):
    shp = op.shape
    ndims = len(shp)
    nodes = []
    rchannel = ndims-ichannel-1
    for i in range(0, shp[ichannel], step):
        opi = nd.slice(
            op, begin=(None,)*ichannel+(i,)+(None,)*rchannel,
            end=(None,)*ichannel+(i+step,)+(None,)*rchannel)
        nodes.append(opi)
    return nodes

def get_norm(data):
    data = data.asnumpy()
    ndims = np.product(data.shape)
    data = np.reshape(data, (ndims,))
    norm = np.linalg.norm(data)
    return norm, data

def test(xshp, wshp, attr, ichannel, step):
    X = generate(xshp)
    W = generate(wshp)
    a = std(X, W, attr)
    b = sc(X, W, attr, ichannel, step)

    assert a.shape == b.shape
    an, ap = get_norm(a)
    bn, bp = get_norm(b)
    rn = np.linalg.norm(ap-bp)
    print(an, bn, rn)

    print(a.abs().max().asscalar(), b.abs().max().asscalar())

def generate(shp):
    assert len(shp) == 4 # NCHW
    data_np = np.random.uniform(size=shp)
    return nd.array(data_np)

if __name__ == '__main__':
    xshp = (16,3,228,228)
    wshp = (64,3,11,11)
    attr = {
        'layout': 'NCHW',
        'num_filter': '64',
        'dilate': '(1,1)',
        'num_group': '1',
        'stride': '(4,4)',
        'no_bias': 'True',
        'kernel': '(11,11)',
    }
    ichannel = 1
    step = 1
    test(xshp, wshp, attr, ichannel, step)
