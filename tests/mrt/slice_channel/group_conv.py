import numpy as np

import mxnet as mx
from mxnet import ndarray as nd

def std(X, W, attr):
    return nd.Convolution(X, W, **attr)

def sc(X, W, attr):
    xshp, wshp = X.shape, W.shape
    C, OC, IC = xshp[1], wshp[0], wshp[1]
    assert C >= IC and C % IC == 0 and C // IC == eval(attr['num_group'])
    num_group = C // IC
    assert num_group == eval(attr['num_group']) and \
        OC >= num_group and OC % num_group == 0
    xs = sym_slice(X, 1, 1)
    ws = kernel_slice_2d(W)
    OPG = OC // num_group
    nattr = attr.copy()
    nattr['num_group'] = '1'
    nattr['num_filter'] = '1'
    nodes = []
    for o in range(OC):
        nnodes = []
        j = int(o/OPG)*IC
        for i in range(IC):
            xoi, woi = xs[i+j], ws[o][i]
            yoi = nd.Convolution(xoi, woi, **nattr)
            nnodes.append(yoi)
        zi = nd.add_n(*nnodes)
        nodes.append(zi)
    return nd.concat(*nodes, dim=1)

def sym_slice(X, ichannel, step):
    shp = X.shape
    ndims = len(shp)
    nodes = []
    rchannel = ndims-ichannel-1
    for i in range(0, shp[ichannel], step):
        Xi = nd.slice(
            X, begin=(None,)*ichannel+(i,)+(None,)*rchannel,
            end=(None,)*ichannel+(i+step,)+(None,)*rchannel)
        nodes.append(Xi)
    return nodes

def kernel_slice_2d(W):
    shp = W.shape
    OC, IC = shp[:2]
    nodes = []
    for o in range(OC):
        Wo = nd.slice(W, begin=(o,None,None,None), end=(o+1,None,None,None))
        nnodes = []
        for i in range(IC):
            Woi = nd.slice(Wo, begin=(None,i,None,None), end=(None,i+1,None,None))
            nnodes.append(Woi)
        nodes.append(nnodes[:])
    return nodes

def get_norm(data):
    data = data.asnumpy()
    ndims = np.product(data.shape)
    data = np.reshape(data, (ndims,))
    norm = np.linalg.norm(data)
    return norm, data

def test(xshp, wshp, attr, ichannel, step, ctx=mx.cpu()):
    X = generate(xshp, ctx=ctx)
    W = generate(wshp, ctx=ctx)
    a = std(X, W, attr)
    b = sc(X, W, attr)

    assert a.shape == b.shape, (a.shape, b.shape)
    an, ap = get_norm(a)
    bn, bp = get_norm(b)
    rn = np.linalg.norm(ap-bp)
    print(an, bn, rn)

    print(a.abs().max().asscalar(), b.abs().max().asscalar())

def generate(shp, ctx=mx.cpu()):
    assert len(shp) == 4 # NCHW
    data_np = np.random.uniform(size=shp)
    return nd.array(data_np, ctx=ctx)

def test1():
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

def test2():
    xshp = (16, 16, 56, 56)
    wshp = (32, 16, 1, 1)
    attr = {
        'layout': 'NCHW',
        'num_filter': '32',
        'dilate': '(1, 1)',
        'num_group': '1',
        'stride': '(1, 1)',
        'no_bias': 'True',
        'kernel': '[1, 1]',
    }
    ichannel = 1
    step = 2
    test(xshp, wshp, attr, ichannel, step, ctx=mx.gpu())

def test3():
    xshp = (1, 8, 114, 114)
    wshp = (8, 1, 3, 3)
    attr = {
        'layout': 'NCHW',
        'num_filter': '8',
        'dilate': '(1, 1)',
        'num_group': '8',
        'stride': '(1, 1)',
        'no_bias': 'True',
        'kernel': '[3, 3]',
    }
    ichannel = 1
    step = 2
    test(xshp, wshp, attr, ichannel, step, ctx=mx.gpu())

if __name__ == '__main__':
    # test1()
    # test2()
    test3()
