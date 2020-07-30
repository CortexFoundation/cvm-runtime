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
    x = nd.array([[[1.1,2.2]], [[3.3,4.4]]])
    gamma = nd.array([1.5])
    beta = nd.array([0.5])
    # gamma = 1.5
    # beta = 0.5
    eps = 0.00100000005

    shp = x.shape
    n = np.product(shp[2:])
    mean = nd.sum(x, axis=2, keepdims=True) / n
    dev = x - mean
    var = nd.sum(dev*dev, axis=2, keepdims=True) / n
    std = nd.sqrt(var) + eps
    frac = dev / std
    out = frac*gamma + beta
    print(out)

def verify_instance_norm_rewrite(shp, eps):
    # assert len(shp) == 4 # NCHW
    vshp = (shp[1],)
    data_np = np.random.uniform(size=shp)
    gamma_np = np.random.uniform(size=vshp)
    beta_np = np.random.uniform(size=vshp)
    x = nd.array(data_np)
    gamma = nd.array(gamma_np)
    beta = nd.array(beta_np)

    # org op
    y = nd.InstanceNorm(x, gamma=gamma, beta=beta, eps=eps)

    # rewrite op
    axis = [i for i in range(len(shp)) if i != 1]
    for i in axis:
        gamma = nd.expand_dims(gamma, axis=i)
        beta = nd.expand_dims(beta, axis=i)

    n = np.product(shp[2:])
    mean = nd.sum(x, axis=axis, keepdims=True) / n
    dev = x - mean
    var = nd.sum(dev*dev, axis=axis, keepdims=True) / n
    std = nd.sqrt(var) + eps
    frac = dev / std
    z = frac*gamma + beta

    # compare
    assert z.shape == y.shape
    zn, zp = get_norm(z)
    yn, yp = get_norm(y)
    rn = np.linalg.norm(zp-yp)
    print(zn, yn, rn)

if __name__ == '__main__':
    verify_instance_norm_rewrite((1, 64, 512, 512), 0.00100000005)
    verify_instance_norm_rewrite((1, 64, 512), 0.00100000005)
    # test()
