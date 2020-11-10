import mxnet as mx
import numpy as np
import slice_utils as utils
from mxnet import ndarray as nd

if __name__ == '__main__':
    a = nd.array(np.random.random((16,32,28,28)))
    # a = nd.array([-1,2]).reshape((1,2,1,1))
    # print("x", a)
    b = nd.array(np.random.random((4,32,1,1)))
    # b = nd.array([3,5,7,9]).reshape((2,2,1,1))
    # print("w", b)
    c = nd.Convolution(
        a, b, no_bias=True, num_group=1, kernel=(1,1), num_filter=4)
    # print("conv", c)
    # import group_conv as gconv
    # attrs = {
        # 'num_group': '1',
        # 'kernel': '(1,1)',
        # 'no_bias': 'True',
        # 'num_filter': '2',
    # }
    # nodes = gconv.sc(a, b, attrs)
    # print("nodes", nodes)

    b1 = nd.transpose(b, axes=(1,0,2,3))
    b1 = b1.reshape((128,1,1,1))
    c1 = nd.Convolution(
        a, b1, no_bias=True, num_group=32, kernel=(1,1), num_filter=128)
    c1 = c1.reshape((16,32,4,28,28))
    # print("gconv", c1)
    c1 = nd.sum(c1, axis=1)
    # print("sum", c1)
    utils.check_valid(c, c1, tol=1e-5)

