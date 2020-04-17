import mxnet as mx
from mxnet import ndarray as nd
import numpy as np
import logging
import time
import os

def test_nd_op(ctx=mx.gpu()):
    nd.waitall()
    a = nd.zeros((3, 1, 5, 5), ctx)
    time.sleep(5)
    print('success')

def test_nd_save(ctx=mx.gpu()):
    a = nd.array([1,2], ctx)
    nd.save(os.path.expanduser("~/data/test_nd_save_data"), a)
    print('success')

if __name__ == '__main__':
    # test_nd_op()
    test_nd_save(ctx=mx.cpu())

