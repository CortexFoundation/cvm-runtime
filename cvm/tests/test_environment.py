import mxnet as mx
import numpy as np
import logging
import time

logger = logging.getLogger()
logger.setLevel(logging.INFO)
ctx = mx.gpu(0)
# ctx = mx.cpu()

if __name__ == '__main__':
    # mx.nd.waitall()
    a = mx.nd.zeros((3, 1, 5, 5), ctx)
    time.sleep(5)
    print('success')

