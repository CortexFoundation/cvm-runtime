import numpy as np
import logging
from mxnet import ndarray as nd

def check_valid(A, B, tol=1e-6, logger=logging.getLogger("mrt.test.channel")):
    assert A.shape == B.shape, \
        "invalid shape, shape of A: {}, shape of B: {}".format(
        A.shape, B.shape)
    shp = A.shape
    Xs = []
    for X in [A, B]:
        assert isinstance(X, nd.NDArray), \
            "invalid type, type of X: %s".format(type(X))
        X = X.asnumpy()
        X = np.ndarray.flatten(X)
        X = X.tolist()
        assert isinstance(X, list), "invalid type, type of X: {}".format(type(X))
        Xs.append(X)
    A, B = Xs
    assert len(A) == len(B), \
        "invalid length, length of A: {}, length of B: {}".format(
        len(A), len(B))
    for i in range(len(A)):
        diff = abs(A[i] - B[i])
        assert diff <= tol, \
            "invalid tolerance, tol: {}, diff: {}".format(tol, diff)
    logger.info("check valid, shape of data: {}".format(shp))

def generate_data_int(shp, th=2**8):
    X = np.random.random(shp)*th*2 - th
    X = X.round()
    X = nd.array(X)
    return X

def test_nd(
    shps, attrs, fwd_func, cmp_func, tol=1e-6, th=2**8):
    X = [generate_data_int(shp, th=th) for shp in shps]
    out1 = fwd_func(*X, **attrs)
    out2 = cmp_func(*X, **attrs)
    check_valid(out1, out2, tol=tol)


class TestCase(object):
    def __init__(self, shps, attrs):
        self.shps = shps
        self.attrs = attrs
    def test(self, fwd_func, cmp_func, tol=1e-6):
        test_nd(self.shps, self.attrs, fwd_func, cmp_func, tol=tol)
