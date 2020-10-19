import mxnet as mx
import numpy as np
from mxnet import ndarray as nd
from mrt.gen import cvm_op
from mrt import cvm_op

def test_symbol(precs, sbs, num_channel, inc):
    X = mx.sym.var("var")
    op = mx.sym.Custom(
        X, precs=precs, sbs=sbs, num_channel=num_channel, inc=inc, name="out",
        op_type="cvm_right_shift_channel")

def check_valid(A, B, tol=1e-6):
    assert A.shape == B.shape, \
        "invalid shape, shape of A: {}, shape of B: {}".format(
        (A.shape, B.shape))
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
        (len(A), len(B)))
    for i in range(len(A)):
        diff = abs(A[i] - B[i])
        assert diff <= tol, "invalid tolerance, tol: {}, diff: {}".format(
        (tol, diff))
    print("check valid")

def forward_right_shift_channel(X, precs, sbs, inc):
    num_channel = X.shape[1]
    out = nd.Custom(
        X, precs=precs, sbs=sbs, num_channel=num_channel, inc=inc,
        op_type="cvm_right_shift_channel")
    return out

def forward_slice_right_shift(X, precs, sbs, inc):
    precs = [eval(prec) for prec in precs.split(',')]
    sbs = [eval(sb) for sb in sbs.split(',')]
    xshp = X.shape
    ndim = len(xshp)
    assert xshp[1] % len(precs) == 0, \
        "division error, xshp[0]: {}, len(precs): {}".format( \
        (xshp[1], len(precs)))
    assert inc == xshp[1] // len(precs), \
        "invalid inc, inc: {}, xshp[1]//len(precs): {}".format( \
        (inc, xshp[1]//len(precs)))
    nodes = []
    for i in range(0, xshp[1], inc):
        begin = (None,i,) + (None,)*(ndim-2)
        end = (None,i+inc) + (None,)*(ndim-2)
        node = nd.slice(X, begin=begin, end=end)
        j = i // inc
        node = nd.Custom(
            node, precision=precs[j], shift_bit=sbs[j],
            op_type="cvm_right_shift")
        nodes.append(node)
    out = nd.concat(*nodes, dim=1)
    return out

def generate_data_int(shp, th=2**24):
    X = np.random.random(shp)*th*2 - th
    X = X.round()
    X = nd.array(X)
    return X

def test_nd(shp, precs, sbs, inc, tol=1e-6):
    X = generate_data_int(shp)
    out1 = forward_right_shift_channel(X, precs=precs, sbs=sbs, inc=inc)
    out2 = forward_slice_right_shift(X, precs=precs, sbs=sbs, inc=inc)
    check_valid(out1, out2, tol=tol)

if __name__ == "__main__":
    test_symbol(precs="8,8,8", sbs="8,8,8", num_channel=3, inc=1)
    test_nd(shp=(16,3,224,224), precs="8,8,8", sbs="8,8,8", inc=1)
    test_nd(shp=(16,128,32,32), precs="8,7,8,8", sbs="8,8,8,8", inc=32)
