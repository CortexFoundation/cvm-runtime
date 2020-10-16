import mxnet as mx
import numpy as np
from mxnet import ndarray as nd
from mrt.gen import cvm_op

def test_symbol():
    X = mx.sym.var("var")
    precs = "8,8,8"
    sbs = "8,8,8"
    op = mx.sym.Custom(
        X, precs=precs, sbs=sbs, name="out",
        op_type="cvm_right_shift_channel")

def test_nd():
    X = np.random.rand(16, 3, 224, 224)
    X = nd.array(X)
    precs = "8,8,8"
    sbs = "8,8,8"
    out = nd.Custom(
        X, precs=precs, sbs=sbs,
        op_type="cvm_right_shift_channel")
    print(type(out))

if __name__ == "__main__":
    test_symbol()
    test_nd()
