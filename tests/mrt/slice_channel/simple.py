import mxnet as mx
import numpy as np
import slice_utils as utils
from mxnet import ndarray as nd

def test_ord1():
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

def forward_conv2gconv_input():
    pass

def backward_gconv2conv_input():
    pass

def forward_conv2gconv_output():
    pass

def backward_gconv2conv_output():
    pass

def fuse_transform_output():
    pass

def test_gen(
    step=2,
    NG=1, OPG=4, IPG=8,
    N=16, H=28, W=28, KH=1, KW=1,
    PH=0, PW=0, SH=1, SW=1, DH=1, DW=1,
    tol=1e-5, th=1e7):
    # for temporory
    assert PH == 0 and PW == 0 and SH == 1 and SW == 1 \
        and DH == 1 and DW == 1 and NG == 1
    YH = (H+2*PH-DH*(KH-1)-1)//SH + 1
    YW = (W+2*PW-DW*(KW-1)-1)//SW + 1
    # for reference
    C = IPG * NG
    O = OPG * NG
    X = nd.array(np.random.random((N,C,H,W)))
    W = nd.array(np.random.random((O,IPG,KH,KW)))
    attrs = {
        'no_bias': True,
        'num_group': NG,
        'kernel': (KH, KW),
        'num_filter': O,
        'stride': (SH, SW),
        'pad': (PH, PW),
        'layout': 'NCHW',
        'dilate': (DH, DW),
    }
    Y = nd.Convolution(X, W, **attrs)

    # transpose and reshape W
    assert IPG % step == 0, "invalid step: {}".format(step)
    # (O,IPG,KH,KW) --transpose--> (IPG,O,KH,KW)
    W1 = nd.transpose(W, axes=(1,0,2,3))
    NIPG = IPG // step
    # (IPG,O,KH,KW) --reshape--> (NIPG,step,O,KH,KW)
    W1 = nd.reshape(W1, shape=(NIPG,step,O,KH,KW))
    # (NIPG,step,O,KH,KW) --transpose--> (NIPG,O,step,KH,KW)
    W1 = nd.transpose(W1, axes=(0,2,1,3,4))
    NO = NIPG * O
    # (NIPG,O,step,KH,KW) --reshape--> (NO,step,KH,KW)
    W1 = nd.reshape(W1, shape=(NO,step,KH,KW))
    NNG = C // step
    nattrs = {
        'no_bias': True,
        'num_group': NNG,
        'kernel': (KH, KW),
        'num_filter': NO,
        'stride': (SH, SW),
        'pad': (PH, PW),
        'layout': 'NCHW',
        'dilate': (DH, DW),
    }
    Y1 = nd.Convolution(X, W1, **nattrs)
    # (N,NO,YH,YW) --reshape--> (N,NIPG,O,YH,YW)
    Y1 = nd.reshape(Y1, shape=(N,NIPG,O,YH,YW))
    # (N,NIPG,O,YH,YW) --sum--> (N,O,YH,YW)
    Y1 = nd.sum(Y1, axis=1)
    utils.check_valid(Y, Y1, tol=tol)

if __name__ == '__main__':
    test_ord1()
    test_gen(
        step=4, IPG=512, OPG=512, H=14, W=14, tol=1e-4)
