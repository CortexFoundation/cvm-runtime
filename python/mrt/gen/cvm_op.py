import mxnet as mx
from mxnet import ndarray as nd

import numpy as np


class RightShiftChannel(mx.operator.CustomOp):
    def __init__(self, precs, sbs, num_channel, inc, **kwargs):
        self.inc = eval(inc)
        self.num_channel = eval(num_channel)
        precs = [eval(prec) for prec in precs.split(',')]
        sbs = [eval(sb) for sb in sbs.split(',')]
        assert len(precs) == len(sbs), \
            "invalid length, length of precs: {}, length of sbs: {}".format( \
            (precs, sbs))
        assert self.num_channel % self.inc == 0, \
            "the number of channels: {} must be divisible by inc: {}".format( \
            (self.num_channel, self.inc))
        assert self.num_channel == self.inc * len(precs), \
            "the multiplication of the length of precs: {} and inc: {} " + \
            "must be equal to the number of channels: {}".format( \
            (len(precs), self.inc, self.num_channel))
        clips = [2**(int(prec)-1)-1 for prec in precs]
        self.mins = [int(-clip) for clip in clips]
        self.maxs = [int(clip) for clip in clips]
        self.sbs = [int(sb) for sb in sbs]
        assert all([sb > 0 for sb in self.sbs])

    def forward(self, is_train, req, in_data, out_data, aux):
        assert is_train == False
        X = in_data[0]
        amins, amaxs = self.mins, self.maxs
        out = X.round()
        for i in range(0, self.num_channel, self.inc):
            j = i+self.inc
            k = i // self.inc
            if self.sbs[k] > 1:
                out[:,i:j] = out[:,i:j] / (2 ** (self.sbs[k]-1))
                out[:,i:j] = out[:,i:j].floor()
            out[:,i:j] = out[:,i:j] + 1
            out[:,i:j] = out[:,i:j] / 2
            out[:,i:j] = out[:,i:j].floor()
            out[:,i:j] = out[:,i:j].clip(a_min=amins[k], a_max=amaxs[k])
        self.assign(out_data[0], req[0], out)

    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        assert False


class Conv2DChannel(mx.operator.CustomOp):
    def __init__(
        self, dilate, kernel, layout, no_bias,
        num_filter, num_group, pad, stride, **kwargs):
        pass

    def forward(self, is_train, req, in_data, out_data, aux):
        assert is_train == False

    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        assert False


@mx.operator.register("cvm_right_shift_channel")
class RightShiftChannelProp(mx.operator.CustomOpProp):
    def __init__(self, precs, sbs, num_channel, inc):
        self.precs = precs
        self.sbs = sbs
        self.num_channel = num_channel
        self.inc = inc
        super(RightShiftChannelProp, self).__init__(need_top_grad=True)
    def list_arguments(self):
        return ['data']
    def list_outputs(self):
        return ['output']
    def infer_shape(self, in_shape):
        X_shape = in_shape[0]
        out_shape = in_shape[0]
        return [X_shape], [out_shape], []
    def infer_type(self, in_type):
        X_type = in_type[0]
        return [X_type], [X_type], []
    def create_operator(self, ctx, shapes, dtypes):
        return RightShiftChannel(
            self.precs, self.sbs, self.num_channel, self.inc)


@mx.operator.register("cvm_conv2d_channel")
class Conv2DChannelProp(mx.operator.CustomOpProp):
    def __init__(
        self, dilate, kernel, layout, num_filter, num_group, pad, stride):
        self.dilate = dilate
        self.kernel = kernel
        self.layout = layout
        self.num_filter = num_filter
        self.num_group = num_group
        self.pad = pad
        self.stride = stride
        super(Conv2DChannelProp, self).__init__(need_top_grad=True)
    def list_arguments(self):
        return ['data', 'weight']
    def list_outputs(self, in_shape):
        return ['output']
    def infer_shape(self, in_shape):
        # verify layout
        assert self.layout == "NCHW", \
            "cvm_conv2d_channel only supports layout: NCHW vs {}".format(
            self.layout)
        # verify shapes
        assert len(in_shape) == 2, "invalid in_shape: {}".format(in_shape)
        X_shape, W_shape = in_shape
        assert len(X_shape) == 4, \
            "input data should be 4d: {}".format(X_shape)
        for attr, name in [
            (self.kernel, "kernel"), (self.pad, "pad"),
            (self.strides, "strides"), (self.dilate, "dilate"),
        ]:
            attr = eval(attr)
            assert len(attr) == 2, \
                "invalid attr, name: {}, value: {}".format((name, attr))
            verify_attr_range(attr[0], name)
            verify_attr_range(attr[1], name)
        assert len(W_shape) == 4, \
            "input weight should be 4d: {}".format(W_shape)
        # verify num_group
        num_group = eval(self.num_group)
        assert num_group > 0 and \
            X_shape[1] % num_group == 0 and W_shape[0] % num_group == 0, \
            "invalid num_group: {}, X_shape: {}, W_shape: {}".format( \
            (num_group, X_shape, W_shape))
        # verify kernel shape
        KH, KW = eval(self.kernel)
        assert KH == W_shape[2] and KW == W_shape[3], \
            "invalid kernel attr, kernel: {}, W_shape: {}".format( \
            (self.kernel, W_shape))
        DH, DW = eval(self.dilate)
        DH_size = 1 + (KH-1)*DH
        DW_size = 1 + (KW-1)*DW
        PH, PW = eval(self.pad)
        assert DH_size < H+2*PH and DW_size < W+2*PW, \
            "invalid kernel attr, kernel: {}, pad: {}, dilate: {}".format( \
            (self.kernel, self.pad, self.dilate))
        # output shape
        # TODO(archRev): out_shape[1] when num_group > 1
        out_shape = [X_shape[0], W_shape[0], 0, 0]
        SH, SW = eval(self.stride)
        H, W = in_shape[2:]
        if H != 0:
            out_shape[2] = (H+PH*2-DH_size) // SH + 1
        if W != 0:
            out_shape[3] = (W+PW*2-DW_size) // SW + 1
        out_shape = tuple(out_shape)
        return [X_shape, W_shape], [out_shape], []
    def infer_type(self, in_type):
        X_type = in_type[0]
        W_type = in_type[1]
        return [X_type, B_type], [X_type], []
    def create_operator(self, ctx, shapes, dtypes):
        return Conv2DChannel(
            self.dilate, self.kernel ,self.layout, self.num_filter,
            self.num_group, self.pad, self.stride)

ATTR_MIN_VALIE = 0
ATTR_MAX_VALUE = 4096

def verify_attr_range(val, name, minv=ATTR_MIN_VALUE, maxv=ATTR_MAX_VALUE):
    assert minv <= val <= maxv, \
        "val: {} not in valid range, name: {}, minv: {}, maxv: {}".format(
        (val, name, minv, maxv))
