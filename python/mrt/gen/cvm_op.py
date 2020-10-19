import mxnet as mx
from mxnet import ndarray as nd

import numpy as np


class RightShiftChannel(mx.operator.CustomOp):
    def __init__(self, precs, sbs, **kwargs):
        super(RightShiftChannel, self).__init__(**kwargs)
        precs = [eval(prec) for prec in precs.split(',')]
        sbs = [eval(sb) for sb in sbs.split(',')]
        assert len(precs) == len(sbs), \
            "invalid length, length of precs: {}, length of sbs: {}".format( \
            (precs, sbs))
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
        num_channel = X.shape[1]
        assert num_channel % len(amins) == 0, \
            "num_channel: {} is not divisible by len(amins): {}".format( \
            (num_channel, len(amins)))
        inc = num_channel // len(amins)
        for i in range(0, num_channel, inc):
            j = i + inc
            k = i // inc
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
        self, dilate, kernel, layout, no_bias, num_filter,
        num_group, pad, stride, inc, **kwargs):
        super(RightShiftChannel, self).__init__(**kwargs)
        self.dilate = eval(dilate)
        self.kernel = eval(kernel)
        self.layout = layout
        self.no_boas = eval(no_bias)
        self.num_filter = eval(num_filter)
        self.num_group = eval(num_group)
        self.pad = eval(pad)
        self.stride = eval(stride)
        self.inc = eval(inc)

    def forward(self, is_train, req, in_data, out_data, aux):
        assert is_train == False
        n_batch, in_channels, x_h, x_W = in_data[0].shape
        oshp = out_data[0].shape
        print(oshp)
        # TODO(archRev): realize

    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        assert False


@mx.operator.register("cvm_right_shift_channel")
class RightShiftChannelProp(mx.operator.CustomOpProp):
    def __init__(self, precs, sbs):
        self.precs = precs
        self.sbs = sbs
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
        return RightShiftChannel(self.precs, self.sbs)


@mx.operator.register("cvm_conv2d_channel")
class Conv2DChannelProp(mx.operator.CustomOpProp):
    def __init__(
        self, dilate, kernel, layout, no_bias,
        num_filter, num_group, pad, stride, inc):
        self.dilate = dilate
        self.kernel = kernel
        self.layout = layout
        self.no_bias = no_bias
        self.num_filter = num_filter
        self.num_group = num_group
        self.pad = pad
        self.stride = stride
        self.inc = inc
        super(Conv2DChannelProp, self).__init__(need_top_grad=True)
    def list_arguments(self):
        no_bias = eval(self.no_bias)
        return ['data', 'weight'] if no_bias else ['data', 'weight', 'bias']
    def list_outputs(self, in_shape):
        return ['output']
    def infer_shape(self, in_shape):
        # verify layout
        assert self.layout == "NCHW", \
            "cvm_conv2d_channel only supports layout: NCHW vs {}".format(
            self.layout)
        # verify shapes
        no_bias = eval(no_bias)
        if no_bias == True:
            assert len(in_shape) == 2, \
                "invalid in_shape: {}".format(in_shape)
        else:
            assert len(in_shape) == 3, \
                "invalid in_shape: {}".format(in_shape)
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
        # num_group
        # TODO(archRev): case when num_group > 1
        num_group = eval(self.num_group)
        assert num_group == 1, "invalid attr num_group: {}".format(num_group)
        num_channel = X_shape[1]
        assert W_shape[1] == X_shape[1], "input channel size: {} and " + \
            "weight channel size: {} not consistant".format( \
            (X_shape[1], W_shape[1]))
        inc = eval(self.inc)
        assert num_channel % inc == 0, \
            "the number of channels: {} must be divisible by inc: {}".format( \
            (num_channel, inc))
        # output shape
        out_shape = [X_shape[0], W_shape[0], num_channel//inc, 0, 0]
        SH, SW = eval(self.stride)
        H, W = in_shape[2:]
        if H != 0:
            out_shape[3] = (H+PH*2-DH_size) // SH + 1
        if W != 0:
            out_shape[4] = (W+PW*2-DW_size) // SW + 1
        out_shape = tuple(out_shape)
        return [X_shape, W_shape], [out_shape], []
    def infer_type(self, in_type):
        X_type = in_type[0]
        W_type = in_type[1]
        return [X_type, B_type], [X_type], []
    def create_operator(self, ctx, shapes, dtypes):
        return Conv2DChannel(
            self.dilate, self.kernel ,self.layout, self.no_bias,
            self.num_filter, self.num_group, self.pad, self.stride, self.inc)

ATTR_MIN_VALUE = 0
ATTR_MAX_VALUE = 4096

def verify_attr_range(val, name, minv=ATTR_MIN_VALUE, maxv=ATTR_MAX_VALUE):
    assert minv <= val <= maxv, \
        "val: {} not in valid range, name: {}, minv: {}, maxv: {}".format(
        (val, name, minv, maxv))
