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
    def __init__(self, **kwargs):
        pass

    def forward(self, is_train, req, in_data, out_data, aux):
        pass

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
    pass
