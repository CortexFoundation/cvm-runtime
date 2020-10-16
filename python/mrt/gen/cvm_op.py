import mxnet as mx
from mxnet import ndarray as nd

import numpy as np


class RightShiftChannel(mx.operator.CustomOp):
    def __init__(self, precs, sbs, **kwargs):
        assert len(precs) == len(sbs)
        self.nc = len(precs)
        print(self.nc, "*****", precs)
        clips = [2**(int(prec)-1)-1 for prec in precs.split(",")]
        self.mins = [int(-clip) for clip in clips]
        self.maxs = [int(clip) for clip in clips]
        self.sbs = [int(sb) for sb in sbs.split(",")]
        assert all([sb > 0 for sb in self.sbs])

    def forward(self, is_train, req, in_data, out_data, aux):
        assert is_train == False
        X = in_data[0]
        amins, amaxs = self.mins, self.maxs
        out = X.round()
        for i in range(self.nc):
            if self.sbs[i] > 1:
                out[:,i] = out[:,i] / (2 ** (self.sbs[i]-1))
                out[:,i] = out[:,i].floor()
            out[:,i] = out[:,i] + 1
            out[:,i] = out[:,i] / 2
            out[:,i] = out[:,i].floor()
            out[:,i] = out[:,i].clip(a_min=amins[i], a_max=amaxs[i])
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
    pass
