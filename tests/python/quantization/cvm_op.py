import mxnet as mx
from mxnet import ndarray as nd

import numpy as np

class Clip(mx.operator.CustomOp):
    def __init__(self, precision, **kwargs):
        super(Clip, self).__init__(**kwargs)
        clip = 2 ** (int(precision) - 1) - 1
        self.min = int(-clip)
        self.max = int(clip)

    def forward(self, is_train, req, in_data, out_data, aux):
        assert is_train == False
        X = in_data[0]
        a_min, a_max = self.min, self.max
        out = X.clip(a_min=a_min, a_max=a_max)
        self.assign(out_data[0], req[0], out)

    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        assert False

class LeftShift(mx.operator.CustomOp):
    def __init__(self, precision, shift_bit, **kwargs):
        super(LeftShift, self).__init__(**kwargs)
        clip = 2 ** (int(precision) - 1) - 1
        self.min = int(-clip)
        self.max = int(clip)
        self.sb = int(shift_bit)
        assert self.sb > 0

    def forward(self, is_train, req, in_data, out_data, aux):
        assert is_train == False
        X = in_data[0]
        a_min, a_max = self.min, self.max
        out = X * (2 ** (self.sb))
        out = out.clip(a_min=a_min, a_max=a_max)
        self.assign(out_data[0], req[0], out)

    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        assert False

class RightShift(mx.operator.CustomOp):
    def __init__(self, precision, shift_bit, **kwargs):
        super(RightShift, self).__init__(**kwargs)
        clip = 2 ** (int(precision) - 1) - 1
        self.min = int(-clip)
        self.max = int(clip)
        self.sb = int(shift_bit)
        assert self.sb > 0

    def forward(self, is_train, req, in_data, out_data, aux):
        assert is_train == False
        X = in_data[0]
        a_min, a_max = self.min, self.max
        out = X / (2 ** (self.sb-1))
        out = out.floor()
        out = out + 1
        out = out / 2
        out = out.floor()
        out = out.clip(a_min=a_min, a_max=a_max)
        self.assign(out_data[0], req[0], out)

    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        assert False

@mx.operator.register("cvm_clip")
class ClipProp(mx.operator.CustomOpProp):
    def __init__(self, precision=8, shift_bit=0):
        self.precision= precision
        self.shift_bit = shift_bit
        super(ClipProp, self).__init__(need_top_grad=False)
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
        return Clip(self.precision)

@mx.operator.register("cvm_left_shift")
class LeftShiftProp(mx.operator.CustomOpProp):
    def __init__(self, precision=8, shift_bit=0):
        self.precision= precision
        self.shift_bit = shift_bit
        super(LeftShiftProp, self).__init__(need_top_grad=False)
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
        return LeftShift(self.precision, self.shift_bit)

@mx.operator.register("cvm_right_shift")
class RightShiftProp(mx.operator.CustomOpProp):
    def __init__(self, precision=8, shift_bit=0):
        self.precision= precision
        self.shift_bit = shift_bit
        super(RightShiftProp, self).__init__(need_top_grad=False)
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
        return RightShift(self.precision, self.shift_bit)



