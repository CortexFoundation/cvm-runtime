""" Customized Op-level realization of MxNet Forward Computing Framework.

    MxNet customized operator property class.

    Only **crucial parts** of the custommized 
    forward operator implementation are elaborated.
"""

import mxnet as mx
from mxnet import ndarray as nd

import numpy as np

class Clip(mx.operator.CustomOp):
    def __init__(self, precision, **kwargs):
        super(Clip, self).__init__(**kwargs)
        clip = 2 ** (int(precision) - 1) - 1
        self.min = float(-clip)
        self.max = float(clip)

    def forward(self, is_train, req, in_data, out_data, aux):
        """ MxNet customized operator forward implementation.

            Clip the input within [-2^prec, 2^prec].

            .. math::
                rnd = round(X)

            where X is the input tensor.

            .. math::
                out = clip(rnd, -2^{prec}, 2^{prec})

            where prec is an operator attribute representing integer bits.
        """
        assert is_train == False
        X = in_data[0]
        a_min, a_max = self.min, self.max
        out = X.round()
        out = out.clip(a_min=a_min, a_max=a_max)
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
        """ MxNet customized operator forward implementation.

            Left shift data for sb bits and clip within [-2^prec, 2^prec].

            .. math::
                rnd = round(X)

            where X is the input tensor.

            .. math::
                val = rnd * 2^{sb}

            .. math::
                out = clip(val1, -2^{prec}, 2^{prec})

            where prec is an operator attribute representing integer bits, sb represents the left shift bits.
        """
        assert is_train == False
        X = in_data[0]
        a_min, a_max = self.min, self.max
        out = X.round()
        out = out * (2 ** (self.sb))
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
        """ MxNet customized operator forward implementation.

            Right shift data for sb bits and clip within [-2^prec, 2^prec].

            .. math::
                rnd = round(X)

            where X is the input tensor.

            .. math::
                val0 = floor(rnd / 2^{sb-1})

            .. math::
                val1 = floot((val0+1) / 2)

            .. math::
                out = clip(val1, -2^{prec}, 2^{prec})

            where prec is an operator attribute representing integer bits, sb represents the right shift bits.
        """
        assert is_train == False
        X = in_data[0]
        a_min, a_max = self.min, self.max
        out = X.round()
        if self.sb > 1:
            out = out / (2 ** (self.sb-1))
            out = out.floor()
        out = out + 1
        out = out / 2
        out = out.floor()
        out = out.clip(a_min=a_min, a_max=a_max)
        self.assign(out_data[0], req[0], out)

    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        assert False

class LUT(mx.operator.CustomOp):
    def __init__(self, in_dim, **kwargs):
        super(LUT, self).__init__(**kwargs)
        self.in_dim = int(in_dim)

    def forward(self, is_train, req, in_data, out_data, aux):
        """ MxNet customized operator forward implementation.

            Embed X with respect to T with vocabulary size of indim. 
            The dimension of the embedding vectors is 1.

            where X is the input tensor and T is the weight tensor.

            .. math::
                val = Embedding(X, T, indim, 1)

            .. math::
                out = squeeze(val, axis=-1)

            where indim is an operator attribute representing vocabulary size of input indices.
        """
        assert is_train == False
        X, T = in_data[0], in_data[1]
        Y = nd.Embedding(X, T, self.in_dim, 1)
        Y = Y.squeeze(axis=-1)
        self.assign(out_data[0], req[0], Y)

    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        assert False

class RightShiftV2(mx.operator.CustomOp):
    def __init__(self, shift_bit, **kwargs):
        super(RightShiftV2, self).__init__(**kwargs)
        self.sb = int(shift_bit)
        assert self.sb > 0

    def forward(self, is_train, req, in_data, out_data, aux):
        assert is_train == False
        X = in_data[0]
        out = X.round()
        if self.sb > 1:
            out = out / 2**self.sb
            out = out.round()
        self.assign(out_data[0], req[0], out)

    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        assert False

class Annotate(mx.operator.CustomOp):
    def __init__(self, in_prec, out_prec, anno_type):
        super(Annotate, self).__init__()
        self.in_prec = int(in_prec)
        self.out_prec = int(out_prec)
        self.anno_type = anno_type

    def forward(self, is_train, req, in_data, out_data, aux):
        assert is_train == False
        self.assign(out_data[0], req[0], in_data[0])

class SimQuant(mx.operator.CustomOp):
    def __init__(self, in_prec, out_prec, scale):
        super(SimQuant, self).__init__()
        self.in_prec = int(in_prec)
        self.out_prec = int(out_prec)
        self.scale = float(scale)

    def forward(self, is_train, req, in_data, out_data, aux):
        assert is_train == False
        X = in_data[0]
        Y = X * self.scale
        self.assign(out_data[0], req[0], Y)

class MRTSimQuant(mx.operator.CustomOp):
    def __init__(self, shift_bit, prec):
        self.sb = int(shift_bit)
        self.tb = int(prec)
        self.denominator = float(2 ** self.sb)
        self.range = (2 ** (self.tb - 1)) - 1

    def forward(self, is_train, req, in_data, out_data, aux):
        assert is_train == False
        X = in_data[0]
        Y = X / self.denominator
        self.assign(out_data[0], req[0], Y)

class Pad(mx.operator.CustomOp):
    def __init__(self, padding):
        super(Pad, self).__init__()
        self.padding = padding

    def forward(self, is_train, req, in_data, out_data):
        assert is_train == False
        self.assign(out_data[0], req[0], in_data[0])


@mx.operator.register("cvm_clip")
class ClipProp(mx.operator.CustomOpProp):
    """ MxNet cvm_clip operator property class.
    """
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
    """ MxNet cvm_left_shift operator property class.
    """
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
    """ MxNet cvm_right_shift operator property class.
    """
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

@mx.operator.register("right_shift")
class RightShiftV2Prop(mx.operator.CustomOpProp):
    """ MxNet right_shift operator property class.
    """
    def __init__(self, shift_bit=0):
        self.shift_bit = shift_bit
        super(RightShiftV2Prop, self).__init__(need_top_grad=False)
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
        return RightShiftV2(self.shift_bit)

@mx.operator.register("cvm_lut")
class LUTProp(mx.operator.CustomOpProp):
    """ MxNet cvm_lut operator property class.
    """
    def __init__(self, in_dim):
        self.in_dim = in_dim
        super(LUTProp, self).__init__(need_top_grad=False)
    def list_arguments(self):
        return ['data', 'table']
    def list_outputs(self):
        return ['output']
    def infer_shape(self, in_shape):
        X_shape = in_shape[0]
        B_shape = in_shape[1]
        out_shape = in_shape[0]
        return [X_shape, B_shape], [out_shape], []
    def infer_type(self, in_type):
        X_type = in_type[0]
        B_type = X_type
        return [X_type, B_type], [X_type], []
    def create_operator(self, ctx, shapes, dtypes):
        return LUT(self.in_dim)

@mx.operator.register("cvm_annotate")
class AnnotateProp(mx.operator.CustomOpProp):
    def __init__(self, in_prec, out_prec, anno_type):
        self.in_prec = in_prec
        self.out_prec = out_prec
        self.anno_type = anno_type
        super(AnnotateProp, self).__init__(need_top_grad=False)
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
        return Annotate(self.in_prec, self.out_prec, self.anno_type)

@mx.operator.register("cvm_sim_quant")
class SimQuantProp(mx.operator.CustomOpProp):
    def __init__(self, in_prec, out_prec, scale):
        self.in_prec = in_prec
        self.out_prec = out_prec
        self.scale = scale
        super(SimQuantProp, self).__init__(need_top_grad=False)
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
        return SimQuant(self.in_prec, self.out_prec, self.scale)

@mx.operator.register("mrt_sim_quant")
class MRTSimQuantProp(mx.operator.CustomOpProp):
    def __init__(self, sb, prec):
        self.sb = sb
        self.prec = prec
        super(MRTSimQuantProp, self).__init__(need_top_grad=False)
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
        return MRTSimQuant(self.sb, self.prec)

@mx.operator.register("cvm_pad")
class PadProp(mx.operator.CustomOpProp):
    def __init__(self, padding):
        self.padding = padding
        super(PadProp, self).__init__(need_top_grad=False)
    def list_arguments(self):
        return ['data']
    def list_outputs(self):
        return ['output']
    def infer_shape(self, in_shape):
        X_shape = in_shape[0]
        pad = [s[0]+s[1] for s in eval(self.padding)]
        out_shape = [s + pad[i] for i, s in enumerate(X_shape)]
        return [X_shape], [out_shape], []
    def infer_type(self, in_type):
        X_type = in_type[0]
        return [X_type], [X_type], []
    def create_operator(self, ctx, shapes, dtypes):
        return Pad(self.padding)





