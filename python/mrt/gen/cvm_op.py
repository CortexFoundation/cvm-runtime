import mxnet as mx
from mxnet import ndarray as nd

import numpy as np


class ChannelConv2D(mx.operator.CustomOp):
    def __init__(
        self, stride=(1,1), dilate=(1,1), num_group=1,
        channel_step=1, **kwargs):
        super(ChannelConv2D, self).__init__(**kwargs)

    def forward(self, is_train, req, in_data, out_data, aux):
        assert is_train == False
        X, W  = in_data[0], in_data[1]
        assert False, "To be implemented now"
        # TODO(archRev): out
        self.assign(out_data[0], req[0], out)

    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        assert False


@mx.operator.register("cvm_channel_conv2d")
class ChannelConv2DProp(mx.operator.CustomOpProp):
    """ MxNet cvm_channel_conv2d operator property class.
    """
    def __init__(
        self, stride=(1,1), dilate=(1,1), num_group=1, channel_step=1):
        self.stride = eval(stride)
        self.dilate = eval(dilate)
        self.num_group = eval(num_group)
        self.channel_step = eval(channel_step)
        super(ChannelConv2DProp, self).__init__(need_top_grad=False)

    def list_arguments(self):
        return ['data', 'weight']

    def list_outputs(self):
        return ['output']

    def infer_shape(self, in_shape):
        X_shape, W_shape = in_shape[0], in_shape[1]
        SH, SW = self.stride
        DH, DW = self.dilate
        H, W = X_shape[2:]
        KH, KW = W_shape[2:]
        OH = int((H-DH*(KH-1)-1) / SH) + 1
        OW = int((W-DW*(KW-1)-1) / SW) + 1
        out_shape = (X_shape[0], W_shape[0], OH, OW)
        return [X_shape, W_shape], [out_shape], []

    def infer_type(self, in_type):
        X_type, W_type = in_type[0], in_type[1]
        return [X_type, W_type], [X_type], []

    def create_operator(self, ctx, shapes, dtypes):
        return ChannelConv2D(self.channel_step)
