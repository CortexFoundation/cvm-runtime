import mxnet as mx
from mxnet.gluon import HybridBlock, nn

import logging
from quant_utils import *
from quant_op import *


def _conv3x3(channels, stride, in_channels, use_bias=False):
    return nn.Conv2D(channels, kernel_size=3, strides=stride, padding=1,
                     use_bias=use_bias, in_channels=in_channels)


class BasicBlockV1Q(HybridBlock):
    def __init__(self, channels, stride, quant_flag, downsample=False,
            residual_sb_initializer='zeros', in_channels=0, prefix='', **kwargs):
        if quant_flag.calib_mode == CalibMode.NONE:
            super(BasicBlockV1Q, self).__init__(prefix=prefix, **kwargs)
        else:
            super(BasicBlockV1Q, self).__init__(**kwargs)

        self.body = nn.HybridSequential(prefix='')
        self.body.add(_conv3x3(channels, stride, in_channels, quant_flag.is_fuse_bn))
        requant_helper(self.body, quant_flag)

        if not quant_flag.is_fuse_bn:
            self.body.add(nn.BatchNorm())

        self.body.add(nn.Activation('relu'))
        requant_helper(self.body, quant_flag)

        self.body.add(_conv3x3(channels, 1, channels, quant_flag.is_fuse_bn))
        requant_helper(self.body, quant_flag)

        if not quant_flag.is_fuse_bn:
            self.body.add(nn.BatchNorm())

        if downsample:
            self.downsample = nn.HybridSequential(prefix='')
            self.downsample.add(nn.Conv2D(channels, kernel_size=1, strides=stride,
                        use_bias=quant_flag.is_fuse_bn, in_channels=in_channels))
            requant_helper(self.downsample, quant_flag)

            if not quant_flag.is_fuse_bn:
                self.downsample.add(nn.BatchNorm())
        else:
            self.downsample = None

        self.quant_flag = quant_flag
        self.logger = logging.getLogger("log.quant.op.residual.block")
        self.logger.setLevel(quant_flag.log_level)

        if self.quant_flag.calib_mode != CalibMode.NONE:
            self.shift_bits = self.params.get('requant_shift_bits',
                                    shape=(1,),
                                    init=residual_sb_initializer,
                                    allow_deferred_init=True)

            self.first_sb = self.params.get('first_shift_bits',
                                    shape=(1,),
                                    init=residual_sb_initializer,
                                    allow_deferred_init=True)

            self.second_sb = self.params.get('second_shift_bits',
                                    shape=(1,),
                                    init=residual_sb_initializer,
                                    allow_deferred_init=True)

    def _alias(self):
        return '_plus'

    def hybrid_forward(self, F, x, shift_bits=None, first_sb=None, second_sb=None):
        residual = x

        x = self.body(x)

        if self.downsample:
            residual = self.downsample(residual)


        if self.quant_flag.calib_mode != CalibMode.NONE:
            # self.logger.info("sb = %s, %s", first_sb.asnumpy(), second_sb.asnumpy())
            residual, _ = quant_helper(residual, shift_bits=first_sb, F=F,
                    logger=self.logger, msg="residual first param")
            x, _ = quant_helper(x, shift_bits=second_sb, F=F,
                    logger=self.logger, msg="residual second param")

        out = residual + x

        if self.quant_flag.calib_mode != CalibMode.NONE:
            # self.logger.info("sb = %s", shift_bits.asnumpy())
            out, _ = quant_helper(out, shift_bits=shift_bits, F=F,
                    logger=self.logger, msg=self.name)

        return F.Activation(out, act_type='relu')


class BottleneckV1Q(HybridBlock):
    def __init__(self, channels, stride, quant_flag, downsample=False,
            residual_sb_initializer='zeros', in_channels=0, prefix='', **kwargs):
        if quant_flag.calib_mode == CalibMode.NONE:
            super(BottleneckV1Q, self).__init__(prefix=prefix, **kwargs)
        else:
            super(BottleneckV1Q, self).__init__(**kwargs)

        self.body = nn.HybridSequential(prefix='')
        self.body.add(nn.Conv2D(channels//4, kernel_size=1, strides=stride,
                    use_bias=True))
        requant_helper(self.body, quant_flag)

        if not quant_flag.is_fuse_bn:
            self.body.add(nn.BatchNorm())

        self.body.add(nn.Activation('relu'))
        requant_helper(self.body, quant_flag)
        self.body.add(_conv3x3(channels//4, 1, channels//4, quant_flag.is_fuse_bn))
        requant_helper(self.body, quant_flag)

        if not quant_flag.is_fuse_bn:
            self.body.add(nn.BatchNorm())

        self.body.add(nn.Activation('relu'))
        requant_helper(self.body, quant_flag)
        self.body.add(nn.Conv2D(channels, kernel_size=1, strides=1,
                    use_bias=True))
        requant_helper(self.body, quant_flag)

        if not quant_flag.is_fuse_bn:
            self.body.add(nn.BatchNorm())

        if downsample:
            self.downsample = nn.HybridSequential(prefix='')
            self.downsample.add(nn.Conv2D(channels, kernel_size=1, strides=stride,
                                          use_bias=quant_flag.is_fuse_bn, in_channels=in_channels))
            requant_helper(self.downsample, quant_flag)

            if not quant_flag.is_fuse_bn:
                self.downsample.add(nn.BatchNorm())
        else:
            self.downsample = None

        self.quant_flag = quant_flag
        self.logger = logging.getLogger("log.quant.op.residual.block")
        self.logger.setLevel(quant_flag.log_level)

        if self.quant_flag.calib_mode != CalibMode.NONE:
            self.shift_bits = self.params.get('requant_shift_bits',
                                    shape=(1,),
                                    init=residual_sb_initializer,
                                    allow_deferred_init=True)

            self.first_sb = self.params.get('first_shift_bits',
                                    shape=(1,),
                                    init=residual_sb_initializer,
                                    allow_deferred_init=True)

            self.second_sb = self.params.get('second_shift_bits',
                                    shape=(1,),
                                    init=residual_sb_initializer,
                                    allow_deferred_init=True)

    def _alias(self):
        return '_plus'

    def hybrid_forward(self, F, x, shift_bits=None, first_sb=None, second_sb=None):
        residual = x

        x = self.body(x)

        if self.downsample:
            residual = self.downsample(residual)


        if self.quant_flag.calib_mode != CalibMode.NONE:
            residual, _ = quant_helper(residual, shift_bits=first_sb, F=F,
                    logger=self.logger, msg="residual first param")
            x, _ = quant_helper(x, shift_bits=second_sb, F=F,
                    logger=self.logger, msg="residual second param")

        out = residual + x

        if self.quant_flag.calib_mode != CalibMode.NONE:
            out, _ = quant_helper(out, shift_bits=shift_bits, F=F,
                    logger=self.logger, msg=self.name)

        return F.Activation(out, act_type='relu')

class ResNetV1Q(HybridBlock):
    def __init__(self, block, layers, channels, quant_flag, classes=1000, **kwargs):
        super(ResNetV1Q, self).__init__(**kwargs)
        assert len(layers) == len(channels) - 1
        self.quant_flag = quant_flag

        with self.name_scope():
            self.features = nn.HybridSequential(prefix='')
            self.features.add(nn.Conv2D(channels[0], 7, 2, 3,
                        use_bias=self.quant_flag.is_fuse_bn))
            requant_helper(self.features, quant_flag)

            if not self.quant_flag.is_fuse_bn:
                self.features.add(nn.BatchNorm())

            self.features.add(nn.Activation('relu'))
            requant_helper(self.features, quant_flag)

            self.features.add(nn.MaxPool2D(3, 2, 1))
            requant_helper(self.features, quant_flag)

            for i, num_layer in enumerate(layers):
                stride = 1 if i == 0 else 2
                self.features.add(self._make_layer(block, num_layer, channels[i+1],
                                                   stride, i+1, in_channels=channels[i]))

            # self.features.add(nn.GlobalAvgPool2D())
            self.features.add(GlobalAvgPool2D(quant_flag))
            requant_helper(self.features, quant_flag)

            self.output = nn.HybridSequential(prefix='')
            self.output.add(nn.Dense(classes, in_units=channels[-1]))
            requant_helper(self.output, quant_flag)

    def _make_layer(self, block, layers, channels, stride, stage_index, in_channels=0):
        layer = nn.HybridSequential(prefix='stage%d_'%stage_index)
        with layer.name_scope():
            layer.add(block(channels, stride, self.quant_flag, channels != in_channels,
                        in_channels=in_channels, prefix=''))
            for _ in range(layers-1):
                layer.add(block(channels, 1, self.quant_flag, False,
                            in_channels=channels, prefix=''))
        return layer

    def hybrid_forward(self, F, x):
        x = self.features(x)
        x = self.output(x)

        return x
