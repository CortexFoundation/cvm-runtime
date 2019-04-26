import mxnet as mx
from mxnet.gluon import HybridBlock, nn
from mxnet import ndarray as nd
from mxnet import sym

import logging

from quant_utils import *

class ReQuant(HybridBlock):
    """Given a input data of type FP32, quantize it into a INT8 and shift bits

    Parameters
    ----------
    target_bits: int
        Quantize input data into target bits' int type
    """
    def __init__(self, quant_flag, pre_lname='conv', prefix=None,
            requant_sb_initializer='zeros', **kwargs):
        prefix_name = pre_lname if pre_lname.endswith("_") else pre_lname+"_"
        super(ReQuant, self).__init__(prefix=prefix_name, **kwargs)

        self.prefix_name = prefix_name
        self.shift_bits = self.params.get('requant_shift_bits',
                                shape=(1,),
                                init=requant_sb_initializer,
                                allow_deferred_init=True)
        self.logger = logging.getLogger("log.quant.op.requant")
        self.logger.setLevel(quant_flag.log_level)

    def hybrid_forward(self, F, x, shift_bits):
        out, _ = quant_helper(x, shift_bits=shift_bits, F=F,
                logger=self.logger, msg=self.prefix_name[:-1])
        return out

    def __repr__(self):
        s = '{name}(pre_lname={pre_lname}, shift_bits={shift_bits})'
        return s.format(name='_requant',
                        pre_lname=self.pre_lname,
                        shift_bits=self.shift_bits)

def requant_helper(graph, quant_flag):
    logger = logging.getLogger("log.quant.op.requant.helper")
    logger.setLevel(quant_flag.log_level)

    if quant_flag.calib_mode == CalibMode.NONE:
        return

    previous_lname = graph[-1].name

    disable_flag = any(dis_layer in previous_lname for dis_layer in quant_flag.disabled_layers)
    if len(quant_flag.disabled_layers) > 0 and disable_flag:
        logger.debug("disable requant for layer [ %s ] in disabled_layers", previous_lname)
        return

    logger.debug("requant for layer [ %s ]"%previous_lname)
    graph.add(ReQuant(quant_flag, pre_lname=previous_lname))


class Pass(HybridBlock):
    def __init__(self, quant_flag, **kwargs):
        super(Pass, self).__init__(**kwargs)

        self.logger = logging.getLogger("log.quant.op.pass")
        self.logger.setLevel(quant_flag.log_level)

    def hybrid_forward(self, F, x):
        out = x
        if isinstance(x, nd.NDArray):
            self.logger.debug("quant %s with data=<%s,%s>, max=%s, min=%s",
                    "pass layer",
                    out.asnumpy().flatten()[0],
                    out.asnumpy().flatten()[0:49].max(),
                    out.max().asnumpy(),
                    out.min().asnumpy())
        return out

    def _alias(self):
        return '_pass'

class GlobalAvgPool2D(nn.GlobalAvgPool2D):
    def __init__(self, quant_flag, weight_initializer='zeros', **kwargs):
        # if quant_flag.calib_mode == CalibMode.NONE:
        super(GlobalAvgPool2D, self).__init__(**kwargs)

        self.quant_flag = quant_flag
        self.logger = logging.getLogger("log.quant.op.pool.avg.global")
        self.logger.setLevel(quant_flag.log_level)

        if quant_flag.calib_mode != CalibMode.NONE:
            self.scale = self.params.get('weight',
                                    shape=(1,),
                                    init=weight_initializer,
                                    allow_deferred_init=True)

    def hybrid_forward(self, F, x, scale=None):
        if self.quant_flag.calib_mode == CalibMode.NONE:
            return super(GlobalAvgPool2D, self).hybrid_forward(F, x)

        out = x.sum(axis=(2, 3))
        # self.logger.debug("After sum: shape=%s, data=<%s,%s,%s>",
                # out.shape,
                # out.asnumpy().flatten()[0],
                # out.max().asnumpy(), out.min().asnumpy())

        if isinstance(x, nd.NDArray):
            out = out * scale
        else:
            out = F.broadcast_mul(out, scale)
        # self.logger.debug("After mean: shape=%s, data=<%s,%s,%s>, div=%s",
                # out.shape,
                # out.asnumpy().flatten()[0],
                # out.max().asnumpy(), out.min().asnumpy(),
                # (x.shape[2]*x.shape[3]))

        out = F.reshape(out, (0, 0, 1, 1))
        return out

# def conv2d_quant(channels, kernel_size, stride, padding, use_bias, in_channels):
    # quant_conv_layer = nn.HybridSequential(prefix='')
    # quant_conv_layer.add(QuantOp())
    # quant_conv_layer.add(nn.Conv2D(channels, kernel_size, strides=stride,
                # padding=padding, use_bias=False))
    # quant_conv_layer.add(QuantOp())
    # if use_bias:
        # quant_conv_layer.add(BiasAddOp())


    # quant aware calibration
    # added_params_name, val_changed_params_name, deleted_params_name = [], [], []
    # def move_params(layer_name, previous_lout_sb):
    #     weight_name = layer_name.replace("_fwd_output", "_weight")
    #     assert weight_name not in qparams
    #     weight_quant_name = weight_name + "_quant"
    #     qparams[weight_name] = qparams[weight_quant_name]

    #     # set layer input shift_bits
    #     layer_input_sb_name = layer_name.replace("_fwd_output", "_input_shift_bits")
    #     assert previous_lout_sb != None
    #     qparams[layer_input_sb_name] = previous_lout_sb
    #     added_params_name.append(layer_input_sb_name)

    #     weight_shift_bits_name = weight_name + "_shift_bits"
    #     bias_name = layer_name.replace("_fwd_output", "_bias")
    #     bias_quant_names = [bias_name+"_quant", bias_name+"_shift_bits"]
    #     if bias_name in qparams:
    #         shift_bits = qparams[weight_shift_bits_name] + qparams[layer_input_sb_name]
    #         qparams[bias_quant_names[0]], qparams[bias_quant_names[1]] = \
    #             quant_helper(qparams[bias_name], shift_bits=shift_bits)
    #         qparams[bias_name] = qparams[bias_quant_names]
    #         added_params_name.extend(bias_quant_names)
    #         val_changed_params_name.append(bias_name)

    # layer_step_wise_prefix = "./data/params.resnet18.quant.tmp"
    # tmp_params_file = layer_step_wise_prefix

    # previous_lout_sb = None
    # for layer_name in outputs:
    #     # prepare name for quant model params
    #     move_params(layer_name,
    #             input_shift_bits if previous_lout_sb == None else previous_lout_sb)
    #     nd.save(tmp_params_file, qparams)

    #     # calculate calib output of quantize layer
    #     sym_block = gluon.SymbolBlock(layers[layer_name], [inputs])
    #     sym_block.load_parameters(tmp_params_file, ctx=ctx, ignore_extra=True)
    #     calib_res = sym_block.forward(image_data.as_in_context(ctx))

    #     # calculate requant shift_bits
    #     _, requant_shift_bits = quant_helper(calib_res)
    #     requant_sb_name = layer_name.replace("_fwd_output", "_requant_shift_bits")
    #     qparams[requant_sb_name] = requant_shift_bits

    #     # next layer input shift bits is the sum of input_shift_bits of the layer,
    #     # weight_shift_bits, and requant_shift_bits
    #     weight_sb_name = layer_name.replace("_fwd_output", "_weight") + "_shift_bits"
    #     linput_sb_name = layer_name.replace("_fwd_output", "_input_shift_bits")
    #     previous_lout_sb = qparams[weight_sb_name] + qparams[linput_sb_name]
    #     previous_lout_sb += qparams[requant_sb_name]

    # print ("[ added_params_name       ]: ", added_params_name)
    # print ("[ deleted_params_name     ]: ", deleted_params_name)


class Dense(HybridBlock):
    def __init__(self, quant_flag, **kwargs):
        super(Dense, self).__init__(prefix='fc0_', **kwargs)

        self.quant_flag = quant_flag

        if not quant_flag.matrix_decomposition:
            setattr(self, 'weight',
                self.params.get('weight',
                        init='zeros',
                        allow_deferred_init=True))
            setattr(self, 'bias',
                self.params.get('bias',
                        init='zeros',
                        allow_deferred_init=True))
            return

        self.matrix_len = 100352
        self.max_len = 60000

        start, step, idx = 0, self.max_len, 0
        while start < self.matrix_len:
            stop = min(start+step, self.matrix_len)

            weight_name = str(idx) + '_weight'
            bias_name = str(idx) + '_bias'
            setattr(self, weight_name,
                self.params.get(weight_name,
                            init='zeros',
                            allow_deferred_init=True))

            setattr(self, bias_name,
                self.params.get(bias_name,
                        init='zeros',
                        allow_deferred_init=True))


            start, idx = stop, idx+1

        for i in range(idx-1):
            plus_name_f = '_plus' + str(i) + '_first_shift_bits'
            plus_name_s = '_plus' + str(i) + '_second_shift_bits'
            setattr(self, plus_name_f,
                self.params.get(plus_name_f,
                            init='zeros',
                            allow_deferred_init=True))
            setattr(self, plus_name_s,
                self.params.get(plus_name_s,
                            init='zeros',
                            allow_deferred_init=True))

            requant_name = '_plus' + str(i) + '_requant_shift_bits'
            setattr(self, requant_name,
                self.params.get(requant_name,
                            init='zeros',
                            allow_deferred_init=True))

    def hybrid_forward(self, F, x, **kwargs):
        if not self.quant_flag.matrix_decomposition:
            x = F.FullyConnected(x, kwargs['weight'],
                    kwargs['bias'], num_hidden=10)
        else:
            nodes = []
            start, step, idx = 0, self.max_len, 0
            while start < self.matrix_len:
                stop = min(start+step, self.matrix_len)

                weight_name = str(idx) + '_weight'
                bias_name = str(idx) + '_bias'
                tmp = F.slice(x, begin=(0, start), end=(10, stop))
                tmp = F.FullyConnected(tmp, kwargs[weight_name],
                        kwargs[bias_name], name=str(idx), num_hidden=10)
                nodes.append(tmp)

                start, idx = stop, idx+1

            i = 0
            while len(nodes) > 1:
                a, b = nodes.pop(0), nodes.pop(0)

                if self.quant_flag.calib_mode != CalibMode.NONE:
                    a_sb_name = '_plus' + str(i) + '_first_shift_bits'
                    a , _ = quant_helper(a, shift_bits=kwargs[a_sb_name], F=F)

                    b_sb_name = '_plus' + str(i) + '_second_shift_bits'
                    b , _ = quant_helper(b, shift_bits=kwargs[b_sb_name], F=F)

                out = a + b

                if self.quant_flag.calib_mode != CalibMode.NONE:
                    requant_name = '_plus' + str(i) + '_requant_shift_bits'
                    out, _ = quant_helper(out, shift_bits=kwargs[requant_name], F=F)

                nodes.append(out)

                i += 1

            x = nodes[0]

        return x
