from .common import *
import mxnet as mx
import mxnet.ndarray as nd

def _round_ste(x):
    return mx.nd.stop_gradient(mx.nd.round(x) - x) + x


def _new_detached_nd(*args):
    res = []
    for item in args:
        res.append(item.detach())
    return res


class UniformAffineQuantizerWrapper(Wrapper):
    _scale_methods = ['max_scale', 'max', 'mse']
    def __init__(self, op, config):
        self.channel_wise = False
        self.scale_method = config['scale_method'] if 'scale_method' in config else _scale_methods[0]
        super(UniformAffineQuantizerWrapper, self).__init__(op, config)
        self.delta_nd = None
        self.delta_op = None
        self.zero_point_nd = None
        self.zero_point_op = None

    def _build_attr_dict(self):
        assert(self._config['q_op_name'] not in self._ori_op.attr('name'))
        # None Symble
        self._attr_dict['op_type'] = self._config['q_op_name']
        self._attr_dict['name'] = f"{self._attr_dict['op_type']}_{self._ori_op.attr('name')}"
        self._attr_dict['n_bits'] = self._config['n_bits']
        self.channel_wise = self._config['channel_wise']
        # Symbles
        self._attr_dict['data'] = self._ori_op
        if not self.channel_wise:
            self.delta_op = mx.sym.Variable(f"{self._attr_dict['name']}_delta", shape=(1))
            self.zero_point_op = mx.sym.Variable(f"{self._attr_dict['name']}_zero_point", shape=(1))
            self._attr_dict['delta'] = self.delta_op
            self._attr_dict['zero_point'] = self.zero_point_op
        elif self.channel_wise:
            # Assume the the fisrt dim of input data is channel
            assert(len(self._ori_op.infer_shape()[1]) == 1)
            ori_op_shape = self._ori_op.infer_shape()[1][0]
            channel_wise_shape = (ori_op_shape[0], * ([1] * (len(ori_op_shape) - 1)))
            self.delta_op = mx.sym.Variable(
                f"{self._attr_dict['name']}_delta",
                shape=channel_wise_shape)
            self.zero_point_op = mx.sym.Variable(
                f"{self._attr_dict['name']}_zero_point",
                shape=channel_wise_shape)
            self._attr_dict['delta'] = self.delta_op
            self._attr_dict['zero_point'] = self.zero_point_op
        else:
            raise TypeError

    def init_param(self, data: nd.NDArray):
        pass

    def _init_param_impl(self, input_data: nd.NDArray, channel_wise:bool=False):
        delta, zero_point = None, None
        if channel_wise:
            x_clone = input_data.copy().detach()
            n_channels = x_clone.shape[0]
            if len(x.shape) == 4:
                x_max = x_clone.abs().max(dim=-1)[0].max(dim=-1)[0].max(dim=-1)[0]
            else:
                x_max = x_clone.abs().max(dim=-1)[0]
            delta = x_max.clone()
            zero_point = x_max.clone()
            # determine the scale and zero point channel-by-channel
            for c in range(n_channels):
                delta[c], zero_point[c] = self.init_quantization_scale(x_clone[c], channel_wise=False)
            if len(x.shape) == 4:
                delta = delta.view(-1, 1, 1, 1)
                zero_point = zero_point.view(-1, 1, 1, 1)
            else:
                delta = delta.view(-1, 1)
                zero_point = zero_point.view(-1, 1)
        else:
            if 'max' in self.scale_method:
                x_min = min(x.min().item(), 0)
                x_max = max(x.max().item(), 0)
                if 'scale' in self.scale_method:
                    x_min = x_min * (self.n_bits + 2) / 8
                    x_max = x_max * (self.n_bits + 2) / 8

                x_absmax = max(abs(x_min), x_max)
                if self.sym:
                    x_min, x_max = -x_absmax if x_min < 0 else 0, x_absmax

                delta = float(x_max - x_min) / (self.n_levels - 1)
                if delta < 1e-8:
                    warnings.warn('Quantization range close to zero: [{}, {}]'.format(x_min, x_max))
                    delta = 1e-8

                zero_point = round(-x_min / delta)
                delta = torch.tensor(delta).type_as(x)

            elif self.scale_method == 'mse':
                # we always use symmetric quantization in mse mode
                x_absmax = x.abs().max()
                x_min = x.min().item()
                best_score = 1000
                for i in range(80):
                    new_max = x_absmax * (1.0 - (i * 0.01))
                    x_q = self.quantize(x, new_max)
                    # L_p norm minimization as described in LAPQ
                    # https://arxiv.org/abs/1911.07190
                    score = lp_loss(x, x_q, p=2.4, reduction='all')
                    if score < best_score:
                        best_score = score
                        delta = (2 * new_max) / (2 ** self.n_bits - 1)
                        zero_point = (new_max / delta).round() if x_min < 0 else 0
                        # re-calculate the scale delta if zero-point is not 0,
            else:
                raise NotImplementedError
#    def init_param(self, data:nd.NDArray, scale_method:str='max'):
#        assert scale_method in _scale_methods
#        if self.channel_wise:
#            data_abs = data.abs()
#            data_max_per_channel = 



class UniformAffineQuantizer(mx.operator.CustomOp):
    def __init__(self, n_bits):
        super(UniformAffineQuantizer, self).__init__()
        self.n_bits = n_bits
        self.n_levels = 2 ** self.n_bits

    def forward(self, is_train, req, in_data, out_data, aux):
        conv_weight, delta, zero_point = in_data[0], in_data[1], in_data[2]
        x_int = _round_ste(conv_weight / delta) + zero_point #TODO: Zero point is hard to implemented in the Fully Quantized Conditions.
        x_quant = mx.nd.clip(x_int, 0, self.n_levels - 1)
        x_dequant = (x_quant - zero_point) * delta
        self.assign(out_data[0], req[0], x_dequant)

    def backward(self, req, out_grad, in_data, out_data, in_grad, aux): # Seems like checkpoint techs in pytorch 
        conv_weight, delta, zero_point = _new_detached_nd(*in_data[:3])# in_data[0].copy().detach(), in_data[1].copy().detach(), in_data[2].copy().detach()
        conv_weight.attach_grad()
        delta.attach_grad()
        zero_point.attach_grad()
        with mx.autograd.record():
            x_int = _round_ste(conv_weight / delta) + zero_point
            x_quant = mx.nd.clip(x_int, 0, self.n_levels - 1)
            x_dequant = (x_quant - zero_point) * delta
        x_dequant.backward(_new_detached_nd(out_grad[0])[0])

        self.assign(in_grad[0], req[0], conv_weight.grad)
        self.assign(in_grad[1], req[1], delta.grad)
        self.assign(in_grad[2], req[2], zero_point.grad)


@mx.operator.register(QUANT_OP_PREFIX + "UniformAffineQuantizer")
class UniformAffineQuantizerProp(mx.operator.CustomOpProp):
    def __init__(self, n_bits):
        super(UniformAffineQuantizerProp, self).__init__()
        n_bits = n_bits if type(n_bits) is int else int(n_bits) 

        assert 2 <= n_bits <= 32, 'bitwidth not supported'
        self.n_bits = n_bits

    def list_arguments(self):
        return ['data', 'delta', 'zero_point']

    def list_outputs(self):
        return ['output']

    def infer_shape(self, in_shape):
        assert(len(in_shape)==3)
        return [*in_shape], [in_shape[0]], []

    def infer_type(self, in_type):
        return [*in_type], [in_type[0]], []

    def create_operator(self, ctx, shapes, dtypes):
        return UniformAffineQuantizer(n_bits=self.n_bits)

