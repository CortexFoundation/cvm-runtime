from .common import *
import mxnet as mx


class ProxyWrapper(Wrapper):
    def __init__(self, op, config):
        super(ProxyWrapper, self).__init__(op, config)

    def _build_attr_dict(self):
        # None Symble
        self._attr_dict['op_type'] = self._config['q_op_name']
        self._attr_dict['name'] = f"{self._attr_dict['op_type']}_{self._ori_op.attr('name')}"
        # Symbles
        self._attr_dict['data'] = self._ori_op
        self._attr_dict['qbias'] = mx.sym.Variable(**self._ori_op.list_attr(), name=f"{self._attr_dict['name']}_qbias")


class Proxy(mx.operator.CustomOp):
    def __init__(self):
        super(Proxy, self).__init__()

    def forward(self, is_train, req, in_data, out_data, aux):
        self.assign(out_data[0], req[0], in_data[1])

    def backward(self, req, out_grad, in_data, out_data, in_grad, aux): # Seems like checkpoint techs in pytorch 
        assert(req[0] == req[1])
        self.assign(in_grad[1], req[0], out_grad[0])


@mx.operator.register(QUANT_OP_PREFIX + "Proxy")
class ProxyProp(mx.operator.CustomOpProp):
    def __init__(self):
        super(ProxyProp, self).__init__()
        
    def list_arguments(self):
        return ['data', 'qbias']

    def list_outputs(self):
        return ['output']

    def infer_shape(self, in_shape):
        assert(len(in_shape)==2)
        return [*in_shape], [in_shape[0]], []

    def infer_type(self, in_type):
        return [*in_type], [in_type[0]], []

    def create_operator(self, ctx, shapes, dtypes):
        return Proxy()

