import mxnet as mx

QUANT_OP_PREFIX = "MRT_"



class Wrapper(object):
    """Basic Class for Quantization Info, Factory Functions, etc.  
    """
    def __init__(self, op:mx.sym.Symbol, config:dict):
        self._ori_op = op
        self._config = config
        self._attr_dict = {}
        self._build_attr_dict()
        self._op = None
        self._param = None

    def _build_attr_dict(self):
        raise NotImplementedError

    def new_op(self):
        self._op = mx.sym.Custom(**self._attr_dict)
        return self._op

    def op(self):
        return self._op

    def attr(self, key:str):
        if key in self._attr_dict:
            return self._attr_dict[key]
        return 'null'

    def key(self):
        return self._attr_dict[name]

    def init_param(self, *args, **kwargs):
        raise NotImplementedError

    def param(self)->dict:
        return self._param
