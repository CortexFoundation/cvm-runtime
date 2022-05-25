import mxnet as mx
from mxnet import optimizer

class Parameter(object):
    def __init__(self, param_name:str, param_nd, index:int):
        self.name = param_name
        self.param_nd = param_nd
        self.index = index
        self.id = id(param_nd)
        self.state = None

    def attach_state(self, state):
        self.state = state

class ParameterDict(object):
    def __init__(self):
        self._index_to_parameter = {}
        self._id_to_parameter = {}
        self._name_to_parameter = {}
        self._param_list = []

    def add_param(self, param:Parameter):
        assert param.index not in self._index_to_parameter
        assert param.id    not in self._id_to_parameter
        assert param.name  not in self._name_to_parameter

        self._index_to_parameter[param.index] = param
        self._id_to_parameter   [param.id   ] = param
        self._name_to_parameter [param.name ] = param
        self._param_list.append(param)

    def __iter__(self):
        return self._param_list.__iter__()
    
    def get(self, key):
        if type(key) is int:
            return self._index_to_parameter[key]
        elif type(key) is str:
            return self._name_to_parameter[key]
        else:
            return self._id_to_parameter[id(key)]

    @classmethod
    def load_param_dict(cls, param_dict:dict):
        pd = cls()
        for index, name in enumerate(param_dict):
            param = Parameter(name, param_dict[name], index)
            pd.add_param(param)
        return pd


class Optimizer(object):
    def __init__(self, param_dict:dict, opt_name: str, opt_kwargs: dict):
        self._pd = ParameterDict.load_param_dict(param_dict)
        self._opt = optimizer.Optimizer.create_optimizer(opt_name, **opt_kwargs)
        self._init_graph()

    def _init_graph(self):
        for param in self._pd:
            param.attach_state(self._opt.create_state(param.index, param.param_nd))
            param.param_nd.attach_grad('write')
    
    def zero_grad(self):
        for param in self._pd:
            param.param_nd.grad[:] = 0

    def step(self):
        for param in self._pd:
            self._opt.update(param.index, param.param_nd, param.param_nd.grad, param.state)