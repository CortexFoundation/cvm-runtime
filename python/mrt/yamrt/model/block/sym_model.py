# General
# None
# Mxnet Backend
import mxnet as mx
from mxnet import autograd
from mxnet import sym
from mxnet import nd
from ..model import *

from mrt.sym_utils import *


class SymbolModel(Model):
    def __init__(self, ctx=mx.cpu()):
        super(SymbolModel, self).__init__()
        self._flush_name()
        self._ops = {}
        self._forward_topo = []
        self._param = {}
        self._ctx = ctx

    def _flush_name(self):
        self._name = f"{list(self._inputs.keys())}->{list(self._outputs.keys())}"

    def attach_sym(self, op:sym.Symbol, param_dict:dict):
        op_name = op.attr('name')
        assert op_name not in self._forward_topo 
        assert op_name not in self._inputs
        childs = op.get_children()
        if childs is not None:
            for child in op.get_children():
                child_name = child.attr("name")
                child_op = child.attr("op_name")
                if child_op == "null":
                    if child_name not in param_dict:
                        if child_name not in self._inputs:
                            self._inputs[child_name] = get_entry_id(child)
                    else:
                        self._param[child_name] = [param_dict[child_name].as_in_context(self._ctx)]
                else:
                    if child_name not in self._param and child_name not in self._ops:
                        self._inputs[child_name] = get_entry_id(child)
        else:
            assert(op.attr('op_name') == 'null')
        self._forward_topo.append(op_name)
        self._ops[op_name] = op
        self._flush_name()

    def _set_train(self):
        self._training = True

    def _set_eval(self):
        self._training = False

    def add_output(self, name):
        assert name in self._forward_topo
        if name not in self._outputs:
            self._outputs[name] = self._ops[name]
        self._flush_name()

    def _passive_update_dict(self, ori_config, to_update):
        for key in to_update:
            if key in ori_config:
                ori_config[key] = to_update[key]

    def forward(self, data:dict):
        for key in self._inputs:
            assert ( key in data )
        data.update(self._param)
        if self._training:
            with autograd.record():
                self._forward_imp(data)
        else:
            self._forward_imp(data)
        res = {}
        for out in self._outputs:
            res[out] = data[out]
        return res

    def _forward_imp(self, intermediate):
        for name in self._forward_topo:
            op = self._ops[name]
            op_name = op.attr('op_name')
            if op.attr('op_name') == 'null':
                assert name in intermediate
            else:
                childs, attr = sym_iter(op.get_children()), op.list_attr()
                cinfos = [(c.attr('name'), get_entry_id(c)) for c in childs]
                nd_inputs = [intermediate[n[0]][n[1]] for n in cinfos]
                nd_out = get_nd_op(op_name) (*nd_inputs, **attr)
                out = [nd_out] if not has_multi_outs(op) else nd_out
                assert name not in intermediate
                intermediate[name] = out

    def __call__(self, data):
        if type(data) is not dict:
            assert type(data) is nd.NDArray
            data = {'data': [data], }
        else:
            # assert 'data' in data
            data = data.copy()
            data['data'] = [data['data']]
        return self.forward(data)

