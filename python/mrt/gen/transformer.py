import logging

import mxnet as mx
from mxnet import ndarray as nd
from os import path

# import as registry pattern
from mrt.gen import tfm_ops  # pylint: disable=unused-import
from mrt import cvm_op   # pylint: disable=unused-import

from mrt.sym_utils import topo_sort
from mrt.tfm_pass import OUT_KEY
from .tfm_types import get_feature, get_buffer, BUF_TYPE_EXP, AFeature
from .tfm_pass import quantize, sym_calibrate, rewrite, \
                      sym_config_infos, sym_slice_channel, \
                      sym_separate_pad, sym_separate_bias, \
                      prepare_for_compile

from mrt import transformer as tfm
from mrt import sym_utils as sutils
from mrt import utils
from mrt import sim_quant_helper as sim
from mrt import tfm_pass as tpass

# TODO: collect hyper-parameters

__all__ = ["MRT", "Model"]

class Model(tfm.Model):
    @staticmethod
    def load(symbol_file, params_file):
        symbol = mx.sym.load(symbol_file)
        params = nd.load(params_file)
        return Model(symbol, params)

    def split(self, keys):
        return split_model(self, keys)

    def prepare(self, input_shape=None):
        model = init(self, input_shape)
        self.symbol, self.params = model.symbol, model.params

    def get_mrt(self):
        return MRT(self)

def init(model, input_shape=None):
    logger = logging.getLogger("mrt.prepare")
    logger.info("Model initializing...")

    _sym, _prm = model.symbol, model.params
    tpass.name_duplicate_check(_sym, _prm)

    if isinstance(input_shape, dict):
        _sym, _prm = tpass.attach_input_shape(_sym, _prm, input_shape)
        _sym, _prm = tpass.fuse_multiple_inputs(_sym, _prm)
    elif input_shape is not None:
        model_inputs = tpass.model_inputs(_sym, _prm)
        assert model_inputs == 1, "Multiple inputs non-known shape"
        _sym, _prm = tpass.input_name_replace(_sym, _prm)
        _sym, _prm = tpass.attach_input_shape(_sym, _prm,
                                              {"data": input_shape})
    tpass.infer_shape(_sym, _prm) # check infer_shape is correct

    _sym, _prm = tpass.fuse_multiple_outputs(_sym, _prm)
    _sym, _prm = tpass.fuse_constant(_sym, _prm)
    _sym, _prm = tpass.fuse_transpose(_sym, _prm)
    _sym, _prm = rewrite(_sym, _prm)
    # _sym, _prm = sym_separate_pad(_sym, _prm)
    _sym, _prm = sym_separate_bias(_sym, _prm)
    _sym, _prm = tpass.fuse_constant(_sym, _prm)
    _sym, _prm = tpass.params_unique(_sym, _prm)

    return Model(_sym, _prm)


class MRT(tfm.MRT):
    def __init__(self, model):
        self.old_names = model.output_names()
        self.current_model = model

        self._data = None
        self.cfg_dict = {}
        self.features = {}
        self.buffers = {}
        self.restore_names = set()

        self._op_default_input_precs()
        self.precs = {}

        self.softmax_lambd = 10
        self.shift_bits = 5

    def set_cfg_dict(self, cfg_dict):
        """ Set up configuration dict for the mrt instance
        """
        self.cfg_dict = cfg_dict

    def calibrate(self, ctx=mx.cpu(), lambd=None, old_ths=None):
        # Configure by cfg_dict
        self.cfg_dict = sym_config_infos(
            self.current_model.symbol, self.current_model.params,
            cfg_dict=self.cfg_dict)

        # Slice Channel for customized granularity
        sym, params = sym_slice_channel(
            self.current_model.symbol, self.current_model.params,
            cfg_dict=self.cfg_dict)
        self.current_model = Model(sym, params)

        # initialize precs
        self.precs = {s.attr('name'):{} \
            for s in topo_sort(self.current_model)}
        if 'data' not in self.precs:
            raise RuntimeError("please invoke `init` function first")
        self.precs['data'][OUT_KEY] = 8

        # Perform Calibration
        self.features = sym_calibrate(
            self.current_model.symbol, self.current_model.params,
            self._data, self.cfg_dict, ctx=ctx)
        return self.features

    def _op_default_input_precs(self):
        op_precs = self.op_input_precs = {}
        for name in ['Convolution', 'FullyConnected',
                     'sigmoid', 'exp', 'softmax']:
            op_precs[name] = 8
        op_precs['sum'] = 8
        for name in ['broadcast_add', 'broadcast_sub',
                     'elemwise_add', 'elemwise_sub', 'slice_like']:
            op_precs[name] = 16
        op_precs['broadcast_mul'] = 16
        op_precs['L2Normalization'] = 8
        op_precs['Concat'] = 16
        op_precs['Embedding'] = 16
        op_precs['slice_like'] = 30
        op_precs['batch_dot'] = 8
        op_precs['add_n'] = 16

    def set_threshold(self, name, threshold):
        self.features[name] = AFeature(threshold)

    def quantize(self):
        # # print out zero point quantization candidates
        #  from mrt.sym_utils import is_params, sym_iter
        #  sym, params = self.current_model.symbol, self.current_model.params
        #  for s in topo_sort(sym):
            #  name, op_name = s.attr('name'), s.attr('op_name')
            #  if op_name in ["broadcast_add"]:
                #  childs = sym_iter(s.get_children())
                #  for c in childs:
                    #  cname = c.attr('name')
                    #  if is_params(c, params):
                        #  weight = params[cname]
                        #  maxv = weight.max().asscalar()
                        #  minv = weight.min().asscalar()
                        #  #  print(cname)
            #  elif op_name in ["Convolution"]:
                #  childs = sym_iter(s.get_children())
                #  for c in childs:
                    #  cname = c.attr('name')
                    #  if is_params(c, params):
                        #  weight = params[cname]
                        #  maxv = weight.max().asscalar()
                        #  minv = weight.min().asscalar()
                        #  print(maxv, minv, cname)
        #  exit()

        # # print out channel slice candidates
        # sym, params = self.current_model.symbol, self.current_model.params
        # for s in topo_sort(sym):
            # name, op_name = s.attr('name'), s.attr('op_name')
            # if op_name in ["Convolution"]:
                # print(name)
        # exit()

        _sym, _prm = quantize(
            self.current_model.symbol, self.current_model.params,
            self.features, self.precs, self.buffers, self.cfg_dict,
            self.op_input_precs, self.restore_names,
            self.shift_bits, self.softmax_lambd)
        self.current_model = Model(_sym, _prm)
        return self.current_model

    def get_output_scales(self):
        assert all([self.buffers[s.attr('name')].name == \
            BUF_TYPE_EXP for s in self.current_model])
        return [self.buffers[s.attr("name")].get() \
            for s in self.current_model]

    def get_inputs_ext(self):
        buf = self.buffers['data']
        assert buf.name == BUF_TYPE_EXP
        inputs_ext = {'data': {
            'scale': buf.get(),
            'target_bit': self.precs['data'][OUT_KEY]}}
        return inputs_ext

    def _serialize(self, dct):
        return {k: v.serialize() for k, v in dct.items()}

    def save(self, model_name, datadir="./data"):
        # pylint: disable=unbalanced-tuple-unpacking
        sym_file, params_file, ext_file = \
            utils.extend_fname(path.join(datadir, model_name), True)
        features = self._serialize(self.features)
        buffers = self._serialize(self.buffers)
        sim.save_ext(
            ext_file, self.old_names, features, self.precs, buffers, self.cfg_dict)
        self.current_model.save(sym_file, params_file)

    @staticmethod
    def _deserialize_feature(features):
        return {k: get_feature(v[0], *v[1:]) for k, v in features.items()}

    @staticmethod
    def _deserialize_buffer(buffers):
        return {k: get_buffer(v[0], *v[1:]) for k, v in buffers.items()}

    @staticmethod
    def load(model_name, datadir="./data"):
        # pylint: disable=unbalanced-tuple-unpacking
        sym_file, params_file, ext_file = \
            utils.extend_fname(path.join(datadir, model_name), True)
        mrt = MRT(Model.load(sym_file, params_file))
        mrt.old_names, features, mrt.precs, buffers, mrt.cfg_dict = \
            sim.load_ext(ext_file)
        mrt.features = MRT._deserialize_feature(features)
        mrt.buffers = MRT._deserialize_buffer(buffers)
        return mrt

def reduce_graph(model, input_shapes):
    _sym, _prm = model.symbol, model.params
    _sym, _prm = tpass.attach_input_shape(
        _sym, _prm, input_shapes)

    _sym, _prm = prepare_for_compile(_sym, _prm)
    _sym, _prm = tpass.fuse_constant(_sym, _prm)
    return Model(_sym, _prm)

def split_model(model, keys):
    symbol, params = model.symbol, model.params
    nodes = [s for s in topo_sort(symbol) if s.attr('name') in keys]
    base = nodes[0] if len(nodes) == 1 else mx.sym.Group(nodes)
    base_params = {k:params[k] for k in base.list_inputs() if k in params}

    graph = {}
    infer_shapes = tpass.infer_shape(symbol, params)
    for sym in topo_sort(symbol):
        name, op_name = sym.attr('name'), sym.attr('op_name')
        childs, attr = sutils.sym_iter(sym.get_children()), sym.list_attr()
        node = sym
        if childs is not None:
            childs = [sutils.get_node(c, graph) for c in childs]
            node = sutils.get_mxnet_op(op_name)(*childs, **attr, name=name)
        if name in keys:
            node = mx.sym.var(name, \
                shape=infer_shapes[name][sutils.get_entry_id(sym)])
        graph[name] = node
    nodes = [sutils.get_node(c, graph) for c in symbol]
    top = nodes[0] if len(nodes) == 1 else mx.sym.Group(nodes)
    top_params = {k:params[k] for k in top.list_inputs() if k in params}

    return Model(base, base_params), Model(top, top_params)
