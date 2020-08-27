import logging

import mxnet as mx
from mxnet import ndarray as nd
from os import path

# import as registry pattern
from mrt.gen import tfm_ops  # pylint: disable=unused-import
from mrt import cvm_op   # pylint: disable=unused-import

from mrt.sym_utils import topo_sort
from mrt.tfm_pass import OUT_KEY
from mrt.gen.tfm_types import get_feature, get_buffer
from mrt.gen.tfm_pass import quantize, sym_calibrate, \
                             rewrite, sym_config_infos

from mrt import transformer as tfm
from mrt import utils
from mrt import sim_quant_helper as sim
from mrt import tfm_pass as tpass

# TODO: collect hyper-parameters

__all__ = ["MRT", "Model"]

class Model(tfm.Model):
    @staticmethod
    def load(symbol_file, params_file):
        """ Model load from disk. """
        symbol = mx.sym.load(symbol_file)
        params = nd.load(params_file)
        return Model(symbol, params)

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
    _sym, _prm = tpass.fuse_constant(_sym, _prm)
    _sym, _prm = tpass.params_unique(_sym, _prm)

    return Model(_sym, _prm)


class MRT(tfm.MRT):
    def __init__(self, model, input_prec=8):
        self.old_names = model.output_names()
        self.current_model = model

        self._data = None
        self.cfg_dict = {}
        self.features = {}
        self.buffers = {}
        self.restore_names = set()

        self._op_default_input_precs()
        self.precs = {s.attr('name'):{} \
            for s in topo_sort(self.current_model)}
        if 'data' not in self.precs:
            raise RuntimeError("please invoke `init` function first")
        self.precs['data'][OUT_KEY] = input_prec

        self.softmax_lambd = 10
        self.shift_bits = 5

    def set_cfg_dict(self, cfg_dict):
        self.cfg_dict = cfg_dict

    def calibrate(self, ctx=mx.cpu(), lambd=None, old_ths=None):
        """ Calibrate the current model after setting mrt data.

            Parameters
            __________
            ctx : mxnet.context
                Context on which intermediate result would be stored,
            lambd : double
                Hyperparameter
            old_ths : dict
                Reference threshold dict could also be specified.

            Returns
            _______
            th_dict : dict
                Threshold dict of node-level output.
        """
        self.cfg_dict = sym_config_infos(
            self.current_model.symbol, self.current_model.params,
            cfg_dict=self.cfg_dict)
        self.features = sym_calibrate(
            self.current_model.symbol, self.current_model.params,
            self._data, self.cfg_dict, ctx=ctx)
        return self.features

    def quantize(self):
        """ Quantize the current model after calibration.

            Returns
            _______
            qmodel : Model
                The quantized model.
        """
        _sym, _prm = quantize(
            self.current_model.symbol, self.current_model.params,
            self.features, self.precs, self.buffers, self.cfg_dict,
            self.op_input_precs, self.restore_names,
            self.shift_bits, self.softmax_lambd)
        self.current_model = Model(_sym, _prm)
        return self.current_model

    def _serialize(self, dct):
        return {k: v.serialize() for k, v in dct.items()}

    def save(self, model_name, datadir="./data"):
        """ Save the current mrt instance into disk.
        """
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
