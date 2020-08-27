""" MRT Interface API

    Refractor of source code, using the registry pattern.
    Rules of coding with pylint.
    Collection of hyper-parameters controller.
    Simplification of public API.
"""

import mxnet as mx
from mxnet import ndarray as nd

# import as registry pattern
from mrt.gen import tfm_ops  # pylint: disable=unused-import
from mrt import cvm_op   # pylint: disable=unused-import

from mrt.sym_utils import topo_sort
from mrt.gen.tfm_pass import sym_quantize, sym_calibrate, \
                             sym_config_infos

from mrt import transformer as tfm

# TODO: collect hyper-parameters

__all__ = ["MRT", "Model"]

class Model(tfm.Model):
    @staticmethod
    def load(symbol_file, params_file):
        """ Model load from disk. """
        symbol = mx.sym.load(symbol_file)
        params = nd.load(params_file)
        return Model(symbol, params)

    def get_mrt(self):
        return MRT(self)


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
        self.precs = {}
        if 'data' not in \
            {sym.attr('name') for sym in topo_sort(self.current_model)}:
            raise RuntimeError("please invoke `init` function first")
        self.precs['data'] = input_prec

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
            self._data, ctx=ctx, cfg_dict=self.cfg_dict)
        return self.features

    def quantize(self):
        """ Quantize the current model after calibration.

            Returns
            _______
            qmodel : Model
                The quantized model.
        """
        _sym, _prm = sym_quantize(
            self.current_model.symbol, self.current_model.params,
            self.th_dict, self.precs, self.scales, self.op_input_precs,
            self.restore_names, self.shift_bits, self.softmax_lambd)
        self.current_model = Model(_sym, _prm)
        return self.current_model
