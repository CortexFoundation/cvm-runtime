import os
import numpy as np
from tempfile import TemporaryDirectory
import onnx
from onnx import checker
from .onnx2mx import import_model as import_onnx_model
from onnxsim import simplify
from .quant import quantize,QuantizationMode
# For mxnet
import mxnet as mx

class ModelHandler(object):
    """ Wrapper of Model, design with user-friendly model API.
    """
    def __init__(self):
        self.sym = None
        self.arg = None
        self.aux = None

    def model(self):
        raise NotImplementedError

    def quant(self,
             per_channel=False,
             nbits=32,
             quantization_mode=QuantizationMode.IntegerOps,
             static=False,
             force_fusions=False,
             symmetric_activation=False,
             symmetric_weight=False,
             quantization_params=None,
             nodes_to_quantize=None,
             nodes_to_exclude=None,
             op_types_to_quantize=[]):
        if not isinstance(self.model, onnx.ModelProto):
            raise RuntimeError
        model = quantize(self.model,
                            per_channel,
                            nbits,
                            quantization_mode,
                            static,
                            force_fusions,
                            symmetric_activation,
                            symmetric_weight,
                            quantization_params,
                            nodes_to_quantize,
                            nodes_to_exclude,
                            op_types_to_quantize)
        assert(model!=None)
        self.model = model

    def _build_mxnet(self):
        with TemporaryDirectory() as tmpdir:
            tmp_path = os.path.join(tmpdir, "tmp.onnx")
            onnx.save(self.model, tmp_path)
            self.sym, self.arg, self.aux = import_onnx_model(tmp_path)

    def build(self):
        self._build_mxnet()
        # TODO mxnet to cvm

class MxnetModelHandler(ModelHandler):
    """ Wrapper of Mxnet Model, design with user-friendly model API.
    """
    def __init__(self):
        super(MxnetModelHandler, self).__init__()
        self.model = None

    def load(self, sym, params, in_shapes, in_types, dynamic=False, dynamic_input_shapes=None):
        with TemporaryDirectory() as tmpdir:
            tmp_path = os.path.join(tmpdir, "tmp.onnx")
            mx.onnx.export_model(sym, params, in_shapes, in_types, tmp_path,
                                            dynamic=dynamic, dynamic_input_shapes=dynamic_input_shapes, run_shape_inference=True)
            self.model = onnx.load_model(tmp_path)
            checker.check_graph(self.model.graph)
            model_simp, check = simplify(self.model)
            if (check):
                self.model = model_simp
            else:
                print("Fail to simplify model")