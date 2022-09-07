# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------
import os
import onnx
import onnx.numpy_helper
import struct
import logging
import numpy as np

from pathlib import Path

from onnx import onnx_pb as onnx_proto
from onnxruntime import SessionOptions, InferenceSession, GraphOptimizationLevel

from .quant_utils import QuantizationMode, QuantizedValueType, QuantizedInitializer, QuantizedValue
from .quant_utils import find_by_name, get_elem_index, get_mul_node, generate_identified_filename, attribute_to_kwarg
from .quant_utils import QuantType, QuantFormat

from .registry import QLinearOpsRegistry, IntegerOpsRegistry

from .onnx_model import ONNXModel
from .quantizer import YAMRTQuantizer
from .qdq_quantizer import QDQQuantizer
from .calibrate import CalibrationDataReader, get_calibrator, CalibrationMethod


def optimize_model(model_path: Path):
    '''
        Generate model that applies graph optimization (constant folding,etc.)
        parameter model_path: path to the original onnx model
        return: optimized onnx model
    '''
    opt_model_path = generate_identified_filename(model_path, "-opt")
    sess_option = SessionOptions()
    sess_option.optimized_model_filepath = opt_model_path.as_posix()
    sess_option.graph_optimization_level = GraphOptimizationLevel.ORT_ENABLE_BASIC
    _ = InferenceSession(model_path.as_posix(), sess_option, providers=['CPUExecutionProvider'])
    optimized_model = onnx.load(opt_model_path.as_posix())
    return optimized_model


def load_model(model_path: Path, optimize=True):
    if optimize:
        #optimize the original model
        onnx_model = ONNXModel(optimize_model(Path(model_path)))
        # to support GEMM
        onnx_model.replace_gemm_with_matmul()
        return onnx_model.model

    return onnx.load(Path(model_path))


def quantize(model,
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
    if nbits == 32 or nbits == 31:
        mode = quantization_mode
        copy_model = onnx_proto.ModelProto()
        copy_model.CopyFrom(model)

        if not op_types_to_quantize or len(op_types_to_quantize) == 0:
            op_types_to_quantize = list(QLinearOpsRegistry.keys()) if static else list(IntegerOpsRegistry.keys())

        quantizer = YAMRTQuantizer(copy_model, per_channel, nbits == 31, mode, static, symmetric_weight,
                                  symmetric_activation, quantization_params, nodes_to_quantize, nodes_to_exclude,
                                  op_types_to_quantize)

        quantizer.quantize_model()
        return quantizer.model.model
    else:
        raise ValueError('Only 8 and 7 bit quantization is currently supported')


def quantize_static(model_input,
                    model_output,
                    calibration_data_reader: CalibrationDataReader,
                    quant_format=QuantFormat.QOperator,
                    op_types_to_quantize=[],
                    per_channel=False,
                    reduce_range=False,
                    activation_type=QuantType.QUInt32,
                    weight_type=QuantType.QUInt32,
                    nodes_to_quantize=[],
                    nodes_to_exclude=[],
                    optimize_model=True,
                    use_external_data_format=False,
                    calibrate_method=CalibrationMethod.MinMax,
                    extra_options = {}):

    mode = QuantizationMode.QLinearOps

    if not op_types_to_quantize or len(op_types_to_quantize) == 0:
        op_types_to_quantize = list(QLinearOpsRegistry.keys())

    model = load_model(Path(model_input), optimize_model)

    calibrator = get_calibrator(model, op_types_to_quantize, calibrate_method=calibrate_method)
    calibrator.collect_data(calibration_data_reader)
    tensors_range = calibrator.compute_range()

    if quant_format is QuantFormat.QOperator:
        quantizer = YAMRTQuantizer(
            model,
            per_channel,
            reduce_range,
            mode,
            True,  # static
            weight_type,
            activation_type,
            tensors_range,
            nodes_to_quantize,
            nodes_to_exclude,
            op_types_to_quantize,
            extra_options)
    else:
        quantizer = QDQQuantizer(
            model,
            per_channel,
            reduce_range,
            mode,
            True,  # static
            weight_type,
            activation_type,
            tensors_range,
            nodes_to_quantize,
            nodes_to_exclude,
            op_types_to_quantize,
            extra_options)

    quantizer.quantize_model()
    quantizer.model.save_model_to_file(model_output, use_external_data_format)


def quantize_dynamic(model_input: Path,
                     model_output: Path,
                     op_types_to_quantize=[],
                     per_channel=False,
                     reduce_range=False,
                     activation_type=QuantType.QUInt32,
                     weight_type=QuantType.QUInt32,
                     nodes_to_quantize=[],
                     nodes_to_exclude=[],
                     optimize_model=True,
                     use_external_data_format=False,
                     extra_options = { }):

    mode = QuantizationMode.IntegerOps

    if not op_types_to_quantize or len(op_types_to_quantize) == 0:
        op_types_to_quantize = list(IntegerOpsRegistry.keys())

    model = load_model(Path(model_input), optimize_model)
    quantizer = YAMRTQuantizer(
        model,
        per_channel,
        reduce_range,
        mode,
        False,  #static
        weight_type,
        activation_type,
        None,
        nodes_to_quantize,
        nodes_to_exclude,
        op_types_to_quantize,
        extra_options)

    quantizer.quantize_model()
    quantizer.model.save_model_to_file(model_output, use_external_data_format)


def quantize_qat(model_input: Path,
                 model_output: Path,
                 op_types_to_quantize=[],
                 per_channel=False,
                 reduce_range=False,
                 activation_type=QuantType.QUInt32,
                 weight_type=QuantType.QUInt32,
                 nodes_to_quantize=[],
                 nodes_to_exclude=[],
                 use_external_data_format=False):
    mode = QuantizationMode.IntegerOps

    #optimize the original model
    optimized_model = optimize_model(Path(model_input))

    if not op_types_to_quantize or len(op_types_to_quantize) == 0:
        op_types_to_quantize = list(IntegerOpsRegistry.keys())

    quantizer = YAMRTQuantizer(
        optimized_model,
        per_channel,
        reduce_range,
        mode,
        False,  #static
        weight_type,
        activation_type,
        None,
        nodes_to_quantize,
        nodes_to_exclude,
        op_types_to_quantize)

    quantizer.quantize_model()
    quantizer.model.save_model_to_file(model_output, use_external_data_format)
