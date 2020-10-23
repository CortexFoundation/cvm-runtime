import ctypes
import os
import numpy as np

from .. import libinfo
from .._base import check_call
from ..common import cpu, runtime_context
from .lib import _LIB

NetworkHandle = ctypes.c_void_p

def CVMAPILoadModel(json_str, param_bytes, ctx=None):
    """ Ctypes wrapper method: CVMAPILoadModel

        Load model from memory binary. The parameters loaded from
        disk could be generated via `cvm.utils.load_model` method.

        Parameters
        ==========
        json_str: bytes
            The UTF-8 encoded bytes of string type reading
            from the model json file.
        param_bytes: bytes
            The binary reading from the model params file.
        ctx: :class:`cvm.CVMContext`
            The context of model loaded into.

    """
    if ctx is None:
        ctx = cpu(0)
    ctx = runtime_context(ctx)

    net = NetworkHandle()
    if isinstance(json_str, str):
        json_str = json_str.encode("utf-8")
    check_call(_LIB.CVMAPILoadModel(
        ctypes.c_char_p(json_str), ctypes.c_int(len(json_str)),
        ctypes.c_char_p(param_bytes), ctypes.c_int(len(param_bytes)),
        ctypes.byref(net),
        ctx.device_type, ctx.device_id))
    return net

def CVMAPIFreeModel(net):
    """ Ctypes wrapper method: CVMAPIFreeModel

        Free model from memory binary.

        Parameters
        ==========
        net : ctypes.c_void_p
            The CVM model handle created by the interface :func:`cvm.runtime.CVMAPILoadModel <.CVMAPILoadModel>`.

    """
    check_call(_LIB.CVMAPIFreeModel(net))

def CVMAPIGetInputLength(net):
    """ Ctypes wrapper method: CVMAPIGetInputLengthModel

        Get the input length of the model, which can be calculated as follows:

        .. math::
            input\_length = input\_bytes * in\_size

        Where in_size depends on model input shapes. 
        For models with precision over 8, input_bytes equals to 4, otherwise 1.

        Parameters
        ==========
        net : ctypes.c_void_p
            The CVM model handle created by the interface :func:`cvm.runtime.CVMAPILoadModel <.CVMAPILoadModel>`.

    """
    size = ctypes.c_ulonglong()
    check_call(_LIB.CVMAPIGetInputLength(net, ctypes.byref(size)))
    return size.value

def CVMAPIGetInputTypeSize(net):
    """ Ctypes wrapper method: CVMAPIGetInputTypeSize

        Get the size of input shape, namely input_bytes.

        For models with precision over 8, input_bytes equals to 4, otherwise 1.

        Parameters
        ==========
        net : ctypes.c_void_p
            The CVM model handle created by the interface :func:`cvm.runtime.CVMAPILoadModel <.CVMAPILoadModel>`.

    """
    size = ctypes.c_ulonglong()
    check_call(_LIB.CVMAPIGetInputTypeSize(net, ctypes.byref(size)))
    return size.value

def CVMAPIGetOutputLength(net):
    """ Ctypes wrapper method: CVMAPIGetOutputLength

        Get the length of the output.

        Postprocess method (argmax, detection) can be specified by model definition. 
        If the postprocess method is not specified, the output will be flatten by default.

        Parameters
        ==========
        net : ctypes.c_void_p
            The CVM model handle created by the interface :func:`cvm.runtime.CVMAPILoadModel <.CVMAPILoadModel>`.

    """
    size = ctypes.c_ulonglong()
    check_call(_LIB.CVMAPIGetOutputLength(net, ctypes.byref(size)))
    return size.value

def CVMAPIGetOutputTypeSize(net):
    """ Ctypes wrapper method: CVMAPIGetOutputTypeSize

        Get the output_bytes. 
        For models with 'postprocess_method' equals to 'argmax', the output_bytes is 1; 
        for models with 'postprocess_method' equals to 'detection', the output_bytes is 4.

        Parameters
        ==========
        net : ctypes.c_void_p
            The CVM model handle created by the interface :func:`cvm.runtime.CVMAPILoadModel <.CVMAPILoadModel>`.

    """
    size = ctypes.c_ulonglong()
    check_call(_LIB.CVMAPIGetOutputTypeSize(net, ctypes.byref(size)))
    return size.value

def CVMAPIInference(net, input_data):
    """ Ctypes wrapper method: CVMAPIInference

        CVM interface for model inference. 
        Model output tensor initialization, forward network computing,output tensor disk serialization are successively performed.

        Parameters
        ==========
        net : ctypes.c_void_p
            The CVM model handle created by the interface :func:`cvm.runtime.CVMAPILoadModel <.CVMAPILoadModel>`.
        input_data : bytes
            The input image bytes.

    """
    osize = CVMAPIGetOutputLength(net)

    output_data = bytes(osize)
    check_call(_LIB.CVMAPIInference(
        net,
        ctypes.c_char_p(input_data), ctypes.c_int(len(input_data)),
        ctypes.c_char_p(output_data)))

    otype_size = CVMAPIGetOutputTypeSize(net)
    ret = []
    max_v = (1 << (otype_size * 8  - 1))
    for i in range(0, osize, otype_size):
        int_val = int.from_bytes(
            output_data[i:i+otype_size], byteorder='little')
        ret.append(int_val if int_val < max_v else int_val - 2 * max_v)

    return ret
