""" CVM Runtime API

    This namespace wraps the python interface of c backend API. Inference methods contain model load, inference, ..., and free etc.

    We have supply two wrapper format via `ctypes` and `cython`.
"""

from ._ctypes.runtime import CVMAPILoadModel, CVMAPIFreeModel
from ._ctypes.runtime import CVMAPIGetInputLength, CVMAPIGetInputTypeSize
from ._ctypes.runtime import CVMAPIInference
from ._ctypes.runtime import CVMAPIGetOutputLength, CVMAPIGetOutputTypeSize

try:
    from ._cy3 import libcvm
except ImportError:
    pass
