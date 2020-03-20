from ._ctypes.function import CVMAPILoadModel, CVMAPIFreeModel
from ._ctypes.function import CVMAPIGetInputLength, CVMAPIGetInputTypeSize
from ._ctypes.function import CVMAPIInference
from ._ctypes.function import CVMAPIGetOutputLength, CVMAPIGetOutputTypeSize

try:
    from ._cy3 import libcvm
except ImportError:
    pass
