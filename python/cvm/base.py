import os
import ctypes

from . import libinfo

# device type name
CPU     = "cpu"
GPU     = "gpu"
FORMAL  = "formal"


class CVMContext:
    _current_context = None

    def __init__(self, device_type):
        self.dev_type = device_type
        self.old_ctx = None

    def __enter__(self):
        return CVMContext.set_global(self.dev_type, self.dev_id)

    def __exit__(self, *args):
        return CVMContext.restore()

    @staticmethod
    def LIB():
        return CVMContext._current_context._lib

    @staticmethod
    def LIB_TYPE():
        return CVMContext._current_context.dev_type

    @staticmethod
    def LIB_NAME():
        return CVMContext._current_context.lib_name

    @staticmethod
    def set_global(device_type):
        old_ctx = CVMContext._current_context
        # context not change, derived from old context
        if old_ctx is not None and old_ctx.dev_type == device_type:
            return old_ctx

        new_ctx = CVMContext(device_type)
        new_ctx.old_ctx = old_ctx
        CVMContext._current_context = new_ctx
        return new_ctx

    @staticmethod
    def restore():
        curr_ctx = CVMContext._current_context
        old_ctx = curr_ctx.old_ctx
        if old_ctx is None:
           raise Exception("No context can be erase")

        CVMContext._current_context = old_ctx
        return old_ctx

_LIB_PATH = libinfo.find_lib_path()
_LIB_NAME = os.path.basename(_LIB_PATH[0])
_LIB = ctypes.CDLL(_LIB_PATH[0], ctypes.RTLD_GLOBAL)

CVMContext.set_global(CPU)

class Status:
    SUCCEED = 0
    ERROR_LOGIC = 1
    ERROR_RUNTIME = 2

def check_call(ret):
    if ret != Status.SUCCEED:
        raise Exception("API called with error code: %d" % ret)
