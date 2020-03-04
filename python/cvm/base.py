import os
import ctypes

import libinfo

CPU     = "cpu"
GPU     = "gpu"
FORMAL  = "formal"
class CVMContext:
    current_context = None

    def __init__(self, device_type, device_id=0):
        self.dev_type = device_type
        self.dev_id = device_id
        self.old_ctx = None

        lib_path = libinfo.find_lib_path()
        self.lib = ctypes.CDLL(lib_path[0], ctypes.RTLD_GLOBAL)
        self.lib_name = os.path.basename(lib_path[0])

    def __enter__(self):
        return self.set()

    def __exit__(self, *args):
        return self.clear()

    def set(self):
        if self.old_ctx != self:
            self.old_ctx = CVMContext.current_context
            CVMContext.current_context = self
        return self

    def clear(self):
        if self.old_ctx is None:
            raise Exception("No context can be erase")

        CVMContext.current_context = self.old_ctx

    @staticmethod
    def LIB():
        return CVMContext.current_context.lib

    @staticmethod
    def LIB_NAME():
        return CVMContext.current_context.lib_name

def set_global_context(device_type, device_id=0):
    CVMContext.current_context = CVMContext(device_type, device_id)

set_global_context(CPU)
_LIB, _LIB_NAME = CVMContext.LIB, CVMContext.LIB_NAME

class Status:
    SUCCEED = 0
    ERROR_LOGIC = 1
    ERROR_RUNTIME = 2

def check_call(ret):
    if ret != Status.SUCCEED:
        raise Exception("API called with error code: %d" % ret)
