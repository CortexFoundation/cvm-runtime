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

        lib_path = libinfo.find_lib_path(self.dev_type)
        self.lib_name = os.path.basename(lib_path[0])

        print(">>> Loading library({:s}) path: {:s}".format(
            self.lib_name, lib_path[0]))

        self.lib = ctypes.CDLL(lib_path[0], ctypes.RTLD_GLOBAL)

    def __enter__(self):
        return CVMContext.set_global(self.dev_type, self.dev_id)

    def __exit__(self, *args):
        return CVMContext.restore()

    @staticmethod
    def LIB():
        if CVMContext._current_context is None:
            raise Exception("It seems not to set the GlobalContext, " +
                "invoke the function `CVMContext.set_global` or " +
                "use the python gramma of `with CVMContext()`.")
        return CVMContext._current_context.lib

    @staticmethod
    def LIB_TYPE():
        if CVMContext._current_context is None:
            raise Exception("It seems not to set the GlobalContext, " +
                "invoke the function `CVMContext.set_global` or " +
                "use the python gramma of `with CVMContext()`.")
        return CVMContext._current_context.dev_type

    @staticmethod
    def LIB_NAME():
        if CVMContext._current_context is None:
            raise Exception("It seems not to set the GlobalContext, " +
                    "invoke the function `CVMContext.set_global` or " +
                    "use the python gramma of `with CVMContext()`.")
        return CVMContext._current_context.lib_name

    @staticmethod
    def set_global(device_type):
        if CVMContext._current_context is not None:
            CVMContext.restore()

        old_ctx = CVMContext._current_context
        assert old_ctx is None
        # context not change, derived from old context
        if old_ctx is not None and old_ctx.dev_type == device_type:
            return old_ctx

        new_ctx = CVMContext(device_type)
        new_ctx.old_ctx = old_ctx
        CVMContext._current_context = new_ctx
        return new_ctx

    @staticmethod
    def restore():
        lib = CVMContext.LIB()
        handle = lib._handle # obtain the SO handle
        print(">>> Freeing library({:s})".format(CVMContext.LIB_NAME()))
        ctypes.CDLL('libdl.so').dlclose(handle)
        CVMContext._current_context = None

        # curr_ctx = CVMContext._current_context
        # old_ctx = curr_ctx.old_ctx
        # if old_ctx is None:
        #     raise Exception("No context can be erase")

        # CVMContext._current_context = old_ctx
        # return old_ctx

# CVMContext.set_global(CPU)
_LIB, _LIB_NAME = CVMContext.LIB, CVMContext.LIB_NAME

class Status:
    SUCCEED = 0
    ERROR_LOGIC = 1
    ERROR_RUNTIME = 2

def check_call(ret):
    if ret != Status.SUCCEED:
        raise Exception("API called with error code: %d" % ret)
