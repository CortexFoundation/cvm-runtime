import os
import ctypes


# device type name
CPU = 0
GPU = 1
FORMAL = 2

def DeviceName(device_type):
    if device_type == CPU:
        return "cpu"
    elif device_type == GPU:
        return "gpu"
    elif device_type == FORMAL:
        return "formal"

    raise RuntimeError("Unknown device type: %d" % device_type)

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
    def DEV_TYPE():
        return CVMContext._current_context.dev_type

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

CVMContext.set_global(CPU)

class Status:
    SUCCEED = 0
    ERROR_LOGIC = 1
    ERROR_RUNTIME = 2

def check_call(ret):
    if ret != Status.SUCCEED:
        raise Exception("API called with error code: %d" % ret)
