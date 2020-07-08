from ._ctypes.context import *

def context(dev_type, dev_id=0):
    if isinstance(dev_type, str):
        dev_type = dev_type.split()[0]
        if dev_type not in CVMContext.STR2MASK:
            raise ValueError("Unknown device type %s" % dev_type)
        dev_type = CVMContext.STR2MASK[dev_type]
    return CVMContext(dev_type, dev_id)

def cpu(dev_id=0):
    """ CVM CPU Context
    """
    return CVMContext(kDLCPU, dev_id)

def gpu(dev_id=0):
    """ CVM GPU(CUDA) Context
    """
    return CVMContext(kDLGPU, dev_id)

def formal(dev_id=0):
    """ CVM Formalization Context
    """
    return CVMContext(kDLFORMAL, dev_id)

def opencl(dev_id=0):
    """ CVM OPENCL Context(Not Well Supported)
    """
    return CVMContext(kDLOPENCL, dev_id)

RuntimeDevAPIMap = {
    kDLCPU: 0,
    kDLGPU: 1,
    kDLFORMAL: 2,
    kDLOPENCL: 3,
}

def runtime_context(ctx):
    """ Transform CVM Symbol&Graph Context into Runtime Format

        Do not invoke this function manually!!!

        CVM runtime API with c backend have an unusual device map:
        0 indicates CPU, 1 means GPU and etc. But user don't need
        to notice this, cause runtime package have process this
        anti-mind map.
    """
    return CVMContext(
        RuntimeDevAPIMap[ctx.device_type],
        ctx.device_id)
