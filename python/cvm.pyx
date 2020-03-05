cimport ccvm
def LoadModel(bytes graph_json, int graph_strlen,
    bytes param_bytes, int param_strlen,
    int device_type, int device_id):
    cdef void *cnetwork
    print("start load model")
    ret = ccvm.CVMAPILoadModel(graph_json, graph_strlen, param_bytes, param_strlen, &cnetwork, device_type, device_id)
    print("end load model")
    return ret, <object>cnetwork

def FreeModel(network):
    return ccvm.CVMAPIFreeModel(<void*>network)

def Inference(network,
    char *input_data, int input_len,
    char *output_data):
    return ccvm.CVMAPIInference(<void*>network, input_data, input_len, output_data)

def GetVersion(network, char *version):
    return ccvm.CVMAPIGetVersion(<void*>network, version)

def GetPreprocessMethod(network, char *method):
    return ccvm.CVMAPIGetPreprocessMethod(<void*>network, method)

def GetInputLength(network):
    cdef unsigned long long[1] csize
    ret = ccvm.CVMAPIGetInputLength(<void*>network, csize)
    return ret, csize[0]

def GetOutputLength(network):
    cdef unsigned long long[1] csize
    ret = ccvm.CVMAPIGetOutputLength(<void*>network, csize)
    return ret, csize[0]

def GetInputTypeSize(network):
    cdef unsigned long long[1] csize
    ret = ccvm.CVMAPIGetInputTypeSize(<void*>network, csize)
    return ret, csize[0]

def GetOutputTypeSize(network):
    cdef unsigned long long[1] csize
    ret = ccvm.CVMAPIGetOutputTypeSize(<void*>network, csize)
    return ret, csize[0]

def GetStorageSize(network):
    cdef unsigned long long[1] cgas
    ret = ccvm.CVMAPIGetStorageSize(<void*>network, cgas)
    return ret, cgas[0]

def GetGasFromModel(network):
    cdef unsigned long long[1] cgas
    ret = ccvm.CVMAPIGetGasFromModel(<void*>network, cgas)
    return ret, cgas[0]

def GetGasFromGraphFile(const char *graph_json):
    cdef unsigned long long[1] cgas
    ret = ccvm.CVMAPIGetGasFromGraphFile(graph_json, cgas)
    return ret, cgas[0]
