cimport ccvm
cdef class CVM:
    cdef void *network
    def LoadModel(self, bytes graph_json, int graph_strlen,
            bytes param_bytes, int param_strlen,
            int device_type, int device_id):
        print("start load model")
        ret = ccvm.CVMAPILoadModel(graph_json, graph_strlen, param_bytes, param_strlen, &self.network, device_type, device_id)
        print("end load model")
        return ret

    def FreeModel(self):
        return ccvm.CVMAPIFreeModel(self.network)

    def Inference(self, char *input_data, int input_len, char *output_data):
        return ccvm.CVMAPIInference(self.network, input_data, input_len, output_data)

    def GetVersion(self, char *version):
        return ccvm.CVMAPIGetVersion(self.network, version)

    def GetPreprocessMethod(self, char *method):
        return ccvm.CVMAPIGetPreprocessMethod(self.network, method)

    def GetInputLength(self):
        cdef unsigned long long[1] csize
        ret = ccvm.CVMAPIGetInputLength(self.network, csize)
        return ret, csize[0]

    def GetOutputLength(self):
        cdef unsigned long long[1] csize
        ret = ccvm.CVMAPIGetOutputLength(self.network, csize)
        return ret, csize[0]

    def GetInputTypeSize(self):
        cdef unsigned long long[1] csize
        ret = ccvm.CVMAPIGetInputTypeSize(self.network, csize)
        return ret, csize[0]

    def GetOutputTypeSize(self):
        cdef unsigned long long[1] csize
        ret = ccvm.CVMAPIGetOutputTypeSize(self.network, csize)
        return ret, csize[0]

    def GetStorageSize(self):
        cdef unsigned long long[1] cgas
        ret = ccvm.CVMAPIGetStorageSize(self.network, cgas)
        return ret, cgas[0]

    def GetGasFromModel(self):
        cdef unsigned long long[1] cgas
        ret = ccvm.CVMAPIGetGasFromModel(self.network, cgas)
        return ret, cgas[0]

    def GetGasFromGraphFile(const char *graph_json):
        cdef unsigned long long[1] cgas
        ret = ccvm.CVMAPIGetGasFromGraphFile(graph_json, cgas)
        return ret, cgas[0]
