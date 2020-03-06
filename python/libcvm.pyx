cimport ccvm
import numpy as np
cdef class CVM:
    cdef void *network
    def LoadModel(self, bytes graph_json, bytes param_bytes, int device_type, int device_id):
        ret = ccvm.CVMAPILoadModel(graph_json, len(graph_json), param_bytes, len(param_bytes), &self.network, device_type, device_id)
        return ret

    def FreeModel(self):
        return ccvm.CVMAPIFreeModel(self.network)

    def Inference(self, char *input_data):
        ret, input_size = self.GetInputLength()
        ret, output_size = self.GetOutputLength()
        output_data = bytes(output_size)
        ret, output_type_size = self.GetOutputTypeSize()

        ret = ccvm.CVMAPIInference(self.network, input_data, input_size, output_data)

        max_v = (1 << (output_type_size * 8 - 1))
        infer_result = []
        for i in range(0, output_size, output_type_size):
            int_val = int.from_bytes(output_data[i:i+output_type_size], byteorder='little')
            infer_result.append(int_val if int_val < max_v else int_val - 2 * max_v)
        return ret, infer_result

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
