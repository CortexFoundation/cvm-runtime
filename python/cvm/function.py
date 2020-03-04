import ctypes
import os
import numpy as np

from base import _LIB, check_call

def c_str(string):
    return ctypes.c_char_p(string.encode('utf-8'))

def load_model(sym_path, prm_path):
    with open(sym_path, "r") as f:
        json_str = f.read()
    with open(prm_path, "rb") as f:
        param_bytes = f.read()
    return json_str, param_bytes

def load_np_data(data_path):
    data = np.load(data_path)
    return data.tobytes()

NetworkHandle = ctypes.c_void_p
class DEVICE_TYPE:
    CPU = 0
    GPU = 1
    FORMAL = 0

def CVMAPILoadModel(json_str, param_bytes,
                    device_type, device_id):
    net = NetworkHandle()
    check_call(_LIB().CVMAPILoadModel(
        c_str(json_str), ctypes.c_int(len(json_str)),
        ctypes.c_char_p(param_bytes), ctypes.c_int(len(param_bytes)),
        ctypes.byref(net),
        ctypes.c_int(device_type), ctypes.c_int(device_id)))
    return net

def CVMAPIFreeModel(net):
    check_call(_LIB().CVMAPIFreeModel(net))

def CVMAPIGetInputLength(net):
    size = ctypes.c_ulonglong()
    check_call(_LIB().CVMAPIGetInputLength(net, ctypes.byref(size)))
    return size.value

def CVMAPIGetInputTypeSize(net):
    size = ctypes.c_ulonglong()
    check_call(_LIB().CVMAPIGetInputTypeSize(net, ctypes.byref(size)))
    return size.value

def CVMAPIGetOutputLength(net):
    size = ctypes.c_ulonglong()
    check_call(_LIB().CVMAPIGetOutputLength(net, ctypes.byref(size)))
    return size.value

def CVMAPIGetOutputTypeSize(net):
    size = ctypes.c_ulonglong()
    check_call(_LIB().CVMAPIGetOutputTypeSize(net, ctypes.byref(size)))
    return size.value

def CVMAPIInference(net, input_data):
    osize = CVMAPIGetOutputLength(net)

    output_data = bytes(osize)
    check_call(_LIB().CVMAPIInference(
        net,
        ctypes.c_char_p(input_data), ctypes.c_int(len(input_data)),
        ctypes.c_char_p(output_data)))

    otype_size = CVMAPIGetOutputTypeSize(net)
    ret = []
    max_v = (1 << (otype_size * 8  - 1))
    for i in range(0, osize, otype_size):
        int_val = int.from_bytes(
            output_data[i:i+otype_size], byteorder='little')
        ret.append(int_val if int_val < max_v else int_val - 2 * max_v)

    return ret


# from base import CVMContext, CPU, GPU
# CVMContext(GPU).set()
# #  CVMContext(CPU).set()
# 
# # model_root = "/home/serving/tvm-cvm/data/jetson/"
# model_root = "/tmp/ssd_512_mobilenet1.0_coco_tfm/"
# # model_root = "/data/std_out/resnet50_v2"
# # odel_root = "/data/std_out/ssd"
# json, params = load_model(os.path.join(model_root, "symbol"),
#                          os.path.join(model_root, "params"))
# 
# net = CVMAPILoadModel(json, params, DEVICE_TYPE.CPU, 0)
# print(CVMAPIGetOutputLength(net))
# 
# data_path = os.path.join(model_root, "data.npy")
# data = load_np_data(data_path)
# 
# out = CVMAPIInference(net, data)
# for i in range(0, len(out), 6):
#    if out[i] == -1:
#        print ("Total object: ", i // 6)
#        break
#    print (out[i:i+6])
