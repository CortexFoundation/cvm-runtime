from __future__ import print_function
import libcvm 
import numpy as np

model_root = "/data/std_out/resnet50_mxg"
json_path = model_root + "/symbol"
params_path = model_root + "/params"
input_path = model_root + "/data.npy"
output_path = model_root + "/result_0.npy"

device_type = 1
device_id = 0

## read graph and param
fp = open(json_path, "r")
graph_json = fp.read().encode()
graph_strlen = len(graph_json)
print("json size = ", len(graph_json))
print(type(graph_json))

fp = open(params_path, "rb")
params_bytes = fp.read()
params_strlen = len(params_bytes)
print("params size = ", len(params_bytes))
print(type(params_bytes))

lib = libcvm.CVM()
## load model
ret = lib.LoadModel(graph_json, graph_strlen, params_bytes, params_strlen, device_type, device_id) 
assert(ret == 0)

ret, gas = lib.GetGasFromModel()
print("ops ", gas/1024/1024)

## get input size and outpu size
ret, input_size = lib.GetInputLength()
assert(ret == 0)
ret, output_size = lib.GetOutputLength()
assert(ret == 0)
input_data = np.load(input_path)
print("input size = ", input_size, len(input_data))
output_data = bytes(output_size) 
correct_data = np.load(output_path)
ret, output_type_size = lib.GetOutputTypeSize()
assert(ret == 0)
print("output type size = ", output_type_size)

## call inference
print("start inference")
ret = lib.Inference(input_data.tobytes(), input_size, output_data)
assert(ret == 0)
max_v = (1 << (output_type_size * 8 - 1))
infer_result = []
for i in range(0, output_size, output_type_size):
    int_val = int.from_bytes(output_data[i:i+output_type_size], byteorder='little')
    infer_result.append(int_val if int_val < max_v else int_val - 2 * max_v)
print("end inference")

## compare result
assert(ret == 0)
for i in range(output_size):
    assert(correct_data.flatten()[i] == infer_result[i])

print("correct")
ret = lib.FreeModel()
assert(ret == 0)
print("free model success")
