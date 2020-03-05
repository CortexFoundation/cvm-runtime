from __future__ import print_function
import cvm
import numpy as np

model_root = "/data/std_out/resnet50_mxg"
json_path = model_root + "/symbol"
params_path = model_root + "/params"
input_path = model_root + "/data.npy"
output_path = model_root + "/result_0.npy"

device_type = 0
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

## load model
ret, network = cvm.LoadModel(graph_json, graph_strlen, params_bytes, params_strlen, device_type, device_id) 
assert(ret == 0)

ret, gas = cvm.GetGasFromModel(network)
print("ops ", gas/1024/1024)

## get input size and outpu size
ret, input_size = cvm.GetInputLength(network)
assert(ret == 0)
ret, output_size = cvm.GetOutputLength(network)
assert(ret == 0)
input_data = np.load(input_path)
print("input size = ", input_size, len(input_data))
output_data = bytes(output_size) 
correct_data = np.load(output_path)

## call inference
print("start inference")
ret = cvm.Inference(network, input_data.tobytes(), input_size, output_data)
print("end inference")

## compare result
assert(ret == 0)
#assert(correct_data.tobytes(), output_data)
#correct_bytes = correct_data.tobytes()
#for i in range(output_size):
#    if correct_bytes[i] != output_data[i]:
#        print(i, correct_bytes[i], output_data[i])
#    assert(correct_bytes[i] == output_data[i])
#


cvm.FreeModel(network)
