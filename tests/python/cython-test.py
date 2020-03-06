from __future__ import print_function
from cvm import utils
import libcvm 
import numpy as np
import os

model_root = "/data/std_out/resnet50_mxg"

device_type = 1
device_id = 0

graph_json, graph_params = utils.load_model(os.path.join(model_root, "symbol"), os.path.join(model_root, "params"))

lib = libcvm.CVM()
## load model
ret = lib.LoadModel(graph_json.encode(), graph_params, device_type, device_id) 
assert(ret == 0)

data_path = os.path.join(model_root, "data.npy")
input_data = utils.load_np_data(data_path)
## call inference
ret, infer_result = lib.Inference(input_data)
assert(ret == 0)


## compare result
ret, output_size = lib.GetOutputLength()
assert(ret == 0)
output_data = bytes(output_size) 
output_path = model_root + "/result_0.npy"
correct_data = np.load(output_path)
ret, output_type_size = lib.GetOutputTypeSize()
assert(ret == 0)
assert(ret == 0)
for i in range(output_size):
    assert(correct_data.flatten()[i] == infer_result[i])

ret = lib.FreeModel()
assert(ret == 0)
