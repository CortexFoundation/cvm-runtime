from __future__ import print_function
from cvm import utils
import libcvm
import numpy as np
import os

model_root = "/data/std_out/resnet50_mxg"

device_type = 1
device_id = 0

graph_json, graph_params = utils.load_model(
    os.path.join(model_root, "symbol"),
    os.path.join(model_root, "params"))

lib = libcvm.CVM(graph_json, graph_params, device_type, device_id)

data_path = os.path.join(model_root, "data.npy")
input_data = utils.load_np_data(data_path)
## call inference
infer_result = lib.Inference(input_data)

## compare result
output_size = lib.GetOutputLength()
output_path = model_root + "/result_0.npy"
correct_data = np.load(output_path)
for i in range(output_size):
    assert(correct_data.flatten()[i] == infer_result[i])

print ("pass")

lib.FreeModel()
