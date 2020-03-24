from cvm.base import CVMContext, CPU, GPU, FORMAL
from cvm.dll import CVMAPILoadModel, CVMAPIInference
from cvm.dll import CVMAPIGetOutputLength, CVMAPIFreeModel
from cvm import utils

import os
import time

# CVMContext.set_global(CPU)
CVMContext.set_global(GPU)
# CVMContext.set_global(FORMAL)

# model_root = "/home/serving/tvm-cvm/data/jetson/"
model_root = "/data/std_out/ssd_512_mobilenet1.0_coco_tfm/"
#  model_root = "/data/wlt/resnet18_v1_tfm"
#  model_root = "/data/std_out/resnet50_v2"
# model_root = "/data/std_out/ssd"
# model_root = "/data/std_out/resnet50_mxg/"
json, params = utils.load_model(
        os.path.join(model_root, "symbol"),
        os.path.join(model_root, "params"))

net = CVMAPILoadModel(json, params)
print(CVMAPIGetOutputLength(net))

data_path = os.path.join(model_root, "data.npy")
data = utils.load_np_data(data_path)

iter_num = 10
start = time.time()
for i in range(iter_num):
    out = CVMAPIInference(net, data)
end = time.time()
print ("Infer Time: ", (end - start) / iter_num)

CVMAPIFreeModel(net)

utils.classification_output(out)
# utils.detection_output(out)

