import cvm
from cvm.runtime import CVMAPILoadModel, CVMAPIInference
from cvm.runtime import CVMAPIGetOutputLength, CVMAPIFreeModel
from cvm import utils

import os
import time

ctx = cvm.cpu()

# model_root = "/home/serving/tvm-cvm/data/jetson/"
model_root = "/data/std_out/ssd_512_mobilenet1.0_coco_tfm/"
#  model_root = "/tmp/resnet18_v1_tfm/"
# model_root = "/data/mrt/ssd_512_mobilenet1.0_voc_tfm"
#  model_root = "/data/std_out/cvm_mnist/"
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

iter_num = 1
start = time.time()
for i in range(iter_num):
    out = CVMAPIInference(net, data)
    # utils.classification_output(out)
end = time.time()
print ("Infer Time: ", (end - start) * 1e3 / iter_num, " ms")

CVMAPIFreeModel(net)

utils.detection_output(out)

