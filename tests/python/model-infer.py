from cvm.base import CVMContext, CPU, GPU, FORMAL
from cvm.function import load_model, load_np_data
from cvm.function import CVMAPILoadModel, CVMAPIInference, CVMAPIGetOutputLength
from cvm import utils

import os

CVMContext.set_global(CPU)
# CVMContext.set_global(GPU)
# CVMContext.set_global(FORMAL)

# model_root = "/home/serving/tvm-cvm/data/jetson/"
# model_root = "/tmp/ssd_512_mobilenet1.0_coco_tfm/"
# model_root = "/data/std_out/resnet50_v2"
model_root = "/data/std_out/ssd"
json, params = load_model(os.path.join(model_root, "symbol"),
                         os.path.join(model_root, "params"))

net = CVMAPILoadModel(json, params)
print(CVMAPIGetOutputLength(net))

data_path = os.path.join(model_root, "data.npy")
data = load_np_data(data_path)

out = CVMAPIInference(net, data)

# utils.classification_output(out)
utils.detection_output(out)

