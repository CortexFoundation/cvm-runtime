import os

MRT_MODEL_ROOT = os.path.expanduser("~/mrt_model")

if not os.path.exists(MRT_MODEL_ROOT):
    os.makedirs(MRT_MODEL_ROOT)

MRT_DATASET_ROOT = os.path.expanduser("~/.mxnet/datasets")

if not os.path.exists(MRT_DATASET_ROOT):
    os.makedirs(MRT_DATASET_ROOT)

YAML_ROOT = os.path.expanduser("~/mrt_yaml_root")
os.makedirs(YAML_ROOT, exist_ok=True)
