import os

MRT_MODEL_ROOT = os.path.expanduser("~/mrt_model")

if not os.path.exists(MRT_MODEL_ROOT):
    os.mkdir(MRT_MODEL_ROOT)
