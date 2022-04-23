import argparse
from os import path
import os

import gluoncv as cv

model_names_quantized_default = []

def get_model_names_quantized(fpath, model_names_quantized):
    for fname in os.listdir(fpath):
        nfpath = path.join(fpath, fname)
        if path.isdir(nfpath):
            get_model_names_quantized(nfpath, model_names_quantized)
        else:
            model_names_quantized.append(path.splitext(fname)[0])

dir_path = os.path.dirname(os.path.realpath(__file__))
model_zoo = path.join(dir_path, "model_zoo")
get_model_names_quantized(model_zoo, model_names_quantized_default)
parser = argparse.ArgumentParser("")
parser.add_argument("-p", "--prefixes", nargs="*", type=str, default=[])
parser.add_argument("-sq", "--show-quantized", action="store_true")
parser.add_argument("-sqo", "--show-quantized-only", action="store_true")
parser.add_argument("-mnq", "--model-names-quantized", nargs="*", type=str, default=model_names_quantized_default)

if __name__ == "__main__":
    args = parser.parse_args()
    model_names_quantized = args.model_names_quantized
    if args.show_quantized_only:
        for model_name in model_names_quantized:
            print(model_name)
    else:
        prefixes = set(args.prefixes)
        show_quantized = args.show_quantized
        supported_models = set(cv.model_zoo.get_model_list())
        for model_name in cv.model_zoo.pretrained_model_list():
            if model_name not in supported_models:
                continue
            if not show_quantized and model_name in model_names_quantized:
                continue
            if prefixes:
                for prefix in prefixes:
                    if model_name.startswith(prefix):
                        print(model_name)
            else:
                print(model_name)
