import unittest
import logging
import os
from os import path
import sys
import json

from mrt.utils import log_init
from mrt.V3.execute import run
from mrt.V3.utils import merge_cfg, override_cfg_args, get_logger
from mrt.V3.evaluate import get_evaluation_info

log_init()
yaml_files = set()
results = {}
base_dir = path.join(path.dirname(path.realpath(__file__)), "..", "..")

def _multi_validate(
    messages, base_func, data_iter, *comp_funcs,
    iter_num=10, logger=logging.getLogger(""), batch_size=16):
    log_str = "Iteration: {:3d} | " + base_func.__name__ + ": {} | "
    for func in comp_funcs:
        log_str += func.__name__ + ": {} | "
    log_str += "Total Sample: {:5d}"
    total = 0

    for i in range(iter_num):
        data, label = data_iter()
        base_acc = base_func(data, label)
        comp_acc = [func(data, label) for func in comp_funcs]
        total += batch_size

        msg = log_str.format(i, base_acc, *comp_acc, total)
        logger.info(msg)
        messages.append(msg)

def output_results():
    rfile_path = path.join(base_dir, "docs", "mrt", "V3_results.rst")
    cur_results = {}
    with open(rfile_path, "r") as f:
        for line in f:
            if not line.startswith("**") or ":" not in line:
                continue
            _, model_name, result = line.split("**")
            result = result[1:]
            cur_results[model_name] = result
    for model_name, result in results.items():
        cur_results[model_name] = result
    lines = [
        "",
        "************************",
        "MRT Quantization Results",
        "************************",
        "",
        ".. _mrt_quantization_results:",
        "",
    ]
    for model_name, result in cur_results.items():
        line = "**{0}**:{1}".format(model_name,result)
        lines.append(line)
        lines.append("")
        lines.append("")
    lines = [line+"\n" for line in lines]
    with open(rfile_path, "w") as f:
        f.writelines(lines)

def register_test_case(yaml_file_name):
    yaml_dir = path.join(base_dir, "tests", "mrt", "model_zoo")
    yaml_file_name_ext = "{}.yaml".format(yaml_file_name)
    yaml_file = path.join(yaml_dir, yaml_file_name_ext)

    if yaml_file in yaml_files:
        raise RuntimeError(
            "test case: {} already registered.".format(yaml_file))
    yaml_files.add(yaml_file)

    def test_func(self):
        base_cfg = merge_cfg(yaml_file)

        # test preparation, calibration, quantization
        argv = [
            "--common.run_evaluate", "False",
            "--common.run_compile", "False",
        ]
        cfg = override_cfg_args(base_cfg, argv)
        run(cfg)

        # test evaluation
        evalfunc, data_iter_func, quantfunc = get_evaluation_info(
            cfg.COMMON, cfg.EVALUATE)
        logger = get_logger(cfg.COMMON.VERBOSITY)
        messages = []
        with self.assertRaises(StopIteration):
            _multi_validate(
                messages, evalfunc, data_iter_func, quantfunc,
                iter_num=cfg.EVALUATE.ITER_NUM, logger=logger,
                batch_size=cfg.EVALUATE.BATCH)
        results[yaml_file_name] = messages[-1]
        output_results()

        # test compilation
        argv = [
            "--common.run_evaluate", "False",
            "--common.run_compile", "True",
            "--common.start_after", "quantize",
        ]
        cfg = override_cfg_args(base_cfg, argv)
        run(cfg)

    def wrapper(cls):
        func_name = "test_case_{}".format(yaml_file_name)
        setattr(cls, func_name, test_func)
        return cls

    return wrapper


# @register_test_case("alexnet")
# @register_test_case("densenet161")
# @register_test_case("mobilenet1_0")
# @register_test_case("mobilenetv2_1.0")
# @register_test_case("resnet18_v1")
# @register_test_case("resnet18_v1b_0.89")
# @register_test_case("resnet50_v1")
# @register_test_case("resnet50_v2")
# @register_test_case("shufflenet_v1")
# @register_test_case("squeezenet1.0")
# @register_test_case("tf_inception_v3")
# @register_test_case("vgg19")
# @register_test_case("cifar_resnet20_v1")
# @register_test_case("mnist")
# @register_test_case("qd10_resnetv1_20")
# @register_test_case("quickdraw")
# @register_test_case("ssd")
# @register_test_case("ssd_512_mobilenet1.0_voc")
# @register_test_case("trec")
# @register_test_case("yolo3_darknet53_voc")
# @register_test_case("yolo3_mobilenet1.0_voc")
class TestV3(unittest.TestCase):
    pass

if __name__ == "__main__":
    unittest.main()
