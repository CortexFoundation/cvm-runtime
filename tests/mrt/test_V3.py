import unittest
import logging
import os
from os import path
from io import StringIO
import sys

from mrt.utils import log_init
from mrt.V3.execute import run
from mrt.V3.utils import merge_cfg, override_cfg_args
from mrt.V3.evaluate import get_evaluation_info

#  old_stdout = sys.stdout
# sys.stdout = StringIO()
#  sys.stdout = old_stdout

log_init()
yaml_files = set()
results = {}
base_dir = path.join(path.dirname(path.realpath(__file__)), "..", "..")

def _multi_validate(
    messages, base_func, data_iter, *comp_funcs, iter_num=10, batch_size=16):
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
        messages.append(msg)

def register_test_case(yaml_file_name):
    yaml_dir = path.join(base_dir, "tests", "mrt", "model_zoo")
    yaml_file_name_ext = "{}.yaml".format(yaml_file_name)
    yaml_file = path.join(yaml_dir, yaml_file_name_ext)

    if yaml_file in yaml_files:
        raise RuntimeError(
            "test case: {} already registered.".format(yaml_file))
    yaml_files.add(yaml_file)

    def test_func(self):
        # test preparation, calibration, quantization, compilation
        base_cfg = merge_cfg(yaml_file)
        argv = [
            "--common.run_evaluate", "False",
            "--common.run_compile", "True",
            "--common.verbosity", "error",
        ]
        cfg = override_cfg_args(base_cfg, argv)
        run(cfg)

        # test evaluation
        argv = [
            "--common.run_evaluate", "True",
            "--common.run_compile", "False",
            "--common.verbosity", "error",
            "--common.start_after", "quantize",
        ]
        cfg = override_cfg_args(base_cfg, argv)
        evalfunc, data_iter_func, quantfunc = get_evaluation_info(
            cfg.COMMON, cfg.EVALUATE)
        messages = []
        with self.assertRaises(StopIteration):
            _multi_validate(
                messages, evalfunc, data_iter_func, quantfunc,
                iter_num=cfg.EVALUATE.ITER_NUM, batch_size=cfg.EVALUATE.BATCH)
        results[yaml_file_name] = messages[-1]

    def wrapper(cls):
        func_name = "test_case_{}".format(yaml_file_name)
        setattr(cls, func_name, test_func)
        return cls

    return wrapper


@register_test_case("alexnet")
class TestV3(unittest.TestCase):
    def test_output_results(self):
        lines = [
            "",
            "************************",
            "MRT Quantization Results",
            "************************",
            "",
            ".. _mrt_quantization_results:",
            "",
        ]
        for k, v in results.items():
            line = "**{}**:".format(k)
            lines.append(line)
            line = "{}".format(v)
            lines.append(line)
            lines.append("")
            lines.append("")
        lines = [line+"\n" for line in lines]
        rfile_path = path.join(base_dir, "docs", "mrt", "V3_results.rst")
        with open(rfile_path, "w") as f:
            f.writelines(lines)

if __name__ == "__main__":
    unittest.main()
