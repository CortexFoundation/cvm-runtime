import os
from os import path
import sys
import logging

import cv2

from mrt.V3.utils import (
    get_cfg_defaults, merge_cfg, override_cfg_args, get_logger)
from mrt.V3.execute import run
from mrt.V3.evaluate import get_evaluation_info
import metric
import metric_v2

def validate(result_dir, func, data_iter_func,
    logger=logging.getLogger(""), iter_num=10, batch_size=16):
    func_name = func.__name__
    func_result_dir = path.join(result_dir, func_name)
    os.makedirs(func_result_dir, exist_ok=False)
    log_str = "Iteration: {:3d} | Total Sample: {:5d}"

    total = 0
    try:
        for i in range(iter_num):
            data, label = data_iter_func()
            img0s_batch = evalfunc(data, label)
            for f, img0s in img0s_batch:
                fpath = path.join(func_result_dir, f)
                cv2.imwrite(fpath, img0s)
            total += batch_size
            msg = log_str.format(i, total)
            logger.info(msg)
    except StopIteration:
        logger.info("Iteration Stopped")

if __name__ == "__main__":
    assert len(sys.argv) >= 1 and len(sys.argv)%2 == 1, \
        "invalid length: {} of sys.argv: {}".format(
        len(sys.argv), sys.argv)
    yaml_file = path.join(
        path.dirname(path.realpath(__file__)), "yolov5s-0040.yaml")
    cfg = get_cfg_defaults()
    cfg = merge_cfg(yaml_file)
    cfg = override_cfg_args(cfg, sys.argv[1:])
    dataset_name = cfg.CALIBRATE.DATASET_NAME
    if dataset_name == "yolov5_dataset":
        run(cfg)
    elif dataset_name == "yolov5_dataset_v2":
        argv = [
            "--common.run_evaluate", "False",
            "--common.run_compile", "False",
        ]
        cfg = override_cfg_args(cfg, argv)
        run(cfg)

        logger = get_logger(cfg.COMMON.VERBOSITY)
        result_dir = path.expanduser("~/yolov5s_results/")

        evalfunc, data_iter_func, _ = get_evaluation_info(
            cfg.COMMON, cfg.EVALUATE)
        validate(
            result_dir, evalfunc, data_iter_func, logger=logger,
            iter_num=cfg.EVALUATE.ITER_NUM, batch_size=cfg.EVALUATE.BATCH)

        _, data_iter_func, quantfunc = get_evaluation_info(
            cfg.COMMON, cfg.EVALUATE)
        validate(
            result_dir, quantfunc, data_iter_func, logger=logger,
            iter_num=cfg.EVALUATE.ITER_NUM, batch_size=cfg.EVALUATE.BATCH)
    else:
        raise RuntimeError("Invalid dataset name: {}".format(dataset_name))
