from os import path
import os
from yacs.config import CfgNode as CN

import numpy as np

from mrt.transformer import Model, MRT
from mrt import dataset as ds
from mrt import sim_quant_helper as sim
from mrt.V3.utils import (
    MRT_CFG, default_device_type, default_device_ids, default_batch,
    get_model_prefix, get_logger, set_batch, load_fname, load_conf,
    check_file_existance)

default_dump_dir = path.expanduser("~/mrt_dump")

MRT_CFG.COMPILE = CN()
MRT_CFG.COMPILE.BATCH = 1
MRT_CFG.COMPILE.DUMP_DIR = default_dump_dir
MRT_CFG.COMPILE.DEVICE_TYPE = default_device_type
MRT_CFG.COMPILE.DEVICE_IDS = default_device_ids

def mrt_compile(cm_cfg, pass_cfg, logger=None):
    model_dir = cm_cfg.MODEL_DIR
    model_name = cm_cfg.MODEL_NAME
    verbosity = cm_cfg.VERBOSITY
    dump_dir = pass_cfg.DUMP_DIR
    device_type = pass_cfg.DEVICE_TYPE
    device_ids = pass_cfg.DEVICE_IDS
    batch = pass_cfg.BATCH

    model_prefix = get_model_prefix(model_dir, model_name)
    if logger is None:
        logger = get_logger(verbosity)
    conf_quant_file = model_prefix + ".quantize.conf"
    check_file_existance(conf_quant_file, logger=logger)
    conf_map = load_conf(conf_quant_file, logger=logger)
    if len(device_ids) > 1:
        raise RuntimeError(
            "device ids should be an integer in compilation stage")
    input_shape = conf_map["input_shape"]

    model_name_tfm = model_name + "_cvm"
    device_ids_compile = device_ids[0]
    if conf_map.get("split_keys", "") != "":
        sym_all_file, prm_all_file, ext_all_file = load_fname(
            model_prefix, suffix="all.quantize", with_ext=True)
        check_file_existance(
            sym_all_file, prm_all_file, ext_all_file, logger=logger)
        qmodel = Model.load(sym_all_file, prm_all_file)
        oscales, inputs_ext = sim.load_ext(ext_all_file)
    else:
        sym_quant_file, prm_quant_file, ext_quant_file = load_fname(
            model_prefix, suffix="mrt.quantize", with_ext=True)
        check_file_existance(
            sym_quant_file, prm_quant_file, ext_quant_file, logger=logger)
        mrt = MRT.load(model_name+".mrt.quantize", datadir=model_dir)
        oscales = mrt.get_output_scales()
        inputs_ext = mrt.get_inputs_ext()
        qmodel = mrt.current_model
    if not path.exists(dump_dir):
        os.makedirs(dump_dir, exist_ok=True)
    qmodel.to_cvm(
        model_name_tfm, datadir=dump_dir,
        input_shape=set_batch(input_shape, batch), target=device_type,
        device_ids=device_ids_compile)
    dataset = ds.DS_REG[conf_map["dataset_name"]](set_batch(input_shape, batch))
    dump_data, _ = dataset.iter_func()()
    dump_data = sim.load_real_data(
        dump_data.astype("float64"), "data", inputs_ext)
    model_root = path.join(dump_dir, model_name_tfm)
    np.save(
        path.join(model_root, "data.npy"), dump_data.astype("int8").asnumpy())
    infos = {
        "inputs_ext": inputs_ext,
        "oscales": oscales,
        "input_shapes": input_shape,
    }
    sim.save_ext(path.join(model_root, "ext"), infos)
    logger.info("compilation stage finished")
