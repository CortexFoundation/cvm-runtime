from os import path
from yacs.config import CfgNode as CN

from mrt.gluon_zoo import save_model
from mrt.transformer import Model
from mrt.V3.utils import (
    MRT_CFG, default_device_type, default_device_ids,
    get_model_prefix, get_logger, set_batch, load_fname, save_conf,
    get_ctx, parser, update_dest2yaml)

default_input_shape = [-1, 3, 224, 224]

MRT_CFG.PREPARE= CN()
MRT_CFG.PREPARE.DEVICE_TYPE = default_device_type
MRT_CFG.PREPARE.DEVICE_IDS = default_device_ids
MRT_CFG.PREPARE.INPUT_SHAPE = default_input_shape
MRT_CFG.PREPARE.SPLIT_KEYS = []

_pname = "PREPARE"
update_dest2yaml({
    parser.add_argument(
        "--device-type-prepare", type=str, choices=["cpu", "gpu"],
        default=default_device_type).dest: (_pname, "DEVICE_TYPE"),
    parser.add_argument(
        "--device-ids-prepare", type=int, nargs="+",
        default=default_device_ids).dest: (_pname, "DEVICE_IDS"),
    parser.add_argument(
        "--input-shape", type=int, nargs="+",
        default=default_input_shape).dest: (_pname, "INPUT_SHAPE"),
    parser.add_argument(
        "--split-keys", type=str, nargs="+",
        default=[]).dest: (_pname, "SPLIT_KEYS"),
})

def prepare(cm_cfg, pass_cfg):
    model_dir = cm_cfg.MODEL_DIR
    model_name = cm_cfg.MODEL_NAME
    verbosity = cm_cfg.VERBOSITY
    device_type = pass_cfg.DEVICE_TYPE
    device_ids = pass_cfg.DEVICE_IDS
    input_shape = pass_cfg.INPUT_SHAPE
    split_keys = pass_cfg.SPLIT_KEYS

    model_prefix = get_model_prefix(model_dir, model_name)
    logger = get_logger(verbosity)
    conf_prep_file = model_prefix + ".prepare.conf"
    conf_map = {}

    # preparation
    sym_path, prm_path = load_fname(model_prefix)
    if not path.exists(sym_path) or not path.exists(prm_path):
        save_model(
            model_name, data_dir=model_dir,
            ctx=get_ctx(device_type, device_ids))
    model = Model.load(sym_path, prm_path)
    model.prepare(set_batch(input_shape, 1))
    sym_prep_file, prm_prep_file = load_fname(
        model_prefix, suffix="prepare")
    model.save(sym_prep_file, prm_prep_file)
    conf_map["input_shape"] = input_shape
    save_conf(conf_prep_file, logger=logger, **conf_map)
    logger.info("preparation stage finihed")

    # model splitting
    if split_keys:
        sym_top_file, prm_top_file = load_fname(model_prefix, suffix='top')
        sym_base_file, prm_base_file = load_fname(
            model_prefix, suffix="base")
        base, top = model.split(split_keys)
        top.save(sym_top_file, prm_top_file)
        base.save(sym_base_file, prm_base_file)
        conf_map["split_keys"] = split_keys
        save_conf(conf_prep_file, logger=logger, **conf_map)
        logger.info("model splitting finished")
    else:
        logger.info("model splitting skipped")
