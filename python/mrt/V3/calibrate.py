from yacs.config import CfgNode as CN

from mrt.transformer import Model
from mrt import dataset as ds
from mrt import conf
from mrt.V3.utils import (
    MRT_CFG, default_device_type, default_device_ids, default_batch,
    get_model_prefix, get_logger, set_batch, load_fname, save_conf,
    load_conf, check_file_existance, get_ctx)

DOC = """
CALIBRATE Stage Options:
    --calibrate.batch           Batch size for calibration.
    --calibrate.num_calib       Number of iterations for calibration.
    --calibrate.lambd           Hyperparameter for the threshold of model internal data.
    --calibrate.dataset_name    Name of the dataset chosen from "voc", "imagenet", "trec", "mnist", "coco", "quickdraw" and "cifar10".
    --calibrate.dataset_dir     Dataset root directory for specific dataset out of list above.
    --calibrate.device_type     Context type for calibration stage chosen from "cpu" or "gpu".
    --calibrate.device_ids      A comma list within square brackets specifying the context ids, eg.[0,1,2].
"""

default_num_calib = 1

MRT_CFG.CALIBRATE = CN()
MRT_CFG.CALIBRATE.BATCH = default_batch
MRT_CFG.CALIBRATE.NUM_CALIB = default_num_calib
MRT_CFG.CALIBRATE.LAMBD = None
MRT_CFG.CALIBRATE.DATASET_NAME = "imagenet"
MRT_CFG.CALIBRATE.DATASET_DIR = conf.MRT_DATASET_ROOT
MRT_CFG.CALIBRATE.DEVICE_TYPE = default_device_type
MRT_CFG.CALIBRATE.DEVICE_IDS = default_device_ids

def calibrate(cm_cfg, pass_cfg, logger=None):
    model_dir = cm_cfg.MODEL_DIR
    model_name = cm_cfg.MODEL_NAME
    verbosity = cm_cfg.VERBOSITY
    dataset_name = pass_cfg.DATASET_NAME
    dataset_dir = pass_cfg.DATASET_DIR
    device_type = pass_cfg.DEVICE_TYPE
    device_ids = pass_cfg.DEVICE_IDS
    calibrate_num = pass_cfg.NUM_CALIB
    lambd = pass_cfg.LAMBD
    batch = pass_cfg.BATCH

    model_prefix = get_model_prefix(model_dir, model_name)
    if logger is None:
        logger = get_logger(verbosity)
    conf_prep_file = model_prefix + ".prepare.conf"
    check_file_existance(conf_prep_file, logger=logger)
    conf_map = load_conf(conf_prep_file, logger=logger)

    # calibration
    if conf_map.get("split_keys", "") == "":
        sym_prep_file, prm_prep_file = load_fname(
            model_prefix, suffix="prepare")
        check_file_existance(sym_prep_file, prm_prep_file, logger=logger)
        mrt = Model.load(sym_prep_file, prm_prep_file).get_mrt()
    else:
        sym_base_file, prm_base_file = load_fname(
            model_prefix, suffix="base")
        check_file_existance(sym_base_file, prm_base_file, logger=logger)
        mrt = Model.load(sym_base_file, prm_base_file).get_mrt()
    shp = set_batch(conf_map["input_shape"], batch)
    dataset = ds.DS_REG[dataset_name](shp, root=dataset_dir)
    data_iter_func = dataset.iter_func()
    if len(device_ids) > 1:
        raise RuntimeError(
            "device ids should be an integer in calibration stage")
    ctx = get_ctx(device_type, device_ids)
    for i in range(calibrate_num):
        data, _ = data_iter_func()
        mrt.set_data(data)
        mrt.calibrate(lambd=lambd, ctx=ctx)
    mrt.save(model_name+".mrt.calibrate", datadir=model_dir)
    conf_map["dataset_name"] = dataset_name
    save_conf(model_prefix+".calibrate.conf", logger=logger, **conf_map)
    logger.info("calibrate stage finished")
