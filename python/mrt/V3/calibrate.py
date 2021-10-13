from yacs.config import CfgNode as CN

from mrt import conf
from mrt.V3.utils import (
    MRT_CFG, default_device_type, default_device_ids, default_batch,
    get_model_prefix, get_logger, set_batch, load_fname, save_conf,
    load_conf, check_file_existance, get_ctx)

MRT_CFG.CALIBRATION = CN()
MRT_CFG.CALIBRATION.BATCH = default_batch,
MRT_CFG.CALIBRATION.NUM_CALIB = 1,
MRT_CFG.CALIBRATION.LAMBD = None,
MRT_CFG.CALIBRATION.DATASET_NAME = "imagenet",
MRT_CFG.CALIBRATION.DATASET_DIR = conf.MRT_DATASET_ROOT,
MRT_CFG.CALIBRATION.DEVICE_TYPE = default_device_type,
MRT_CFG.CALIBRATION.DEVICE_IDS = default_device_ids,

def calibrate(
    model_dir, model_name, verbosity, dataset_name, dataset_dir,
    device_type, device_ids, calibrate_num, lambd, batch=default_batch):
    model_prefix = get_model_prefix(model_dir, model_name)
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

def yaml_calibrate():
    CM = MRT_CFG.COMMON
    CN = MRT_CFG.CALIBRATE
    calibrate(
        CM.MODEL_DIR, CM.MODEL_NAME, CM.VERBOSITY, CN.DATASET_NAME,
        CN.DATASET_DIR, CN.DEVICE_TYPE, CN.DEVICE_IDS, CN.NUM_CALIB,
        CN.LAMBD, batch=CN.BATCH)
