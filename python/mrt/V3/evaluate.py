from yacs.config import CfgNode as CN
import logging

import mxnet as mx
from mxnet import gluon, ndarray as nd

from mrt.transformer import Model, MRT, reduce_graph
from mrt import dataset as ds
from mrt import utils
from mrt import sim_quant_helper as sim
from mrt.V3.utils import (
    MRT_CFG, get_model_prefix, get_logger, set_batch, load_fname, load_conf,
    check_file_existance, get_ctx, get_batch_axis)

DOC = """
EVALUATE Stage Options:
    --evaluate.batch            Batch size for evaluation.
    --evaluate.device_type      Context type for evaluation stage chosen from "cpu" or "gpu".
    --evaluate.device_ids       A comma list within square brackets specifying the context ids, eg.[0,1,2].
    --evaluate.iter_num         Number of evaluating iteration steps.
"""

MRT_CFG.EVALUATE = CN()
MRT_CFG.EVALUATE.BATCH = None
MRT_CFG.EVALUATE.DEVICE_TYPE = None
MRT_CFG.EVALUATE.DEVICE_IDS = None
MRT_CFG.EVALUATE.ITER_NUM = 10

def evaluate(cm_cfg, pass_cfg, logger=None):
    model_dir = cm_cfg.MODEL_DIR
    model_name = cm_cfg.MODEL_NAME
    verbosity = cm_cfg.VERBOSITY
    device_type = pass_cfg.DEVICE_TYPE
    device_ids = pass_cfg.DEVICE_IDS
    iter_num = pass_cfg.ITER_NUM
    batch = pass_cfg.BATCH
    if batch is None:
        batch = cm_cfg.BATCH
    if device_type is None:
        device_type = cm_cfg.DEVICE_TYPE
    if device_ids is None:
        device_ids = cm_cfg.DEVICE_IDS

    model_prefix = get_model_prefix(model_dir, model_name)
    if logger is None:
        logger = get_logger(verbosity)
    conf_quant_file = model_prefix + ".quantize.conf"
    check_file_existance(conf_quant_file, logger=logger)
    conf_map = load_conf(conf_quant_file, logger=logger)
    ctx = get_ctx(device_type, device_ids)
    if isinstance(ctx, mx.Context):
        ctx = [ctx]

    # forward function for the orginal model
    omodel = Model.load(*load_fname(model_prefix))
    graph = omodel.to_graph(ctx=ctx)
    dataset_name = conf_map["dataset_name"]
    input_shape = conf_map["input_shape"]
    dataset = ds.DS_REG[dataset_name](set_batch(input_shape, batch))
    data_iter_func = dataset.iter_func()
    metric = dataset.metrics()
    baxis = get_batch_axis(input_shape)
    olen = len(omodel.symbol)

    def forward(net, data, ctx):
        """ Multiple xpu run support.
        """
        data = gluon.utils.split_and_load(
            data, ctx_list=ctx, batch_axis=baxis, even_split=False)
        outs = [net(d) for d in data]
        if olen == 1:
            outs = nd.concatenate(outs)
        else:
            outs = [nd.concatenate([outs[i][j] \
                for i in range(len(outs))]) for j in range(olen)]
        return outs

    def evalfunc(data, label):
        outs = forward(graph, data, ctx=ctx)
        acc = dataset.validate(metric, outs, label)
        return acc

    # forward function for the quantized model
    num_xpus = len(ctx)
    if batch % num_xpus:
        raise RuntimeError("Batch must be divisible by the number of xpus")
    split_batch = batch // num_xpus
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
    rqmodel = reduce_graph(qmodel, {
        'data': set_batch(input_shape, split_batch)})
    qgraph = rqmodel.to_graph(ctx=ctx)
    qmetric = dataset.metrics()

    def quantize(data, label):
        data = sim.load_real_data(data, 'data', inputs_ext)
        outs = forward(qgraph, data, ctx)
        outs = outs / oscales[0] if olen == 1 \
            else [(t / oscales[i]) for i, t in enumerate(outs)]
        acc = dataset.validate(qmetric, outs, label)
        return acc

    # evaluate
    if iter_num > 0:
        logger.info("Validating...")
        utils.multi_validate(
            evalfunc, data_iter_func, quantize, iter_num=iter_num,
            logger=logging.getLogger('mrt.validate'), batch_size=batch)
        logger.info("evaluatation stage finished")
    else:
        logger.info("evaluatation stage skipped")
