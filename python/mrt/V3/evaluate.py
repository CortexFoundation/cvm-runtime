"""
Evaluation Module for MRT V3.

Evaluate function definition, default YAML configurations for MRT evaluation
Stage options and Command line help prompt are also included.
"""

from yacs.config import CfgNode as CN
import logging
import time

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

def forward(net, data, ctx, baxis, olen):
    """
    Multiple xpu run support.

    Parameters
    ----------
    net : mxnet.gluon.block.SymbolBlock
        Graph for inference.
    data : mxnet.ndarray.ndarray.NDArray
        Input data to pass into the graph.
    ctx : mx.context.Context
        Context for inference.
    baxis : int
        Axis id of batch dimension.
    olen : int
        Length of the output.

    Returns
    -------
    outs : mxnet.ndarray.ndarray.NDArray or list
        inference result of the graph with respect to the given input data,
        for multiple outputs, outs will be a list the entry type of which is
        mxnet.ndarray.ndarray.NDArray.
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

def get_evaluation_info(cm_cfg, pass_cfg, logger=None):
    """
    YAML configuration API to get evaluation function,
    quantization function and dataset iteration function

    Parameters
    ----------
    cm_cfg : yacs.config.CfgNode
        CfgNode of common stage.
    pass_cfg : yacs.config.CfgNode
        CfgNode of calibration stage.
    logger : logging.RootLogger
        Console logger.
    """
    model_dir = cm_cfg.MODEL_DIR
    model_name = cm_cfg.MODEL_NAME
    verbosity = cm_cfg.VERBOSITY
    device_type = pass_cfg.DEVICE_TYPE
    device_ids = pass_cfg.DEVICE_IDS
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
    model_prefix_fixed = model_prefix + ".fixed"
    omodel = Model.load(*load_fname(model_prefix_fixed))
    graph = omodel.to_graph(ctx=ctx)
    dataset_name = conf_map["dataset_name"]
    input_shape = conf_map["input_shape"]
    dataset = ds.DS_REG[dataset_name](set_batch(input_shape, batch))
    data_iter_func = dataset.iter_func()
    metric = dataset.metrics()
    baxis = get_batch_axis(input_shape)
    olen = len(omodel.symbol)

    # def forward(net, data, ctx):
        # """ Multiple xpu run support.
        # """
        # data = gluon.utils.split_and_load(
            # data, ctx_list=ctx, batch_axis=baxis, even_split=False)
        # outs = [net(d) for d in data]
        # if olen == 1:
            # outs = nd.concatenate(outs)
        # else:
            # outs = [nd.concatenate([outs[i][j] \
                # for i in range(len(outs))]) for j in range(olen)]
        # return outs

    def evalfunc(data, label):
        # outs = forward(graph, data, ctx=ctx)
        outs = forward(graph, data, ctx, baxis, olen)
        start = time.time()
        acc = dataset.validate(metric, outs, label)
        end = time.time()
        return acc, int((end-start)*1e3)

    # forward function for the quantized model
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
    qgraph = qmodel.to_graph(ctx=ctx)
    qmetric = dataset.metrics()

    def quantize(data, label):
        data = sim.load_real_data(data, 'data', inputs_ext)
        # outs = forward(qgraph, data, ctx)
        outs = forward(qgraph, data, ctx, baxis, olen)
        outs = outs / oscales[0] if olen == 1 \
            else [(t / oscales[i]) for i, t in enumerate(outs)]
        start = time.time()
        acc = dataset.validate(qmetric, outs, label)
        end = time.time()
        return acc, int((end-start)*1e3)

    return evalfunc, data_iter_func, quantize

def evaluate(cm_cfg, pass_cfg, logger=None):
    """
    YAML configuration API of MRT evaluation stage.

    Parameters
    ----------
    cm_cfg : yacs.config.CfgNode
        CfgNode of common stage.
    pass_cfg : yacs.config.CfgNode
        CfgNode of calibration stage.
    logger : logging.RootLogger
        Console logger.
    """
    evalfunc, data_iter_func, quantize = get_evaluation_info(
        cm_cfg, pass_cfg, logger=logger)

    iter_num = pass_cfg.ITER_NUM
    batch = pass_cfg.BATCH
    if batch is None:
        batch = cm_cfg.BATCH

    if iter_num > 0:
        logger.info("Validating...")
        utils.multi_validate(
            evalfunc, data_iter_func, quantize, iter_num=iter_num,
            logger=logging.getLogger('mrt.validate'), batch_size=batch)
        logger.info("evaluatation stage finished")
    else:
        logger.info("evaluatation stage skipped")

def get_ctx_eval(ctx):
    """
    Get the context instance for evaluation stage

    Parameters
    ----------
    ctx : mx.context.Context
        The input context.

    Returns
    -------
    ctx : mx.context.Context
        The modified context.
    """
    if isinstance(ctx, mx.Context):
        ctx = [ctx]
    elif isinstance(ctx, list):
        assert all([isinstance(c, mx.Context) for c in ctx]), \
            "invalid value of ctx: {}".format(ctx)
    else:
        assert False, "invalid type of ctx: {}".format(type(ctx))
    return ctx

def inference_original_model(
    symbol_file, params_file, data, batch_axis=0,
    device_type=MRT_CFG.EVALUATE.DEVICE_TYPE,
    device_ids=MRT_CFG.EVALUATE.DEVICE_IDS):
    """
    MRT Inference API for original model.

    Parameters
    ----------
    symbol_file : str
        Path to the quantized mxnet symbol JSON file.
    params_file : str
        Path to the quantized mxnet parameters file.
    data: mxnet.ndarray.ndarray.NDArray
        Input data to pass into the graph.
    batch_axis : int
        Axis id of batch dimension.
    device_type : str
        Context type string chosen from `cpu` or `gpu`.
    device_ids : list
        List of context ids.

    Returns
    -------
    outs : mxnet.ndarray.ndarray.NDArray or list
        inference result of the graph with respect to the given input data,
        for multiple outputs, outs will be a list the entry type of which is
        mxnet.ndarray.ndarray.NDArray.
    """

    ctx = get_ctx_eval(get_ctx(device_type, device_ids))
    omodel = Model.load(symbol_file, params_file)
    graph = omodel.to_graph(ctx=ctx)
    olen = len(omodel.symbol)

    outs = forward(graph, data, ctx, batch_axis, olen)
    return outs

def inference_quantized_model(
    qsymbol_file, qparams_file, qext_file, data, batch_axis=0, split=False,
    device_type=MRT_CFG.EVALUATE.DEVICE_TYPE,
    device_ids=MRT_CFG.EVALUATE.DEVICE_IDS):
    """
    MRT Inference API for quantized model.

    Parameters
    ----------
    qsymbol_file : str
        Path to the quantized mxnet symbol JSON file.
    qparams_file : str
        Path to the quantized mxnet parameters file.
    qext_file : str
        Path to the quantized extension file which store intermediate results.
    data: mxnet.ndarray.ndarray.NDArray
        Input data to pass into the graph.
    batch_axis : int
        Axis id of batch dimension.
    split: bool
        Flag indicating whether the model is split before quantization.
    device_type : str
        Context type string chosen from `cpu` or `gpu`.
    device_ids : list
        List of context ids.

    Returns
    -------
    outs : mxnet.ndarray.ndarray.NDArray or list
        inference result of the graph with respect to the given input data,
        for multiple outputs, outs will be a list the entry type of which is
        mxnet.ndarray.ndarray.NDArray.
    """

    ctx = get_ctx_eval(get_ctx(device_type, device_ids))

    if split:
        qmodel = Model.load(qsymbol_file, qparams_file)
        oscales, inputs_ext = sim.load_ext(qext_file)
    else:
        mrt = MRT(Model.load(qsymbol_file, qparams_file))
        mrt.old_names, mrt.th_dict, mrt.precs, mrt.scales = \
            sim.load_ext(qext_file)
        oscales = mrt.get_output_scales()
        inputs_ext = mrt.get_inputs_ext()
        qmodel = mrt.current_model

    rqmodel = reduce_graph(qmodel, {'data': data.shape})
    qgraph = rqmodel.to_graph(ctx=ctx)
    data = sim.load_real_data(data, 'data', inputs_ext)
    olen = len(rqmodel.symbol)

    outs = forward(qgraph, data, ctx, batch_axis, olen)
    outs = outs / oscales[0] if olen == 1 \
        else [(t / oscales[i]) for i, t in enumerate(outs)]
    return outs
