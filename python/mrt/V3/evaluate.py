from yacs.config import CfgNode as CN

from mrt import conf
from mrt.V3.utils import (
    MRT_CFG, default_device_type, default_device_ids, default_batch,
    get_model_prefix, get_logger, set_batch, load_fname,
    load_conf, check_file_existance, get_ctx, get_batch_axis)

MRT_CFG.EVALUATE = CN()
MRT_CFG.EVALUATE.BATCH = default_batch
MRT_CFG.EVALUATE.DEVICE_TYPE = default_device_type
MRT_CFG.EVALUATE.DEVICE_IDS = default_device_ids
MRT_CFG.EVALUATE.ITER_NUM = 10

def evaluate(
    model_dir, model_name, verbosity, device_type, device_ids, iter_num,
    batch=default_batch):
    model_prefix = get_model_prefix(model_dir, model_name)
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

def yaml_evaluate():
    CM = MRT_CFG.COMMON
    CN = MRT_CFG.EVALUATE
    evaluate(
        CM.MODEL_DIR, CM.MODEL_NAME, CM.VERBOSITY, CN.DEVICE_TYPE, CN.DEVICE_IDS,
        CN.ITER_NUM, batch=CN.BATCH)
