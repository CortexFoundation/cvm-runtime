import mxnet as mx
from mxnet import gluon

import numpy as np

from quant_op import *
from quant_utils import *
from utils import *

def fuse_bn_parameters(resnet_params, quant_flag):
    logger = logging.getLogger("log.fuse.conv_bn")
    logger.setLevel(quant_flag.log_level)

    if not quant_flag.is_fuse_bn:
        logger.info("skip fuse conv_bn pass with quant_flag.is_fuse_bn=False")
        return resnet_params

    logger.debug("fuse conv_bn parameters")

    added_params_name, val_changed_params_name, deleted_params_name = [], [], []
    param_names = resnet_params.keys()
    for name in list(param_names):
        if "conv" in name:
            if "bias" in name:
                continue

            bias_name = name.replace("weight", "bias")

            tmp = name.replace("conv", "batchnorm")
            bn_gamma_name = tmp.replace("weight", "gamma")
            bn_beta_name = tmp.replace("weight", "beta")
            bn_running_mean_name = tmp.replace("weight", "running_mean")
            bn_running_var_name = tmp.replace("weight", "running_var")

            assert bn_gamma_name in param_names, name
            assert bn_beta_name in param_names, name
            assert bn_running_mean_name in param_names, name
            assert bn_running_var_name in param_names, name

            conv_weight = resnet_params[name]
            bn_gamma = resnet_params[bn_gamma_name]
            bn_beta = resnet_params[bn_beta_name]
            bn_running_mean = resnet_params[bn_running_mean_name]
            bn_running_var = resnet_params[bn_running_var_name]

            assert conv_weight.shape[0] == bn_gamma.shape[0]
            assert conv_weight.shape[0] == bn_beta.shape[0]
            assert conv_weight.shape[0] == bn_running_mean.shape[0]
            assert conv_weight.shape[0] == bn_running_var.shape[0]

            # the epsilon variable matters the output.
            epsilon = 1e-5
            tmp = (nd.sqrt(bn_running_var + epsilon))
            scale = bn_gamma / tmp
            offset = bn_beta - scale * bn_running_mean

            bias_scale = scale
            weight_scale = scale.repeat(np.product(
                    conv_weight.shape[1:])).reshape(conv_weight.shape)

            val_changed_params_name.append(name)
            (val_changed_params_name if bias_name in param_names
                else added_params_name).append(bias_name)
            deleted_params_name.extend([bn_gamma_name, bn_beta_name,
                bn_running_mean_name, bn_running_var_name])

            resnet_params[name] = conv_weight * scale.reshape((scale.shape[0], 1, 1, 1))
            offset += bias_scale * (resnet_params[bias_name] if bias_name in param_names else 0)
            resnet_params[bias_name] = offset

            del resnet_params[bn_gamma_name]
            del resnet_params[bn_beta_name]
            del resnet_params[bn_running_mean_name]
            del resnet_params[bn_running_var_name]

    logger.debug("[ added_params_name       ]: %s", added_params_name)
    logger.debug("[ val_changed_params_name ]: %s", val_changed_params_name)
    logger.debug("[ deleted_params_name     ]: %s", deleted_params_name)
    return resnet_params

def matrix_decomposition(params, quant_flag):
    if not quant_flag.matrix_decomposition:
        return params

    param_names = params.keys()
    for name in list(param_names):
        if name == 'fc0_weight':
            weight = params['fc0_weight']
            matrix_len = 100352
            start, step, idx = 0, 60000, 0
            while start < matrix_len:
                stop = min(start+step, matrix_len)

                weight_name = 'fc0_' + str(idx) + '_weight'
                params[weight_name] = weight.slice(
                        begin=(None, start), end=(None, stop))

                bias_name = 'fc0_' + str(idx) + '_bias'
                params[bias_name] = params['fc0_bias']

                start, idx = stop, idx+1

            del params['fc0_weight']

    return params

def calibrate_parameters(graph, qparams, ctx, calib_data, quant_flag, name_scope=""):
    logger = logging.getLogger("log.calib")
    logger.setLevel(quant_flag.log_level)

    if quant_flag.calib_mode == CalibMode.NONE:
        logger.info("skip calibration pass with quant_flag.calib_mode=NONE")
        return qparams

    inputs = mx.sym.var('data')
    cpu_ctx = mx.cpu()
    len_scope = len(name_scope)

    print ("graph", graph.collect_params().keys())
    layers = graph(inputs).get_internals()
    # TODO: !!!IMPORTANT!!!, outputs must be logic sequential
    outputs = [sym for sym in layers.list_outputs() if sym.endswith("_output") ]
    outputs_quant_flags = {output: True for output in outputs}

    # image_data = calib_data.data[0]
    _, input_shift_bits = quant_helper(calib_data)
    calib_res = {}

    def collect_quant_layers():
        if len(quant_flag.disabled_layers) > 0:
            for lname in outputs:
                if any(dis_layer in lname for dis_layer in quant_flag.disabled_layers):
                    outputs_quant_flags[lname] = False

    def calib_IO_data():
        logger = logging.getLogger("log.calib.output_data")
        logger.setLevel(quant_flag.log_level)

        added_params_name = []
        for lname in outputs:
            stacked_graph = gluon.SymbolBlock(layers[lname], [inputs])
            load_parameters(stacked_graph, qparams, prefix=name_scope, ctx=ctx)
            qres = stacked_graph.forward(calib_data.as_in_context(ctx))

            prename = lname[len_scope:].replace("_fwd", "").replace("_output", "")
            output_sb_name = prename + "_output_shift_bits"

            quant_res, output_sb = quant_helper(qres)
            qparams[output_sb_name]= output_sb.as_in_context(cpu_ctx)

            calib_res[lname] = {
                    "header": quant_res[0].asnumpy().flatten()[0],
                    "max": quant_res.max().asnumpy(),
                    "min": quant_res.min().asnumpy(),
                    "shape": quant_res.shape,
                    "shift_bits": output_sb.asnumpy(),
            }

            logger.debug("calib output data in layer(%s) with shift bits(%s)",
                    lname, output_sb.asnumpy())

            del qres
            del quant_res
            added_params_name.append(output_sb_name)

        logger.debug("[ added_params_name   ]: %s", added_params_name)

    def rewrite_op_params():
        # rewrite possible operator and calculate input&output shift bits
        logger = logging.getLogger("log.calib.op.rewrite")
        logger.setLevel(quant_flag.log_level)

        added_params_name, val_changed_params_name = [], []
        for lname in outputs:
            prename = lname[len_scope:].replace("_fwd", "").replace("_output", "")
            input_sb_name = prename + "_input_shift_bits"
            output_sb_name = prename + "_output_shift_bits"

            inputs_name, inputs_sb = layers[lname].get_children().list_outputs(), []
            for name in inputs_name:
                if name.endswith("_output"):
                    pname = name[len_scope:].replace("_fwd", "").replace("_output", "")
                    inputs_sb.append(qparams[pname+"_output_shift_bits"])
                elif name == "data":
                    inputs_sb.append(input_shift_bits)
            inputs_num = len(inputs_sb)

            assert inputs_num > 0, "Unsupported rewrite op %s with no input"%(lname)
            if inputs_num > 2:
                logger.error("Unsupported layer %s with inputs<%s> larger than 2",
                        lname, inputs_name)
                assert False
            elif inputs_num == 2:
                assert "_plus" in prename, \
                    "Only supported rewrite binary-op _plus, instead of %s"%(lname)

                assert outputs_quant_flags[lname], "binary-op %s must be quantize"%(lname)

                f_sb = inputs_sb[0]
                f_plus_sb_name = prename + "_first_shift_bits"
                qparams[f_plus_sb_name] = nd.zeros((1,))

                s_sb = inputs_sb[1]
                s_plus_sb_name = prename + "_second_shift_bits"
                qparams[s_plus_sb_name] = nd.zeros((1,))

                if any(f_sb > s_sb):
                    qparams[s_plus_sb_name] = f_sb - s_sb
                    input_sb = f_sb
                else:
                    qparams[f_plus_sb_name] = s_sb - f_sb
                    input_sb = s_sb

                logger.debug("binary-op _plus inputs shift bits: %s, %s",
                        qparams[f_plus_sb_name].asnumpy(), qparams[s_plus_sb_name].asnumpy())
                added_params_name.append([f_plus_sb_name, s_plus_sb_name])
            else:
                input_sb = inputs_sb[0]

                if "pool" in prename:
                    attrs = layers[lname].list_attr()
                    pool_type = attrs["pool_type"]
                    pool_size = attrs["kernel"]
                    global_pool = attrs["global_pool"]

                    unchangable_pool_type = ["max", "sum"]
                    unsupported_pool_type = ["sum", "lp"]
                    rewritable_pool_type = ["avg"]

                    if pool_type in unchangable_pool_type:
                        pass
                    elif pool_type in unsupported_pool_type:
                        logger.critical("Unsupported pooling op for quantization: %s",
                                lname)
                        assert False
                    elif pool_type in rewritable_pool_type:
                        # avg pool_type
                        shape = calib_res[inputs_name[0]]["shape"]
                        if len(pool_size.split(',')) != 2 or global_pool != "True":
                            logger.critical("Only support GlobalAvgPool2D instead of attrs %s",
                                attrs)
                            assert False

                        weight_name = prename + "_weight"
                        assert weight_name not in qparams
                        qparams[weight_name] = nd.array([1. / (shape[2] * shape[3])])
                        logger.debug("average pool op add weight params: %s",
                                qparams[weight_name].asnumpy())
                    else:
                        logger.critical("Unrecognized pooling op type: %s", lname)
                        assert False

            qparams[input_sb_name] = input_sb
            if not outputs_quant_flags[lname]:
                qparams[output_sb_name] = input_sb
                val_changed_params_name.append(output_sb_name)

            added_params_name.append(input_sb_name)

        logger.debug("[ added_params_name       ]: %s", added_params_name)
        logger.debug("[ val_changed_params_name ]: %s", val_changed_params_name)

    def quant_op_params():
        logger = logging.getLogger("log.calib.quant.params")
        logger.setLevel(logging.INFO)

        added_params_name, deleted_params_name = [], []
        # quantize weight params
        for lname in outputs:
            prename = lname[len_scope:].replace("_fwd", "").replace("_output", "")
            print (lname, prename)
            weight_name = prename + "_weight"
            quant_name = prename + "_weight_quant"
            sb_name = prename + "_weight_shift_bits"

            if (weight_name in qparams) and outputs_quant_flags[lname]:
                qparams[quant_name], qparams[sb_name] = quant_helper(qparams[weight_name])

                del qparams[weight_name]
                deleted_params_name.append(weight_name)
                added_params_name.extend([quant_name, sb_name])

        # quantize bias params
        for lname in outputs:
            prename = lname[len_scope:].replace("_fwd", "").replace("_output", "")
            bias_name = prename + "_bias"
            bias_quant_name = prename + "_bias_quant"
            bias_sb_name = prename + "_bias_shift_bits"
            weight_sb_name = prename + "_weight_shift_bits"
            input_sb_name = prename + "_input_shift_bits"

            if (bias_name in qparams) and outputs_quant_flags[lname]:
                shift_bits = qparams[weight_sb_name] + qparams[input_sb_name]
                qparams[bias_quant_name], qparams[bias_sb_name] = quant_helper(
                        qparams[bias_name], shift_bits=shift_bits,
                        target_bits=BIAS_TARGET_BITS, logger=logger, msg=prename)

                del qparams[bias_name]
                deleted_params_name.append(bias_name)
                added_params_name.append([bias_quant_name, bias_sb_name])

        logger.debug("[ added_params_name   ]: %s", added_params_name)
        logger.debug("[ deleted_params_name ]: %s", deleted_params_name)

    def requant_params():
        logger = logging.getLogger("log.calib.requant")
        logger.setLevel(quant_flag.log_level)

        added_params_name, deleted_params_name = [], []
        for lname in outputs:
            prename = lname[len_scope:].replace("_fwd", "").replace("_output", "")
            input_sb_name = prename + "_input_shift_bits"
            output_sb_name = prename + "_output_shift_bits"
            weight_sb_name = prename + "_weight_shift_bits"
            requant_sb_name = prename + "_requant_shift_bits"

            if outputs_quant_flags[lname]:
                qparams[requant_sb_name] = qparams[output_sb_name] - qparams[input_sb_name]
                if weight_sb_name in qparams:
                    qparams[requant_sb_name] -= qparams[weight_sb_name]

                # out_quant, out_sb = quant_helper(calib_res[idx])
                log_res = calib_res[lname]
                logger.debug("quant layer=%s, requant,%s=out,<%s,%s,%s,%s>-input,%s-weight,%s",
                        prename, qparams[requant_sb_name].asnumpy(),
                        log_res["header"], log_res["max"], log_res["min"],
                        log_res["shift_bits"], qparams[input_sb_name].asnumpy(),
                        # out_quant.asnumpy().flatten()[0],
                        # out_quant.max().asnumpy(), out_quant.min().asnumpy(),
                        # out_sb.asnumpy(), qparams[input_sb_name].asnumpy(),
                        qparams[weight_sb_name].asnumpy() if weight_sb_name in qparams else '0')

                added_params_name.append(requant_sb_name)

        logger.debug("[ added_params_name   ]: %s", added_params_name)
        logger.debug("[ deleted_params_name ]: %s", deleted_params_name)

    def reduce_params():
        logger = logging.getLogger("log.calib.reduce")
        logger.setLevel(quant_flag.log_level)

        added_params_name, deleted_params_name = [], []
        param_names = qparams.keys()
        for pname in list(param_names):
            if pname.endswith("_quant"):
                assert pname[:-6] not in param_names
                qparams[pname[:-6]] = qparams[pname]

                del qparams[pname]
                deleted_params_name.append(pname)
                added_params_name.append(pname[:-6])

        for lname in outputs:
            prename = lname[len_scope:].replace("_fwd", "").replace("_output", "")
            input_sb_name = prename + "_input_shift_bits"
            output_sb_name = prename + "_output_shift_bits"
            weight_sb_name = prename + "_weight_shift_bits"
            bias_sb_name = prename + "_bias_shift_bits"

            del_names = [input_sb_name, output_sb_name]
            if weight_sb_name in qparams:
                del_names.append(weight_sb_name)
            if bias_sb_name in qparams:
                del_names.append(bias_sb_name)

            deleted_params_name.extend(del_names)
            for name in del_names:
                del qparams[name]


        logger.debug("[ added_params_name       ]: %s", added_params_name)
        logger.debug("[ deleted_params_name     ]: %s", deleted_params_name)

    collect_quant_layers()

    logger.info(">>>> calculate output shift bits")
    calib_IO_data()

    logger.info(">>>> op rewrite for quantization")
    rewrite_op_params()

    logger.info(">>>> quantize params shift bits")
    quant_op_params()

    logger.info(">>>> calculate layer requant shift_bits")
    requant_params()

    logger.info(">>>> resume params name")
    reduce_params()

    logger.debug(">>>> calibration params: %s", qparams.keys())

    return qparams
