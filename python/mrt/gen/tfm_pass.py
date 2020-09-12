import logging
import json

import mxnet as mx

from mrt.tfm_base import N
from mrt.tfm_pass import convert_params_dtype, infer_shape, \
                         fuse_constant
from mrt.sym_utils import is_inputs, get_entry_id, is_var, \
                          get_nd_op, topo_visit_transformer
from mrt.tfm_pass import OUT_KEY
from .tfm_types import get_quantizer, DEFAULT_QUANT_TYPE, get_optimizor, \
                       DEFAULT_OPT_INFO, BUF_TYPE_EXP, FT_TYPE_EXP
from .tfm_utils import get_buffer_exp, get_bit_exp, scale_exp, \
                       get_quantizer_exp
from .tfm_base import apply_pass
from .tfm_types import LAYER_WISE_TYPE, CHANNEL_WISE_TYPE, \
                       DEFAULT_GN_INFO, QUANT_REG, OPT_REG, make_key_opt
from .tfm_ops import Convolution, FullyConnected

from mrt import sym_utils as sutils

#----------------------------
# Module main interfaces
#----------------------------

_RES_NAME = "_RES_"

def sym_config_infos(symbol, params, cfg_dict={}, logger=logging):
    """ Customized graph-level topo pass definition.

        Interface for MRT main2 configuration
        Create customized samplers and optimizors.

        Use it just before calibration.
    """
    names = set()

    def _collect_names(symbol, params, **kwargs):
        names.add(symbol.attr("name"))

    topo_visit_transformer(symbol, params, _collect_names)
    noncfgs = set()
    keys = list(cfg_dict.keys())
    for name in keys:
        if name == _RES_NAME:
            continue
        if name not in names:
            del cfg_dict[name]
            noncfgs.add(name)
    if noncfgs:
        logger.warn(
            "Symbols (names: %s) not found in graph." + \
            "Please double check config file (.ini)." % list(noncfgs))
    if _RES_NAME in cfg_dict:
        cfg_info = cfg_dict.pop(_RES_NAME)
        keys = cfg_dict.keys()
        for name in [n for n in names if n not in keys]:
            cfg_dict[name] = cfg_info

    def _sym_config_infos(sym, params, **kwargs):
        name = sym.attr("name")
        cfg_info = cfg_dict.get(name, {})

        gn_info = cfg_info.get("gn_info", DEFAULT_GN_INFO)

        quant_type = cfg_info.get("quant_type", DEFAULT_QUANT_TYPE)
        get_quantizer(quant_type)

        opt_info = cfg_info.get(
            "opt_type", make_key_opt(DEFAULT_OPT_INFO))
        get_optimizor(opt_info)

        cfg_dict[name] = cfg_info if cfg_info else \
            {"gn_info": gn_info, "quant_type": quant_type,
            "opt_info": opt_info}

    topo_visit_transformer(symbol, params, _sym_config_infos)
    return cfg_dict

_NONETYPE_NAME = "_NoneType"

def deserialize(cfg_groups):
    """ Interface for MRT main2 configuration

        Check the validity and compatibility of feature, sampler and optimizor configurations.

        Parameters
        ----------
        cfg_groups : dict
            configuration information (quantizer type, optimizor information) maps to node names (before calibration).
    """
    cfg_dict = {}
    for names, val_dict in cfg_groups.items():
        try:
            names = json.loads(names.replace(".", ","))
        except:
            raise ValueError("Invalid value, names: %s" % names)

        # Deserialize
        cfg_info = {}
        for attr, val in val_dict.items():
            if attr not in ["gn_info", "opt_info"]:
                cfg_info[attr] = val
                continue
            try:
                val = json.loads(
                    val.replace(".", ",").replace(";", ":"))
            except:
                raise ValueError(
                    "Invalid value, names: %s, attr: %s, val: %s" % \
                    (names, attr, val))
            cfg_info[attr] = val

        # Granularity Settings Validate
        gn_info = cfg_info.get("gn_info", DEFAULT_GN_INFO)
        if "gn_type" not in gn_info:
            raise ValueError(
                "Please specify the opt_type, names: %s, " + \
                "opt_info: %s" % (names, opt_info))
        gn_type = gn_info["gn_type"]
        if gn_type == CHANNEL_WISE_TYPE:
            if "ichannel" not in gn_info:
                raise ValueError(
                    "Please specify the axis number of channel " + \
                    "(ichannel), names: %s" % names)
            ichannel = gn_info["ichannel"]
            if not isinstance(ichannel, int):
                raise ValueError(
                    "Please specify the correct axis number of channel " + \
                    "(ichannel), names: %s, ichannel: %s" % \
                    (names, ichannel))
            if "step" not in gn_info:
                raise ValueError(
                    "Please specify the step size of channel " + \
                    "(step), names: %s" % names)
            step = gn_info["step"]
            if not isinstance(step, int):
                raise ValueError(
                    "Please specify the correct step of channel " + \
                    "(step), names: %s, step: %s" % (names, step))
        elif gn_type == LAYER_WISE_TYPE:
            if len(gn_info) > 1:
                raise ValueError(
                    "Redundant values in gn_info: %s" % gn_info)
        else:
            raise TypeError(
                "Unsupported granulari type: %s, names: %s" % \
                (gn_type, names))
        cfg_info["gn_info"] = gn_info

        # Quantizer Settings Validate
        quant_type = cfg_info.get("quant_type", DEFAULT_QUANT_TYPE)
        if quant_type not in QUANT_REG:
            raise TypeError(
                "Unsupported quantizer type: %s, names: %s" % \
                (quant_type, names))
        cfg_info["quant_type"] = quant_type

        # Optimizor Settings Validate
        opt_info = cfg_info.get("opt_info", DEFAULT_OPT_INFO)
        if "opt_type" not in opt_info:
            raise ValueError(
                "Please specify the opt_type, names: %s, " + \
                "opt_info: %s" % (names, opt_info))
        opt_type = opt_info["opt_type"]
        if opt_type not in OPT_REG:
            raise TypeError(
                "Unsupported optimizor type: %s, names: %s" % \
                (opt_type, names))
        if quant_type not in OPT_REG[opt_type].list_supported_quant_types():
            raise ValueError(
                "quantizer type: (%s) is not supported by " + \
                "optimizor type: (%s), names: %s" % \
                (quant_type, opt_type, names))
        opt_attrs = opt_info.copy()
        opt_attrs.pop("opt_type")
        opt_attr_types = OPT_REG[opt_type].list_attr_types()
        for k, v in opt_attrs.items():
            if k not in opt_attr_types:
                raise ValueError(
                    "Attribute: (%s) is not found in " + \
                    "optimizor type: (%s), names: %s" % \
                    (k, opt_type, names))
            dtypes = opt_attr_types[k]
            if isinstance(v, int) and float in dtypes and int not in dtypes:
                v = float(v)
            elif v == _NONETYPE_NAME:
                v = None
            if not any([isinstance(v, dtype) for dtype in dtypes]):
                raise TypeError(
                    "Attribute: (%s) dtype: (%s) is not compatible " + \
                    "with any of supported dtypes: (%s), names: %s, " + \
                    "optimizor type: %s" % \
                    (k, type(v), dtypes, names, opt_type))
        cfg_info["opt_info"] = make_key_opt(opt_info)

        for name in names:
            assert name not in cfg_dict
            cfg_dict[name] = cfg_info.copy()

    return cfg_dict

#----------------------------
# Module calibrate interfaces
#----------------------------

def sym_calibrate(symbol, params, data, cfg_dict, **kwargs):
    # TODO(archRev): independent of other interfaces besides sample, can be move to tfm_pass
    logger = logging.getLogger('log.mrt')
    _, deps = sutils.topo_sort(
        symbol, logger=logger, with_deps=True)
    features, out_cache = {}, {}
    ctx = kwargs.get('ctx', mx.cpu())
    logger.info("calibrate model outputs")
    nparams = convert_params_dtype(
        params, src_dtypes="float64", dest_dtype="float32")

    def _impl(op, params, graph, **kwargs):
        deps= kwargs['deps']
        logger = logging.getLogger('log.mrt.calibrate')
        name, op_name = op.attr('name'), op.attr('op_name')
        childs, attr = sutils.sym_iter(
            op.get_children()), op.list_attr()
        quant_type, opt_info = \
            cfg_dict[name]["quant_type"], cfg_dict[name]["opt_info"]
        quantizer, optimizor = \
            get_quantizer(quant_type), get_optimizor(opt_info)

        if op_name == 'null':
            out = data if is_inputs(op, params) else params[name]
        elif childs is None:
            out = get_nd_op(op_name)(**attr)
        else:
            cinfos = [(c.attr('name'), get_entry_id(c)) for c in childs]
            nd_inputs = [out_cache[n[0]][n[1]] for n in cinfos]
            out = get_nd_op(op_name)(*nd_inputs, **attr)
            for n, _ in cinfos:
                assert n in deps
                if name not in deps[n]:
                    # for op like: op = broadcast_mul(X, X)
                    # `cinfos` will have duplicate entries
                    # avoid removing more than once
                    continue
                deps[n].remove(name)
                if len(deps[n]) == 0:
                    del out_cache[n]

        out = [out] if len(op) == 1 else out
        out_cache[name] = [o.as_in_context(ctx) for o in out]
        raw_ft = quantizer.sample(out[0])
        hist_ft = features[name] if name in features else None
        features[name] = optimizor.get_opt(
            raw_ft, out[0], hist_ft=hist_ft, logger=logger, name=name)

    topo_visit_transformer(
        symbol, nparams, _impl, logger=logger,
        deps=deps, data=data, **kwargs)
    out_cache.clear()

    return features

#----------------------------
# Module quantize interfaces
#----------------------------

@N.register_nm("rewrite")
def rewrite(symbol, params):
    infer_shapes = infer_shape(symbol, params)
    return topo_visit_transformer(symbol, params,
            apply_pass("rewrite", infer_shapes=infer_shapes))

@N.register_nm("sym_separate_pad")
def sym_separate_pad(symbol, params):
    """ Separate pad attribute as an independent symbol in rewrite stage.
    """
    def _separate_pad(op, **kwargs):
        name, op_name = op.attr('name'), op.attr('op_name')
        attr, childs = op.list_attr(), sutils.sym_iter(op.get_children())

        if op_name not in [Convolution.op_name]:
            return op

        assert 'layout' in attr and attr['layout'] == 'NCHW'
        PH, PW = sutils.get_attr(attr, 'pad', (0,0))
        del attr['pad']
        if PH == 0 and PW == 0:
            return sutils.get_mxnet_op(op_name)(*childs, **attr, name=name)

        childs[0] = mx.sym.pad(
            childs[0], pad_width=(0,0,0,0,PH,PH,PW,PW),
            mode='constant', constant_value=0, name=N.n('pad'))
        op = sutils.get_mxnet_op(op_name)(*childs, **attr, name=name)
        return op

    return topo_visit_transformer(symbol, params, _separate_pad)

@N.register_nm("sym_separate_bias")
def sym_separate_bias(symbol, params):
    """ Separate bias attribute as an independent symbol in rewrite stage.
    """
    def _separate_bias(op, **kwargs):
        name, op_name = op.attr('name'), op.attr('op_name')
        attr, childs = op.list_attr(), sutils.sym_iter(op.get_children())

        if childs and len(childs) < 3 or op_name not in \
            [Convolution.op_name, FullyConnected.op_name]:
            return op

        attr['no_bias'] = True
        op = sutils.get_mxnet_op(op_name)(
            childs[0], childs[1], **attr, name=N.n(name))
        bn = childs[2].attr('name')
        if op_name == Convolution.op_name:
            assert 'layout' in attr and attr['layout'] == 'NCHW'
            B = mx.sym.expand_dims(childs[2], axis=0, name=N.n('expand_dims'))
            B = mx.sym.expand_dims(B, axis=-1, name=N.n('expand_dims'))
            B = mx.sym.expand_dims(B, axis=-1, name=N.n(bn))
        else:
            B = mx.sym.expand_dims(childs[2], axis=0, name=N.n(bn))
        op = mx.sym.broadcast_add(op, B, name=name)
        return op

    return topo_visit_transformer(symbol, params, _separate_bias)

#----------------------------
# Channel Slice interfaces
#----------------------------

@N.register_nm("slice_channel")
def sym_slice_channel(symbol, params, cfg_dict={}):
    """ Customized graph-level topo pass definition.

        Interface for granularity control.
        While layer-wise feature is by default,
        MRT support channel-wise features specified in cfg_dict.
    """
    infer_shapes = infer_shape(symbol, params)

    def _slice_channel(op, **kwargs):
        name, op_name = op.attr("name"), op.attr("op_name")
        gn_info = cfg_dict[name].get("gn_info", DEFAULT_GN_INFO)
        gn_type = gn_info["gn_type"]
        if gn_type == CHANNEL_WISE_TYPE:
            op = apply_pass(
                "slice_channel", cfg_dict=cfg_dict,
                infer_shapes=infer_shapes)(op, **kwargs)
        return op

    sym, params = topo_visit_transformer(symbol, params, _slice_channel)
    sym, params = fuse_constant(sym, params)
    return sym, params

#----------------------------
# Module quantize interfaces
#----------------------------

@N.register_nm("quantize")
def quantize(
    symbol, params, features, precs, buffers, cfg_dict,
    op_input_precs, restore_names, shift_bits, softmax_lambd):

    infer_shapes = infer_shape(symbol, params)

    def restore(op, **kwargs):
        features, precs, buffers = \
            kwargs['features'], kwargs['precs'], kwargs['buffers']
        name, op_name = op.attr('name'), op.attr('op_name')
        childs, attr = sutils.sym_iter(op.get_children()), op.list_attr()

        childs = [] if childs is None else childs

        new_childs = []
        for c in childs:
            cname = c.attr('name')
            sc = buffers[c.attr('name')].get() \
                if cname in buffers else 1
            new_childs.append(c if sc == 1 else c / sc)

        out = sutils.get_mxnet_op(op_name)(*new_childs, **attr, name=name)
        ft = features[name]
        assert ft.name == FT_TYPE_EXP
        absmax = features[name].get()
        precs[name][OUT_KEY] = get_bit_exp(absmax)
        buffers[name] = get_buffer_exp(1)
        return out

    def _quant(op, **kwargs):
        op = apply_pass("quantize",
            infer_shapes=kwargs['infer_shapes'],
            features=kwargs['features'],
            cfg_dict=kwargs['cfg_dict'],
        )(op, **kwargs) if op.attr('name') not in restore_names \
            else restore(op, **kwargs)

        if is_var(op, kwargs['params']):
            return op

        name = op.attr('name')
        features, buffers = kwargs['features'], kwargs['buffers']
        precs = kwargs['precs']
        ft = features[name]
        assert ft.name == FT_TYPE_EXP
        absmax = ft.get()
        name, op_name = op.attr('name'), op.attr('op_name')
        #  if name == "mrt_quantize_realize_19":
            #  childs = sutils.sym_iter(op.get_children())
            #  for c in childs:
                #  cname, cop_name = c.attr('name'), c.attr('op_name')
                #  print(cname, cop_name)
        buf = buffers[name]
        assert buf.name == BUF_TYPE_EXP
        scale = buf.get()
        tight_prec = get_bit_exp(absmax*scale)
        if precs[name][OUT_KEY] > tight_prec:
            op = mx.sym.Custom(
                op, precision=tight_prec,
                name=N.n('clip'), op_type='cvm_clip')
            clip_name = op.attr('name')
            infer_shapes[clip_name] = infer_shapes[name]
            features[clip_name] = ft
            precs[clip_name] = {OUT_KEY: tight_prec}
            if name in precs and name in precs[name]:
                oprec = precs[name][name]
                del precs[name][name]
                precs[clip_name][clip_name] = oprec
            buffers[clip_name] = buf
            cfg_dict[clip_name] = cfg_dict[name]

        return op

    sym, params = topo_visit_transformer(symbol, params,
            _quant,
            infer_shapes=infer_shapes, features=features,
            precs=precs, buffers=buffers, cfg_dict=cfg_dict,
            op_input_precs=op_input_precs,
            shift_bits=shift_bits,
            softmax_lambd=softmax_lambd)

    def quantize_output(op, **kwargs):
        name = op.attr('name')
        features = kwargs['features']
        precs, buffers = kwargs['precs'], kwargs['buffers']

        # Requantize output symbol
        if name in precs and name in precs[name]:
            oprec = precs[name][name]
            ft = features[name]
            assert ft.name == FT_TYPE_EXP
            oscale = scale_exp(ft.get(), oprec)
            quant = get_quantizer_exp()
            op, oprec, oscale = quant.quantize(
                op, oprec, oscale=oscale, oname=name, **kwargs)

            oname = op.attr('name')
            features[oname] = features[name]
            precs[oname] = oprec
            buffers[oname] = get_buffer_exp(oscale)
        return op

    return topo_visit_transformer(sym, params,
            quantize_output, features=features,
            precs=precs, buffers=buffers,
            cfg_dict=cfg_dict,
            shift_bits=shift_bits,
            softmax_lambd=softmax_lambd)

@N.register_nm("prepare_for_compile")
def prepare_for_compile(symbol, params):
    infer_shapes = infer_shape(symbol, params)
    return topo_visit_transformer(symbol, params,
            apply_pass("prepare_for_compile", infer_shapes=infer_shapes))
