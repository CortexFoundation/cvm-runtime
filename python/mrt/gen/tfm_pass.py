import logging

import mxnet as mx

from mrt.tfm_base import N
from mrt.tfm_pass import convert_params_dtype, infer_shape
from mrt.sym_utils import is_inputs, get_entry_id, is_var, \
                          get_nd_op, topo_visit_transformer
from mrt.gen.tfm_types import get_quantizer, DEFAULT_QUANT_TYPE, \
                              get_optimizor, DEFAULT_OPT_INFO
from mrt.gen.tfm_base import apply_pass

from mrt import sym_utils as sutils

#----------------------------
# Module main interfaces
#----------------------------

_RES_NAME = "_RES_"

def sym_config_infos(symbol, params, cfg_dict=None, logger=logging):
    """ Customized graph-level topo pass definition.

        Interface for MRT main2 configuration
        Create customized samplers and optimizors.

        Use it just before calibration.
    """
    names = set()

    def _collect_names(symbol, params, **kwargs):
        names.add(symbol.attr("name"))

    topo_visit_transformer(symbol, params, _collect_names)
    cfg_dict, noncfgs = {} if cfg_dict is None else cfg_dict, set()
    keys = cfg_dict.keys()
    for name in keys:
        if name == _RES_NAME:
            continue
        if name not in names:
            del cfg_dict[name]
            noncfgs.add(name)
    if noncfgs:
        logger.warn(
            "Symbols (names: %s) not found in graph." + \
            "Please double check config file (.ini)." % noncfgs)
    if _RES_NAME in cfg_dict:
        cfg_info = cfg_dict.pop(_RES_NAME)
        keys = cfg_dict.keys()
        for name in [n for n in names if n not in keys]:
            cfg_dict[name] = cfg_info

    def _sym_config_infos(sym, params, **kwargs):
        name = sym.attr("name")
        cfg_info = cfg_dict.get(name, {})

        quant_type = cfg_info.get("quant_type", DEFAULT_QUANT_TYPE)
        get_quantizer(quant_type)

        opt_info = cfg_info.get("opt_type", DEFAULT_OPT_INFO)
        get_optimizor(opt_info)

        cfg_dict[name] = cfg_info if cfg_info else \
            {"quant_type": quant_type, "opt_info": opt_info}

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
    for key, val in cfg_groups.items():
        key = key if key else "{}"
        cfg_info = json.loads(key)
        names = json.loads(val)

        # quantizer 
        quant_type = cfg_info.get("quant_type", DEFAULT_QUANT_TYPE)
        if quant_type not in QUANT_REG:
            raise TypeError(
                "Unsupported quantizer type: %s, names: %s" % \
                (quant_type, names))

        # optimizor
        opt_info = cfg_info.get("opt_info", DEFAULT_OPT_INFO)
        opt_type = opt_info[0]
        if opt_type not in OPT_REG:
            raise TypeError(
                "Unsupported optimizor type: %s, names: %s" % \
                (opt_type, names))
        if quant_type not in OPT_REG[opt_type].list_supported_features():
            raise ValueError(
                "quantizer type: (%s) is not supported by " + \
                "optimizor type: (%s), names: %s" % \
                (quant_type, opt_type, names))
        opt_attrs = {} if len(opt_info) == 1 else \
            {v[i]: v[i+1] for i in range(1, len(opt_info), 2)}
        opt_attr_types = OPT_REG[opt_type].list_attr_types()
        for k, v in opt_attr.items():
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

        for name in names:
            if name in cfg_dict:
                raise ValueError(
                    "Duplicate name: %s, parsed value: %s" % (name, val))
            cfg_dict[name] = cfg_info

    return cfg_dict

#----------------------------
# Module calibrate interfaces
#----------------------------

def sym_calibrate(symbol, params, data, cfg_dict, **kwargs):
    """ Customized graph-level topo pass definition.

        Generalized MRT calibration framework pass.
    """
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

#----------------------------
# Module quantize interfaces
#----------------------------

@N.register_nm("quantize")
def quantize(
    symbol, params, features, precs, buffers, cfg_dict,
    op_input_precs, restore_names, shift_bits, softmax_lambd):
    """ Customized graph-level topo pass definition.

        Generalized MRT quantization framework pass.
    """
    infer_shapes = infer_shape(symbol, params)

    def restore(op, **kwargs):
        features, precs, buffers = \
            kwargs['features'], kwargs['precs'], kwargs['buffers']
        name, op_name = op.attr('name'), op.attr('op_name')
        childs, attr = sym_iter(op.get_children()), op.list_attr()

        childs = [] if childs is None else childs

        buffers[c.attr("name")].get_scale()
        new_childs = [c / scales[c.attr('name')] \
            if scales.get(c.attr('name'), 1) != 1 else c \
                     for c in childs]

        out = get_mxnet_op(op_name)(*new_childs, **attr, name=name)
        precs[name][OUT_KEY] = get_bit(th_dict[name])
        scales[name] = 1

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
        assert ft.name == "Absmax"
        th = ft.get()
        buf = buffers[name]
        assert buf.name == "Scale"
        scale = buf.get()
        tight_prec = get_bit(th_dict[name] * scales[name])
        if precs[name][OUT_KEY] > tight_prec:
            op = mx.sym.Custom(op, precision=tight_prec,
                    name=N.n('clip'), op_type='cvm_clip')
            clip_name = op.attr('name')
            infer_shapes[clip_name] = infer_shapes[name]
            features[clip_name] = ft
            precs[clip_name] = { OUT_KEY: tight_prec }
            buffers[clip_name] = buf
            if name in precs and name in precs[name]:
                oprec = precs[name][name]
                del precs[name][name]
                precs[clip_name][clip_name] = oprec

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
        th_dict = kwargs['th_dict']
        precs, scales = kwargs['precs'], kwargs['scales']

        # Requantize output symbol
        if name in precs and name in precs[name]:
            oprec = precs[name][name]
            os = scale(th_dict[name], oprec)
            op, oprec, os = requant(op, oprec, os, oname=name, **kwargs)

            oname = op.attr('name')
            th_dict[oname] = th_dict[name]
            precs[oname] = oprec
            scales[oname] = os
        return op

    return topo_visit_transformer(sym, params,
            quantize_output, th_dict=th_dict,
            precs=precs, scales=scales,
            shift_bits=shift_bits,
            softmax_lambd=softmax_lambd)
