""" Optimizor and Sampler definition for MRT calibration.
    Quantizer definition for MRT calibration.
    Feature Types Definition For MRT calibration and quantization.
    Quant Types Definition For MRT quantization.
"""

import math
import logging

from mxnet import ndarray as nd
import numpy as np
import json

from mrt import sym_utils as sutils
from mrt.tfm_base import N

_NULL_NAME = "_NULL_NAME_"
_RES_NAME = "_RES_"
_NONETYPE = type(None)
_NONETYPE_NAME = "_NoneType"

#----------------------------
# Feature Types Definition
#----------------------------

INFO_REG = {
    # "us": UniformSymmetricInfo,
    # "ua": UniformAffineInfo,
}

def register_info(name):
    def _wrapper(info):
        info.name = name
        if name in INFO_REG:
            raise NameError(
                "QuantInfo" + name + " has been registered")
        INFO_REG[name] = info
        return info
    return _wrapper


class QuantInfo:
    """
        out -> features

        Parameters & Inputs
        -> Quantize()
            : Weight -> Int8

        QuantizeExtraInfo()
        f = Feature(out)
                                S, P
        S2, P2, info = f.Quantize(S1, P1, info=default,
                target_precision, target_scale)

        S, P = f.Quantize(S, P, scale, target_precision)

        - add new operators

        1. naive: abs(max(out)) = opt_value > [-opt_value, opt_value]
            -> [-127, 127]
            scale.shape = (1,)
        2. outlier remove -> opt_value < abs(max(out)) -> [-127, 127]:
            KMeans methods
        3. out -> [minV, maxV] -> [-r, r]
            zero_point = (minV + maxV) / 2
            (out - zero_point) <> [-r, r] -> [-127, 127]
        4. layerwise-quantize: out -> (N, C, H, W) * scale -> [-127, 127]
            -> layer (N, i, H, W) -> opt_value, -> [-127, 127]:
            image classification or detection
            mobilenet -> significant improve
            [opt_value_i] -> (1, C, 1, 1) >> cvm_right_shift
            scale.shape = (out[1],)
            -> precision
        5. for i in out -> [-127, 127]
            scale.shape = out.shape
    """
    name = None

    def set_feature(self, *args):
        raise NotImplementedError(
            "Derived " + self.name + " feature not override the" + \
            " base `set_feature` function defined in QuantInfo")

    def get_feature(self):
        raise NotImplementedError(
            "Derived " + self.name + " feature not override the" + \
            " base `get_feature` function defined in QuantInfo")

    def set_buf(self, p, sc=None):
        """ Set scale or zero point for symbol before quantization
        """
        raise NotImplementedError(
            "Derived " + self.name + " feature not override the" + \
            " base `set_buf` function defined in QuantInfo")

    def get_buf(self):
        raise NotImplementedError(
            "Derived " + self.name + " feature not override the" + \
            " base `get_buf` function defined in QuantInfo")

    def quantize(self, sym, p, sc=None, **kwargs):
        if sutils.is_params(sym, kwargs["params"]):
            return self._quantize_parameter(sym, p, sc=sc, **kwargs)
        return self._quantize_operator(sym, p, sc=sc, **kwargs)

    def _quantize_parameter(self, sym, p, sc=None, **kwargs):
        raise NotImplementedError(
            "Derived " + self.name + " quantizer not override the" + \
            " base `_quantize_parameter` function " + \
            "defined in QuantInfo")

    def _quantize_operator(self, sym, p, sc=None, **kwargs):
        raise NotImplementedError(
            "Derived " + self.name + " quantizer not override the" + \
            " base `_quantize_opoerator` function " + \
            "defined in QuantInfo")

    @staticmethod
    def sample(out):
        raise NotImplementedError(
            "Derived " + self.name + " feature not override the" + \
            " base `sample` function defined in QuantInfo")


@register_featrue("us")
class UniformSymmetricInfo(QuantInfo):
    """ Information data type for uniform symmetric quantizaton
    """
    def __init__(self, absmax):
        self.absmax = None
        self.sc = 1

    def set_feature(self, *args):
        self.absmax = args[0]

    def get_feature(self):
        return self.absmax

    def get_range(self):
        return get_feature()

    def set_buf(self, p, sc=None):
        self.sc = (2**(p-1)-1) / self.absmax \
            if sc is None else sc

    def get_buf(self):
        return self.sc

    def _quantize_parameter(self, sym, p, sc=None, **kwargs):
        return W, wp, wsc

    def _quantize_operator(self, sym, p, sc=None, **kwargs):
        return X, xp, xsc

    @staticmethod
    def sample(out):
        return [out.abs().max().asscalar()]


@register_feature("ua")
class UniformAffineInfo(QuantInfo):
    """ Information data type for uniform affine quantizaton
    """
    def __init__(self):
        self.minv = None
        self.maxv = None
        self.sc = 1
        self.zp = 0

    def set_feature(self, *args):
        self.minv, self.maxv = args[0], args[1]

    def get_feature(self):
        return self.minv, self.maxv

    def get_range(self):
        return self.maxv - self.minv

    def set_buf(self, p):
        self.sc = (2**p-1) / (self.maxv-self.minv) \
            if sc is None else sc
        self.zp = math.ceil(self.sc * self.minv)

    def get_buf(self):
        return self.sc, self.zp

    def _quantize_parameter(self, sym, p, **kwargs):
        return W, wp, wsc

    def _quantize_operator(self, sym, p, **kwargs):
        return X, xp, xsc

    @staticmethod
    def sample(out):
        return [out.max().asscalar(), out.min().asscalar()]

#----------------------------
# Optimizor Registration
#----------------------------

OPT_REG = {
    # "hv": AbsmaxLayerOptimizor,
    # "ma": MovingAverageOptimizor,
    # "kld": KLDivergenceOptimizor,
    # "or": OutlierRemovalOptimizor,
}

DEFAULT_OPT_INFO = ("hv", "lambd", None)

DEFAULT_OPTIMIZOR = HVOptimizor()

OPT_INSTANCES = {
    DEFAULT_OPT_INFO: DEFAULT_OPTIMIZOR,
    # ("hv", "lambd", 25): HVOptimizor(lambd=25),
    # ("ma","c", 0.01): MAOptimizor(),
    # ("kld", "eps", 0.05)
}

def register_optimizor(name):
    def _wrapper(optimizor):
        optimizor.name = name
        if name in OPT_REG:
            raise NameError(
                "Optimizor" + name + " has been registered")
        OPT_REG[name] = optimizor
        return optimizor
    return _wrapper


class Optimizor:
    """ Currently supported optimizor types intended for sampling:
            1. historical value
            2. moving average
            3. kl divergence

        Optimizor types to be implemented:
            1. outlier removal

        Notice:
            The users can implement customized optimizors with respect to the features.
            e.g. Designing different optimizors for different components of the feature.
    """
    name = None

    def __init__(self, **attrs):
        for attr, value in attrs.items():
            setattr(self, attr, value)

    def get_opt(self, ft, out, **kwargs):
        raise NotImplementedError(
            "Derived " + self.name + " optimizor not override the" + \
            " base `get_opt` function defined in Optimizor")

    @staticmethod
    def list_supported_features():
        raise NotImplementedError(
            "Derived " + self.name + " optimizor not override the" + \
            " base `list_supported_features` function defined in Optimizor")

    @staticmethod
    def list_attr_types():
        return {}


@register_optimizor("hv")
class HVOptimizor(Optimizor):
    """ Generalized historical value optimizor
    """
    lambd = None # hyperparameter for fine tuning

    def __init__(self, **attrs):
        super().__init__(**attrs)

    def get_opt(self, ft, out, hft=None, name="[not specified]", **kwargs):
        logger = kwargs.get(
            "logger", logging.getLogger("mrt.calibrate.optimize"))

        if hft is None:
            return ft
        if isinstance(ft, AbsmaxLayerFeature):
            # hyperparameter 'lambd' for fine tuning
            absmax = ft.get_feature()
            habsmax = hft.get_feature()
            if self.lambd is not None:
                mean = nd.mean(out).asscalar()
                sqrt_n = math.sqrt(np.product(out.shape))
                std = nd.norm(out-mean).asscalar() / sqrt_n
                alpha = abs(mean) + lambd*std
                nabsmax = alpha if alpha < 0.95*absmax else absmax
            # historical feature update
            opt_absmax = max(habsmax, absmax)
            p = logger.debug if opt_absmax < 30 else logger.warn
            p("collect symbol %-40s, out_shape=%-20s, opt: (%s)",
              name, out.shape, opt_absmax)
            opt = AbsmaxLayerFeature(opt_absmax)
        elif isinstance(ft, AbsmaxChannelSampler):
            absmax = ft.get_feature()
            habsmax = ft.get_feature()
            opt = AbsmaxChannelFeature(nd.broadcast_maximum(absmax, habsmax))
        elif isinstance(ft, MinMaxLayerFeature):
            minv, maxv = ft.get_feature()
            hminv, hmaxv = hft.get_feature()
            opt = MinMaxLayerFeature(min(minv, hminv), max(maxv, hmaxv))
        elif isinstance(ft, MinMaxChannelFeature):
            minv, maxv = ft.get_feature()
            hminv, hmaxv = hft.get_feature()
            opt = MinMaxChannelFeature(
                nd.broadcast_minimum(minv, hminv),
                nd.broadcast_maximum(maxv, hmaxv))
        else:
            raise TypeError(
                "Unsupported feature type: %s for HVOptimizor" % type(f))
        return opt

    @staticmethod
    def list_supported_features():
        return ["absmax", "absmax_ch", "minmax", "minmax_ch"]

    @staticmethod
    def list_attr_types():
        return {"lambd": [_NONETYPE, float]}


@register_optimizor("ma")
class MAOptimizor(Optimizor):
    """ Generalized moving average optimizor
    """
    c = 0.01 # hyperparameter for moving average

    def __init__(self, **attrs):
        super().__init__(**attrs)

    def get_opt(self, ft, out, hft=None):
        if hf is None:
            return f
        if isinstance(ft, AbsmaxLayerFeature):
            absmax = f.get_feature()
            habsmax = ft.get_feature()
            opt = AbsmaxLayerFeature((1-self.c)*habsmax + self.c*absmax)
        elif isinstance(ft, MinMaxLayerFeature):
            absmax = ft.get_feature()
            habsmax = ft.get_feature()
            opt = AbsmaxChannelFeature((1-self.c)*habsmax + self.c*absmax)
        elif isinstance(ft, MinMaxLayerFeature):
            minv, maxv = ft.get_feature()
            hminv, hmaxv = hft.get_feature()
            opt = MinMaxLayerFeature(
                (1-self.c)*hminv + self.c*minv,
                (1-self.c)*hmaxv + self.c*maxv)
        elif isinstance(ft, MinMaxChannelFeature):
            minv, maxv = ft.get_feature()
            hminv, hmaxv = hft.get_feature()
            opt = MinMaxChannelFeature(
                (1-self.c)*hminv + self.c*minv,
                (1-self.c)*hmaxv + self.c*maxv)
        else:
            raise TypeError(
                "Unsupported feature type: %s for MAOptimizor" % type(f))
        return opt

    @staticmethod
    def list_supported_features():
        return ["absmax", "minmax"]

    @staticmethod
    def list_attr_types():
        return {"c": [float]}


@register_optimizor("kld")
class KLDOptimizor(Optimizor):
    """ KL divergence optimizor for AbsmaxLayerFeature
    """
    # Optimizor parameter for kl divergence
    bucket_bit = 12
    quant_bit = 8
    # Optimizor parameter for smooth distribution
    eps = 0.0001

    def __init__(self, **attrs):
        super().__init__(**attrs)

    def _smooth_distribution(self, p):
        is_zeros = (p == 0).astype(np.float32)
        is_nonzeros = (p != 0).astype(np.float32)
        n_zeros = is_zeros.sum()
        n_nonzeros = p.size - n_zeros
        if not n_nonzeros:
            raise ValueError('The discrete probability distribution is malformed. All entries are 0.')
        eps1 = self.eps * float(n_zeros) / float(n_nonzeros)
        assert eps1 < 1.0, 'n_zeros=%d, n_nonzeros=%d, eps1=%f' % (n_zeros, n_nonzeros, eps1)
        hist = p.astype(np.float32)
        hist += self.eps * is_zeros + (-eps1) * is_nonzeros
        assert (hist <= 0).sum() == 0
        return hist

    def _kldiverge(self, absmax, out):
        num_bins, num_quantized_bins = (1 << self.bucket_bit) - 1, (1 << self.quant_bit) - 1
        hist, hist_edges = np.histogram(out, bins=num_bins, range=(-absmax, absmax))
        zero_bin_idx = num_bins // 2
        num_half_quantized_bins = num_quantized_bins // 2

        step = 1
        thresholds = np.zeros((zero_bin_idx - num_half_quantized_bins) // step + 1)
        divergence = np.zeros_like(thresholds)
        quantized_bins = np.zeros(num_quantized_bins, dtype=np.int32)

        table = np.zeros(hist.size+1)
        for i in range(1, table.size):
            table[i] = table[i-1] + hist[i-1]

        for i in range(num_half_quantized_bins,
                       zero_bin_idx+1, step):
            p_bin_idx_start = zero_bin_idx - i
            p_bin_idx_stop = zero_bin_idx + i + 1
            thresholds[(i-num_half_quantized_bins) // step] = hist_edges[p_bin_idx_stop]
            sliced_nd_hist = hist[p_bin_idx_start:p_bin_idx_stop]

            p = sliced_nd_hist.copy()
            p[0] += table[p_bin_idx_start] - table[0]
            p[-1] += table[-1] - table[p_bin_idx_stop]
            is_nonzeros = (p != 0).astype(np.int32)

            num_merged_bins = sliced_nd_hist.size // num_quantized_bins
            for j in range(num_quantized_bins):
                start = p_bin_idx_start + j * num_merged_bins
                stop = start + num_merged_bins
                quantized_bins[j] = table[stop] - table[start]
            quantized_bins[-1] += table[p_bin_idx_stop] - table[p_bin_idx_start +
                   num_quantized_bins * num_merged_bins]

            expand_bins = sliced_nd_hist.size / num_quantized_bins
            q = np.zeros(sliced_nd_hist.size, dtype=np.float32)
            for j in range(num_quantized_bins):
                start = j * num_merged_bins
                if j == num_quantized_bins - 1:
                   stop = len(is_nonzeros)
                else:
                   stop = start + num_merged_bins
                norm = is_nonzeros[start:stop].sum()
                if norm != 0:
                    q[start:stop] = float(quantized_bins[j]) / float(norm)
            q[p == 0] = 0
            p = self._smooth_distribution(p)
            try:
                q = self._smooth_distribution(q)
            except ValueError:
                divergence[(i-num_half_quantized_bins) // step] = float("inf")
            divergence[(i-num_half_quantized_bins) // step] = stats.entropy(p, q)

        min_divergence_idx = np.argmin(divergence)
        opt_th = thresholds[min_divergence_idx]
        return opt_th

    def get_opt(self, ft, out, hft=None):
        if not isinstance(ft, AbsmaxLayerFeature):
            raise TypeError(
                "KLDOptimizor do not support feature type: %s, " + \
                "only AbsmaxLayerFeature is supported" % type(f))

        absmax = ft.get_feature()
        kval = self._kldiverge(absmax, out)
        opt = kval if hft is None else max(kval, hft.get_feature())
        return opt

    @staticmethod
    def list_supported_features():
        return ["absmax"]

    @staticmethod
    def list_attr_types():
        return {"bucket_bit": [int], "quant_bit": [int], "eps": [float]}


@register_optimizor("or")
class OROptimizor(Optimizor):
    pass

#----------------------------
# Module main2 interfaces
#----------------------------

def sym_config_infos(symbol, params, cfg_dict=None, logger=logging, **kwargs):
    """ Customized graph-level topo pass definition.

        Interface for MRT main2 configuration
        Create customized samplers and optimizors.

        Use it just before calibration.
    """
    names = set()

    def _collect_names(symbol, params):
        names.add(symbol.attr("name"))

    sutils.topo_visit_transformer(
        symbol, params, _collect_names, **kwargs)
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
            "Please double check config file (.ini)." % \
            noncfgs)
    if _RES_NAME in cfg_dict:
        cfg_info = cfg_dict.pop(_RES_NAME)
        keys = cfg_dict.keys()
        for name in [n for n in names if n not in keys]:
            cfg_dict[name] = cfg_info

    def _sym_config_infos(sym, params, **kwargs):
        name = sym.attr("name")
        cfg_info = cfg_dict.get(name, {})
        syms_set.add(name)

        # feature
        ft_type = cfg_info.get("ft_type", DEFAULT_FT_TYPE)

        # optimizor
        opt_info = cfg_info.get("opt_type", DEFAULT_OPT_INFO)
        opt_type = opt_info[0]
        if opt_info not in OPT_INSTANCES:
            opt_attrs = {} if len(opt_info) == 1 else \
                {v[i]: v[i+1] for i in range(1, len(opt_info), 2)}
            OPT_INSTANCES[opt_info] = OPT_REG[opt_type](**opt_attrs)

        cfg_dict[name] = cfg_info if cfg_info else \
            {"ft_type": ft_type, "smp_type": smp_type, "opt_info", opt_info}

    sutils.topo_visit_transformer(
        symbol, params, _sym_config_infos, **kwargs)
    return cfg_dict

def deserialize(val_dict):
    """ Interface for MRT main2 configuration

        Check the validity and compatibility of feature, sampler and optimizor configurations.
    """

    cfg_dict = {}
    for val, names in val_dict.items():
        val = val if val else "{}"
        cfg_info = json.loads(val)

        # feature
        ft_type = cfg_info.get("ft_type", DEFAULT_FT_TYPE)
        if ft_type not in FT_REG:
            raise TypeError(
                "Unsupported feature type: %s, names: %s" % \
                (ft_type, names))

        # optimizor
        opt_info = cfg_info.get("opt_type", DEFAULT_OPT_INFO)
        opt_type = opt_info[0]
        if opt_type not in OPT_REG:
            raise TypeError(
                "Unsupported optimizor type: %s, names: %s" % \
                (opt_type, names))
        if ft_type not in OPT_REG[opt_type].list_supported_features():
            raise ValueError(
                "Feature type: (%s) is not supported by " + \
                "optimizor type: (%s), names: %s" % \
                (ft_type, opt_type, names))
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
            for dtype in [k]:
            if not any([isinstance(v, dtype) for dtype in dtypes]):
                raise TypeError(
                    "Attribute: (%s) dtype: (%s) is not compatible " + \
                    "with any of supported dtypes: (%s), names: %s, " + \
                    "optimizor type: %s" % \
                    (k, type(v), dtypes, names, opt_type))

    return cfg_dict

#----------------------------
# Module calibrate interfaces
#----------------------------

def sample(out, ft_type, opt_info, **kwargs):
    """ Interface for MRT generalized calibration Sampling
    """
    if not isinstance(out, nd.NDArray):
        raise TypeError("Unsupported data type: %s" % type(out))
    smp = FT_REG[ft_type].sample(out)
    ft = FT_REG[ft_type](**smp)
    return OPT_INSTANCES[opt_info].get_opt(ft, out, **kwargs)

def sym_calibrate_gen(symbol, params, data, **kwargs):
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
        deps, old_ths = kwargs['deps'], kwargs['old_ths']
        logger = logging.getLogger('log.mrt.calibrate')
        name, op_name = op.attr('name'), op.attr('op_name')
        childs, attr = sutils.sym_iter(
            op.get_children()), op.list_attr()
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
        hft = features[name] if name in features else None
        ft_type = cfg_dict[name].get("ft_type")
        opt_info = cfg_dict[name].get("opt_info")
        features[name] = sample(out[0], ft_type, opt_info, hft=hft)

    sutils.topo_visit_transformer(
        symbol, nparams, _impl, logger=logger,
        deps=deps, data=data, **kwargs)
    out_cache.clear()

    return features

#----------------------------
# Module quantize interfaces
#----------------------------

@N.register_nm("quantize")
def sym_quantize_gen(
    symbol, params, features, precs, buffers, op_input_precs,
    restore_names, shift_bits, softmax_lambd):
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
            th_dict=kwargs['th_dict'],
        )(op, **kwargs) if op.attr('name') not in restore_names \
            else restore(op, **kwargs)

        if is_var(op, kwargs['params']):
            return op

        name = op.attr('name')
        th_dict, scales = kwargs['th_dict'], kwargs['scales']
        precs = kwargs['precs']
        th = th_dict[name]
        scale = scales[name]
        tight_prec = get_bit(th_dict[name] * scales[name])
        if precs[name][OUT_KEY] > tight_prec:
            op = mx.sym.Custom(op, precision=tight_prec,
                    name=N.n('clip'), op_type='cvm_clip')
            clip_name = op.attr('name')
            infer_shapes[clip_name] = infer_shapes[name]
            th_dict[clip_name] = th_dict[name]
            precs[clip_name] = { OUT_KEY: tight_prec }
            scales[clip_name] = scales[name]
            if name in precs and name in precs[name]:
                oprec = precs[name][name]
                del precs[name][name]
                precs[clip_name][clip_name] = oprec

        return op

    sym, params = topo_visit_transformer(symbol, params,
            _quant,
            infer_shapes=infer_shapes, th_dict=th_dict,
            precs=precs, scales=scales,
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
