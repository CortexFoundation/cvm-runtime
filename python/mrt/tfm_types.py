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

_NULL_NAME = "_NULL_NAME_"
_RES_NAME = "_RES_"
_NONETYPE = type(None)
_NONETYPE_NAME = "_NoneType"

#----------------------------
# Feature Types Definition
#----------------------------

FT_REG = {
    # "absmax": AbsmaxFeature,
    # "minmax": MinMaxFeature,
}

def register_feature(name):
    def _wrapper(feature):
        feature.name = name
        if name in FT_REG:
            raise NameError(
                "Feature" + name + " has been registered")
        FT_REG[name] = feature
        return feature
    return _wrapper


class Feature:
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

    def get_feature(self):
        raise NotImplementedError(
            "Derived " + self.name + " feature not override the" + \
            " base `get_feature` function defined in Feature")

    @staticmethod
    def sample(out):
        raise NotImplementedError(
            "Derived " + self.name + " feature not override the" + \
            " base `sample` function defined in Feature")


@register_featrue("absmax")
class AbsmaxFeature(Feature):
    """ Feature for symmetric layer-wise granularity
    """
    def __init__(self, absmax):
        self.absmax = absmax

    def get_feature(self):
        return self.absmax

    @staticmethod
    def sample(out):
        return [out.abs().max().asscalar()]


@register_feature("minmax")
class MinMaxFeature(Feature):
    """ Feature for zero point layer-wise granularity
    """
    def __init__(self, minv, maxv):
        self.minv = maxv
        self.maxv = maxv

    def get_feature(self):
        return self.minv, self.maxv

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
# Module calbrate interfaces
#----------------------------

def sample(out, ft_type, opt_info, **kwargs):
    """ Interface for MRT calibration Sampling
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
    ft_dict, out_cache = {}, {}
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
        hft = ft_dict[name] if name in ft_dict else None
        ft_type = cfg_dict[name].get("ft_type")
        opt_info = cfg_dict[name].get("opt_info")
        ft_dict[name] = sample(out[0], ft_type, opt_info, hft=hft)

    sutils.topo_visit_transformer(
        symbol, nparams, _impl, logger=logger,
        deps=deps, data=data, **kwargs)
    out_cache.clear()

    return ft_dict

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

        for name in names:
            if name in cfg_dict:
                raise ValueError(
                    "Duplicate name: %s, parsed value: %s" % \
                    (name, val))
            cfg_dict[name] = cfg_info

    return cfg_dict

#----------------------------
# Quantized Buffer Definition
#----------------------------

BUF_REG = {
    # "symmetric": SymmetricBuf,
    # "affine": AffineBuf,
}

def register_buf(name):
    def _wrapper(buf):
        buf.name = name
        if name in BUF_REG:
            raise NameError(
                "Buf" + name + " has been registered")
        BUF_REG[name] = buf
        return buf
    return _wrapper


class Buf:
    name = None

    def get_buf(self):
        raise NotImplementedError(
            "Derived " + self.name + " buf not override the" + \
            " base `get_buf` function defined in Buf")


@register_buf("symmetric")
class SymmetricBuf(Buf):
    def __init__(self, sc):
        self.sc = sc

    def get_buf(self):
        return self.sc


@register_buf("affine")
class AffineBuf(Buf):
    def __init__(self, sc, zp):
        self.sc, self.zp = sc, zp

    def get_buf(self):
        return self.sc, self.zp

#----------------------------
# Quantizer Types Definition
#----------------------------

QUANT_REG = {
    # "usq": UniformSymmetricQuantizer,
    # "uaq": UniformAffineQuantizer,
    # "usgq": UniformSymmetricGroupQuantizer,
}

DEFAULT_QUANT_TYPE = "usq"

DEFAULT_QUANTIZER = UniformSymmetricQuantizer()

QUANT_INSTANCES = {
    DEFAULT_QUANT_TYPE: DEFAULT_QUANTIZER,
    # "uaq": UniformAffineQuantizer(),
    # "usgq": UniformSymmetricGroupQuantizer(),
}

def register_quantizer(name):
    def _wrapper(quantizer):
        quantizer.name = name
        if name in QUANT_REG:
            raise NameError(
                "Quantizer" + name + " has been registered")
        QUANT_REG[name] = quantizer
        return quantizer
    return _wrapper


class Quantizer:
    def quantize(self, oprec, **kwargs):
        raise NotImplementedError(
            "Derived " + self.name + " quantizer not override the" + \
            " base `quantize` function defined in Quantizer")

    def _quantize_symbol(self, symbol, oprec, buf, **kwargs):
        if sutils.is_params(symbol, kwargs["params"]):
            return self._quantize_parameter(symbol, oprec, buf, **kwargs)
        return self._quantize_operator(symbol, oprec, buf, **kwargs)

    def _quantize_parameter(self, symbol, oprec, buf, **kwargs):
        raise NotImplementedError(
            "Derived " + self.name + " quantizer not override the" + \
            " base `_quantize_parameter` function " + \
            "defined in Quantizer")

    def _quantize_operator(self, symbol, oprec, buf, **kwargs):
        raise NotImplementedError(
            "Derived " + self.name + " quantizer not override the" + \
            " base `_quantize_opoerator` function " + \
            "defined in Quantizer")

    @staticmethod
    def list_supported_features():
        raise NotImplementedError(
            "Derived " + self.name + " quantizer not override the" + \
            " base `list_supported_features` " + \
            "function defined in Quantizer")


@register_quantizer("usq")
class UniformSymmetricQuantizer(Quantizer):
    """ Uniform symmetric quantizer
    """
    def quantize(self, symbol, oprec, **kwargs):
        absmax = kwargs["ft_dict"][symbol.attr("name")].get_feature()
        sc = (2**(oprec-1)-1) / absmax
        buf = SymmetricBuf(sc)
        return self._quantize_symbol(symbol, oprec, buf, **kwargs)

    def _quantize_parameter(self, symbol, oprec, buf, **kwargs):
        return W, wprec, wscale

    def _quantize_operator(self, symbol, oprec, buf, **kwargs):
        return X, xprec, xscale

    @staticmethod
    def list_supported_features():
        return ["absmax"]


@register_quantizer("uaq")
class UniformAffineQuantizer(Quantizer):
    """ Uniform affine quantizer
    """
    def quantize(self, symbol, oprec, **kwargs):
        minv, maxv = \
            kwargs["ft_dict"][symbol.attr("name")].get_feature()
        sc, zp = (2**oprec-1) / (maxv-minv), math.ceil(sc*minv)
        buf = AffineBuf(sc, zp)
        return self._quantize_symbol(symbol, oprec, buf, **kwargs)

    def _quantize_parameter(self, symbol, oprec, buf, **kwargs):
        return W, wprec, wscale

    def _quantize_operator(self, symbol, oprec, buf, **kwargs):
        return X, xprec, xscale

    @staticmethod
    def list_supported_features():
        return ["minmax"]


@register_quantizer("usgq")
class UniformSymmetricGroupQuantizer(UniformSymmetricQuantizer):
    """ Uniform symmetric group quantizer
    """
    def quantize(self, symbol, oprec, **kwargs):
        absmax = max([kwargs["ft_dict"[sym.attr("name")].get_feature() \
            for sym in sutils.sym_iter(symbol)])
        sc = (2**(oprec-1)-1) / absmax
        buf = SymmetricBuf(sc)

        Xs, xprecs, xscales = [], [], []
        for sym in sutils.sym_iter(symbol):
            X, xprec, xscale = \
                self._quantize_symbol(sym, oprec, buf, **kwargs)
            Xs.append(X)
            xprecs.append(xprec)
            xscales.append(xscale)
        return Xs, xprecs, xscales

#----------------------------
# Module quantize interfaces
#----------------------------

def quantize_symbol(symbol, oprec, quant_type, **kwargs):
    return QUANT_INSTANCES[quant_type].quantize(
        symbol, oprec, **kwargs)

