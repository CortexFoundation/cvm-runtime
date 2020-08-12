""" Optimizor and Sampler definition for MRT calibration.
    Feature Data Types Definition For MRT calibration and quantization.
"""

from mxnet import ndarray as nd
import numpy as np

from .sym_utils import topo_visit_transformer

#----------------------------
# Feature Registration
#----------------------------

FT_REG = {
    # "absmax": AbsmaxLayerFeature,
    # "absmax_ch": AbsmaxChannelFeature,
    # "minmax": MinMaxLayerFeature,
    # "minmax_ch": MinMaxChannelFeature,
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
            " base `get_val` function defined in Feature")


@register_featrue("absmax")
class AbsmaxLayerFeature(Feature):
    """ Feature for symmetric layer-wise granularity
    """
    def __init__(self, absmax):
        self.absmax = absmax

    def get_feature(self):
        return self.absmax


@register_feature("absmax_ch")
class AbsmaxChannelFeature(Feature):
    """ Feature for symmetric channel-wise granularity
    """
    def __init__(self, absmax):
        self.absmax = absmax

    def get_feature(self):
        return self.absmax


@register_feature("minmax")
class MinMaxLayerFeature(Feature):
    """ Feature for zero point layer-wise granularity
    """
    def __init__(self, minv, maxv):
        self.minv = maxv
        self.maxv = maxv

    def get_feature(self):
        return self.minv, self.maxv

@register_feature("minmax_ch")
class MinMaxChannelFeature(Feature):
    """ Feature for zero point channel-wise granularity
    """
    def __init__(self, minv, maxv):
        self.minv = maxv
        self.maxv = maxv

    def get_feature(self):
        return self.minv, self.maxv

#----------------------------
# Optimizor Registration
#----------------------------

_NoneType = type(None)

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

    def get_opt(self, ft, out, hft=None, **kwargs):
        logger = kwargs.get(
            "logger", logging.getLogger("mrt.calibrate.optimize"))
        name = kwargs.get("name", "<unspecified>")

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
        return {"lambd": [_NoneType, float]}


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
        return ["absmax", "absmax_ch", "minmax", "minmax_ch"]

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
# Sampler Registration
#----------------------------

SMP_REG = {
    # "absmax": AbsmaxLayerSampler,
    # "absmax_ch": AbsmaxChannelSampler,
    # "minmax": MinMaxLayerSampler,
    # "minmax_ch": MinMaxChannelSampler,
}

DEFAULT_SMP_INFO = ("absmax")

DEFAULT_SAMPLER = AbsmaxLayerSampler()

SMP_INSTANCES = {
    DEFAULT_SMP_INFO: DEFAULT_SAMPLER,
    # ("absmax"): AbsmaxLayerSampler(),
    # ("absmax_ch", "ich", 1): AbsmaxChannelSampler(),
    # ("absmax_ch", "ich", 2): AbsmaxChannelSampler(ich=2),
}

def register_sampler(name):
    def _wrapper(sampler):
        sampler.name = name
        if name in FT_REG:
            raise NameError(
                "Sampler" + name + " has been registered")
        SMP_REG[name] = sampler
        return sampler
    return _wrapper


class Sampler:
    name = None

    def __init__(self, **attrs):
        for attr, value in attrs.items():
            setattr(self, attr, value)

    def sample(self, out):
        raise NotImplementedError(
            "Derived " + self.name + " sampler not override the" + \
            " base `sample` function defined in Sampler")

    @staticmethod
    def list_supported_features():
        raise NotImplementedError(
            "Derived " + self.name + " sampler not override the" + \
            " base `list_supported_features` function defined in Sampler")

    @staticmethod
    def list_attr_types():
        return {}


@register_sampler("absmax")
class AbsmaxLayerSampler(Sampler):
    """ Sampler for symmetric layer-wise granularity
    """
    def __init__(self, **attrs):
        super().__init__(**attrs)

    def sample(self, out):
        return out.abs().max().asscalar()


@register_sampler("absmax_ch")
class AbsmaxChannelSampler(Sampler):
    """ Sampler for symmetric channel-wise granularity
    """
    ich = 1 # Optimizor parameter, the axis id of channel

    def __init__(self, **attrs):
        super().__init__(**attrs)

    def sample(self, out):
        return out.abs().max(axis=self.ich).asscalar()

    @staticmethod
    def list_attr_types():
        return {"ich": [int]}


@register_sampler("minmax")
class MinMaxLayerSampler(Sampler):
    """ Sampler for zero_point channel-wise granularity
    """
    def __init__(self, **attrs):
        super().__init__(**attrs)

    def sample(self, out):
        return out.max().asscalar(), out.min().asscalar()


@register_sampler("minmax_ch")
class MinMaxChannelSampler(Sampler):
    """ Sampler for zero_point channel-wise granularity
    """
    ich = 1 # Optimizor parameter, the axis id of channel

    def __init__(self, **attrs):
        super().__init__(**attrs)

    def sample(self, out):
        return out.max(axis=self.ich).asscalar(), \
            out.min(axis=self.ich).asscalar()

    @staticmethod
    def list_attr_types():
        return {"ich": [int]}

#----------------------------
# Module calbration interfaces
#----------------------------

def sample(
    out, ft_type=DEFAULT_FT_TYPE, smp_info=DEFAULT_SMP_INFO,
    opt_info=DEFAULT_OPT_INFO, **kwargs):
    """ Interface for MRT calibration Sampling
    """
    name = kwargs.get("name", "<unspecified>")
    if not isinstance(out, nd.NDArray):
        raise TypeError(
            "Unsupported data type: %s" % (type(out), name))
    sample = SMP_INSTANCES[smp_type].sample(out)
    ft = FT_REG[ft_type](**sample)
    opt = OPT_INSTANCES[opt_info].get_opt(ft, out, **kwargs)
    return opt

def sym_calibrate_gen(symbol, params, data, **kwargs):
    """ Customized graph-level topo pass definition.

        Generized MRT calibration framework pass.
    """
    ft_dict = {}
    # TODO(archRev): implementation
    return ft_dict

#----------------------------
# Module main2 interfaces
#----------------------------

def sym_config_info(symbol, params, cfg_dict=None, logger=logging, **kwargs):
    """ Customized graph-level topo pass definition.

        Interface for MRT main2 configuration
        Create customized samplers and optimizors.
    """

    def _extract_attr(info):
        if not info:
            return {}
        return {v[i]: v[i+1] for i in range(0, len(info), 2)}

    cfg_dict, syms_set = {} if cfg_dict is None else cfg_dict, {}

    def _impl(sym, params, **kwargs):
        name = sym.attr("name")
        cfg_info = cfg_dict.get(name, {})
        syms_set.add(name)

        # feature
        ft_type = cfg_info.get("ft_type", DEFAULT_FT_TYPE)

        # sampler
        smp_info = cfg_info.get("smp_info", DEFAULT_SMP_INFO)
        smp_type = smp_info[0]
        if smp_info not in SMP_INSTANCES:
            smp_attrs = _extract_attr(smp_info[1:])
            SMP_INSTANCES[smp_info] = SMP_REG[smp_type](**smp_attrs)

        # optimizor
        opt_info = cfg_info.get("opt_type", DEFAULT_OPT_INFO)
        opt_type = opt_info[0]
        if opt_info not in OPT_INSTANCES:
            opt_attrs = _extract_attr(opt_info[1:])
            OPT_INSTANCES[opt_info] = OPT_REG[opt_type](**opt_attrs)

        cfg_dict[name] = cfg_info if cfg_info else \
            {"ft_type": ft_type, "smp_info": smp_info, "opt_info", opt_info}

    sym, params = topo_visit_transformer(symbol, params, _impl, **kwargs)
    syms_notset = {}
    for name in cfg_dict.keys():
        if name not in syms_set:
            del cfg_dict[name]
            syms_notset.add(name)
    if syms_notset:
        logger.warn(
            "Symbols (names: %s) not found in graph." + \
            "Please double check config file (.ini)." % syms_notset)
    return sym, params

def deserialize(val_dict):
    """ Interface for MRT main2 configuration

        Check the validity and compatibility of feature, sampler and optimizor configurations.
    """
    def _deserialize(val):
        # TODO(archRev): implementation
        return {}

    cfg_dict = {}
    for name, val in val_dict.items():
        cfg_info = _deserialize(val)

        # feature
        ft_type = cfg_info.get("ft_type", DEFAULT_FT_TYPE)
        if ft_type not in FT_REG:
            raise TypeError(
                "Unsupported feature type: %s, name: %s" % (ft_type, name))

        # sampler
        smp_info = cfg_info.get("smp_info", DEFAULT_SMP_INFO)
        smp_type = smp_info[0]
        if smp_type not in SMP_REG:
            raise TypeError(
                "Unsupported sampler type: %s, name: %s" % (smp_type, name))
        if ft_type not in SMP_REG[smp_type].list_supported_features():
            raise ValueError(
                "Feature type: (%s) is supported by sampler type: (%s)" % \
                (ft_type, smp_type))

        # optimizor
        opt_info = cfg_info.get("opt_type", DEFAULT_OPT_INFO)
        opt_type = opt_info[0]
        if opt_type not in OPT_REG:
            raise TypeError(
                "Unsupported optimizor type: %s, name: %s" % (opt_type, name))
        if ft_type not in OPT_REG[opt_type].list_supported_features():
            raise ValueError(
                "Feature type: (%s) is supported by sampler type: (%s)" % \
                (ft_type, opt_type))

        cfg_dict[name] = cfg_info
    return cfg_dict
