""" Optimizor and Sampler definition for MRT calibration.
    Feature Data Types Definition For MRT calibration and quantization.
"""

from mxnet import ndarray as nd
import numpy as np

#----------------------------
# Optimizor Registration
#----------------------------

_NoneType = type(None)

OPT_REG = {
    # e.g.
    # "hv": AbsmaxLayerOptimizor,
    # "ma": MovingAverageOptimizor,
    # "kld": KLDivergenceOptimizor,
    # "or": OutlierRemovalOptimizor,
}

OPT_INSTANCES = {
    # e.g.
    # ("hv", "lambd", None): HVOptimizor(),
    # ("hv", "lambd", 25): HVOptimizor(lambd=25),
    # ("ma","c", 0.01): MAOptimizor(),
    # ("kld", "eps", 0.05)
}

DEFAUT_OPT = ("hv", "lambd", None)

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
    """ Currently supported optimizor types for sampling:
            1. historical value
            2. moving average
            3. kl divergence

        Optimizor types to be implemented:
            1. outlier removal

        Notice:
            The users can implement customized optimizors.
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
    def list_attrs():
        return {}


@register_optimizor("hv")
class HVOptimizor(Optimizor):
    """ Generalized historical value optimizor
    """
    lambd = None # hyperparameter for fine tuning

    def __init__(self, **attrs):
        super().__init__(**attrs)

    def get_opt(self, ft, out, hft=None,
        logger=logging.getLogger("mrt.calibrate.optimize")):

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
            opt = AbsmaxLayerFeature(max(habsmax, absmax))
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
    def list_attrs():
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
    def list_attrs():
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
    def list_attrs():
        return {"bucket_bit": [int], "quant_bit": [int], "eps": [float]}


@register_optimizor("or")
class OROptimizor(Optimizor):
    pass

#----------------------------
# Sampler Registration
#----------------------------

SAMP_REG = {
    # e.g.
    # "absmax": AbsmaxLayerSampler,
    # "absmax_ch": AbsmaxChannelSampler,
    # "minmax": MinMaxLayerSampler,
    # "minmax_ch": MinMaxChannelSampler,
}

SAMP_INSTANCES = {
    # e.g.
    # ("absmax"): AbsmaxLayerSampler(),
    # ("absmax_ch", 1): AbsmaxChannelSampler(),
    # ("absmax_ch", 2): AbsmaxChannelSampler(ch=2),
}

DEFAUT_SAMP = ("absmax")

def register_sampler(name):
    def _wrapper(sampler):
        sampler.name = name
        if name in FT_REG:
            raise NameError(
                "Sampler" + name + " has been registered")
        SAMP_REG[name] = sampler
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
    def list_attrs():
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
    def list_attrs():
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
    def list_attrs():
        return {"ich": [int]}

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
# Module interfaces
#----------------------------

def sample(sym, infos):
    """ Interface for MRT calibration Sampling
    """
    name = sym.attr("name")
    ft_type, samp_type, opt_type = \
        infos["ft_type"], infos["samp_type"], infos["opt_type"]

def _make_key(name, **attrs):
    lst = [name]
    for k, v in attrs.items():
        lst.append(k)
        lst.append(v)
    return tuple(lst)

def config():
    """ Interface for MRT main2 configuration
    """
    if ft_type not in FT_REG:
        raise TypeError(
            "Unsupported feature type: %s, name: %s" % (ft_type, name))

    if samp_type[0] not in SAMP_REG:
        raise TypeError(
            "Unsupported sampler type: %s, name: %s" % (samp_type, name))

    if opt_type[0] not in OPT_REG:
        raise TypeError(
            "Unsupported optimizor type: %s, name: %s" % (opt_type, name))


