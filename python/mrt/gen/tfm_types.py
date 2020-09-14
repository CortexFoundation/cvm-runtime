""" Optimizor definition for MRT calibration.
    Quantizer definition for MRT calibration and quantization.
    Feature Types Definition for MRT calibration and quantization.
    Buffers Types Definition for MRT quantization.
    Granularity constant vars definition.
"""

import math
import logging
import json

import mxnet as mx
from mxnet import ndarray as nd
import numpy as np
import json

from mrt.tfm_base import MAX_BIT, N
from mrt.tfm_pass import OUT_KEY

from mrt import sym_utils as sutils
from mrt import tfm_utils as tutils
from mrt import sim_quant_helper as sim

_NULL_NAME = "_NULL_NAME_"
_NONETYPE = type(None)

#----------------------------
# Feature Types Definition
#----------------------------

FT_REG = {
    # "Absmax": AFeature,
    # "AbsmaxList": ALFeature,
    # "MinMax": MMFeature,
}

def register_feature(name):
    def _wrapper(ft):
        ft.name = name
        if name in FT_REG:
            raise NameError("Feature" + name + " has been registered")
        FT_REG[name] = ft
        return ft
    return _wrapper


class Feature:
    """
        out -> features

        Parameters & Inputs
        -> Quantize()
        : Weight -> Int8

        QuantizeExtraInfo()
        f = Feature(out)

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

    def __init__(self, *args):
        raise NotImplementedError(
            "Derived " + self.name + " feature not override the" + \
            " base `__init__` function defined in Feature")

    def get(self):
        raise NotImplementedError(
            "Derived " + self.name + " feature not override the" + \
            " base `get` function defined in Feature")

    def serialize(self):
        raise NotImplementedError(
            "Derived " + self.name + " feature not override the" + \
            " base `toJSON` function defined in Feature")


@register_feature("Absmax")
class AFeature(Feature):
    """ Absmax Feature
    """
    def __init__(self, *args):
        assert len(args) == 1
        self.absmax = args[0]

    def get(self):
        return self.absmax

    def serialize(self):
        return [self.name, self.absmax]


@register_feature("AbsmaxList")
class ALFeatrue(Feature):
    """ Absmax Channel-wise Feature
    """
    def __init__(self, *args):
        self.absmax_list = args

    def get(self):
        return self.absmax_list

    def serialize(self):
        return [self.name, self.absmax_list]


@register_feature("MinMax")
class MMFeature(Feature):
    """ Min and Max Feature
    """
    def __init__(self, *args):
        assert len(args) == 2
        self.minv, self.maxv = args[0], args[1]

    def get(self):
        return self.minv, self.maxv

    def serialize(self):
        return [self.name, self.minv, self.maxv]

def get_feature(ft_type, *args):
    if ft_type not in FT_REG:
        raise TypeError("Unknown feature type: %20s", ft_type)
    return FT_REG[ft_type](*args)

#----------------------------
# Buffer Types Definition
#----------------------------

BUF_REG = {
    # "Scale": SBuffer,
    # "ScaleList": SLBuffer,
    # "ScaleZpoint": SZBuffer,
}

def register_buffer(name):
    def _wrapper(buf):
        buf.name = name
        if name in BUF_REG:
            raise NameError("Buffer" + name + " has been registered")
        BUF_REG[name] = buf
        return buf
    return _wrapper


class Buffer:
    name = None

    def __init__(self, *args):
        raise NotImplementedError(
            "Derived " + self.name + " buffer not override the" + \
            " base `__init__` function defined in Buffer")

    def get(self):
        raise NotImplementedError(
            "Derived " + self.name + " buffer not override the" + \
            " base `get` function defined in Buffer")

    def serialize(self):
        raise NotImplementedError(
            "Derived " + self.name + " buffer not override the" + \
            " base `serialize` function defined in Buffer")


@register_buffer("Scale")
class SBuffer(Buffer):
    """ Scale Buffer
    """
    def __init__(self, *args):
        assert len(args) == 1
        self.scale = args[0]

    def get(self):
        return self.scale

    def serialize(self):
        return [self.name, self.scale]


@register_buffer("ScaleList")
class SLBuffer(Buffer):
    """ Scale List Buffer
    """
    def __init__(self, *args):
        self.scale_list = args

    def get(self):
        return self.scale_list

    def serialize(self):
        return [self.name, self.scale_list]


@register_buffer("ScaleZpoint")
class SZBuffer(Buffer):
    """ Scale and Zero Point Buffer
    """
    def __init__(self, *args):
        assert len(args) == 2
        self.scale, self.zpoint = args[0], args[1]

    def get(self):
        return self.scale, self.zpoint

    def serialize(self):
        return [self.name, self.scale, self.zpoint]

def get_buffer(buf_type, *args):
    if buf_type not in BUF_REG:
        raise TypeError("Unknown buffer type: %20s", buf_type)
    return BUF_REG[buf_type](*args)

#----------------------------
# Quantizer Types Definition
#----------------------------

QUANT_REG = {
    # "UniformSymmetric": USQuantizer,
    # "UniformSymmetricChannelwise": USCQuantizer,
    # "UniformAffine": UAQuantizer,
}

def register_quantizer(name):
    def _wrapper(quantizer):
        quantizer.name = name
        if name in QUANT_REG:
            raise NameError("Quantizer" + name + " has been registered")
        QUANT_REG[name] = quantizer
        return quantizer
    return _wrapper


class Quantizer:
    name = None

    def sample(self, data, **kwargs):
        raise NotImplementedError(
            "Derived " + self.name + " quantizer not override the" + \
            " base `sample` function defined in Quantizer")

    def get_range(self, prec):
        raise NotImplementedError(
            "Derived " + self.name + " quantizer not override the" + \
            " base `get_range` function defined in Quantizer")

    def get_prec(self, val):
        raise NotImplementedError(
            "Derived " + self.name + " quantizer not override the" + \
            " base `get_prec` function defined in Quantizer")

    def quantize(self, sym, oprec, oscale=None, **kwargs):
        if sutils.is_params(sym, kwargs["params"]):
            return self._quantize_parameter(sym, oprec, oscale=oscale, **kwargs)
        return self._quantize_operator(sym, oprec, oscale=oscale, **kwargs)

    def _quantize_parameter(self, W, oprec, oscale=None, **kwargs):
        raise NotImplementedError(
            "Derived " + self.name + " quantizer not override the" + \
            " base `_quantize_parameter` function " + \
            "defined in Quantizer")

    def _quantize_operator(self, X, oprec, oscale=None, **kwargs):
        raise NotImplementedError(
            "Derived " + self.name + " quantizer not override the" + \
            " base `_quantize_opoerator` function " + \
            "defined in Quantizer")

    def int_realize(self, data, prec, **kwargs):
        raise NotImplementedError(
            "Derived " + self.name + " quantizer not override the" + \
            " base `int_realize` function " + \
            "defined in Quantizer")

US_QUANT_TYPE = "UniformSymmetric"


@register_quantizer(US_QUANT_TYPE)
class USQuantizer(Quantizer):
    """ Information data type for uniform symmetric quantizaton
    """
    def sample(self, data, **kwargs):
        absmax = float(data.abs().max().asscalar())
        return AFeature(absmax)

    def get_range(self, prec):
        mrange = 2**(prec-1) - 1
        return -mrange, mrange

    def _get_buffer(self, oprec, absmax):
        return self.get_range(oprec)[1] / absmax

    def get_prec(self, val):
        return tutils.get_bit(val)

    def _quantize_parameter(self, W, oprec, oscale=None, **kwargs):
        """ Symmetric Quantization of weight (real value)
        """
        logger = logging.getLogger("log.mrt.realize")
        params, features = kwargs["params"], kwargs["features"]
        precs = kwargs['precs']
        wn = W.attr("name")
        wqn = N.n(wn)

        oprec = precs[wn].get(kwargs['oname'], oprec)
        ft = features[wn]
        absmax = ft.get()

        if absmax == 0:
            oprec, oscale = 1, 1 if oscale is None else oscale
            params[wqn] = sutils.nd_zeros(params[wn].shape)
        else:
            oscale = self._get_buffer(oprec, absmax) \
                if oscale is None else oscale
            params[wqn], oprec = self.int_realize(
                params[wn]*oscale, oprec, logger=logger)
        attr = {"precision": str(oprec)}
        # TODO: CVM precision update
        # attr = {"precision": "int"+str(oprec)}
        W = mx.sym.var(wqn, shape=params[wqn].shape, attr=attr)
        return W, oprec, oscale

    def _quantize_operator(self, X, oprec, oscale=None, **kwargs):
        """ Symmetric Quantization of symbol expansion (int value)
        """
        logger = kwargs.get("logger", logging.getLogger("log.mrt.realize"))
        params, features = kwargs["params"], kwargs["features"]
        precs, buffers = kwargs["precs"], kwargs["buffers"]
        graph, shift_bits = kwargs["graph"], kwargs["shift_bits"]
        xn, xopn = X.attr("name"), X.attr("op_name")
        xqn = N.n(xn)

        oprec = precs[xn].get(kwargs['oname'], oprec)
        iscale, iprec = buffers[xn].get(), precs[xn][OUT_KEY]
        ft = features[xn]
        absmax = ft.get()
        if absmax == 0:
            return X, 1, 1 if oscale is None else oscale
        oscale = self._get_buffer(oprec, absmax) \
            if oscale is None else oscale

        sb = iprec - oprec
        if sb > shift_bits:
            iprec -= sb
            X = tutils.realize(X, sb, iprec)
            iscale = iscale / (2**sb)

        if oscale is not None or iprec > oprec:
            rescale = oscale / iscale
            bits = MAX_BIT - iprec
            frac, exp = sim.cvm_float(rescale, bits)
            sim_scale = frac * (2**exp)
            scale_err = abs((sim_scale - rescale) / rescale)
            if scale_err > 0.001:
                logger.warn(
                    "Operator  %-20s name=%-40s quantize with sb=%s" +
                    " scale=%s, error=%s",
                    xopn, xn, sb, iscale, scale_err)
            oscale = iscale * frac * (2**exp)
            if frac > 1:
                var = sutils.nd_const(frac, graph, params)
                X = mx.sym.broadcast_mul(
                    X, var, name=N.n("mrt_quantize_scale"))
            oprec = self.get_prec(oscale*absmax)
            X = tutils.realize(X, -exp, oprec)
            logger.debug(
                "Operator  %-20s name=%-40s requantize" +
                " with scale=%-16.8f<%d, %d>" +
                " iprec=%s, iscale=%-10.5f, oprec=%s, oscale=%-10.5f",
                xopn, xn, rescale, frac, exp, iprec, iscale, oprec, oscale)
        else:
            oprec, oscale = iprec, iscale
            logger.debug(
                "Operator  %-20s name=%-40s clip with iprec=%s, oprec=%s",
                xopn, xn, iprec, oprec)

        return X, oprec, oscale

    def int_realize(self, data, prec, **kwargs):
        logger = kwargs.get("logger", logging)

        out = data.round()
        lower, upper = self.get_range(prec)
        if out.abs().max() > upper:
            logger.warn(
                "quant out of range int%d with data=<%s,%s>",
                prec, out.max().asnumpy(), out.min().asnumpy())
        out = out.clip(a_min=lower, a_max=upper)
        return out, self.get_prec(out)


@register_quantizer("UniformSymmetricChannel")
class USCQuantizer(USQuantizer):
    """ Information data type for uniform symmetric channel-wise quantizaton
    """
    def sample(self, data, **kwargs):
        ichannel = kwargs['ichannel']
        step = kwargs.get('step', 1)
        C = data.shape[ichannel]
        absmax = data.abs().max(axis=ichannel, keepdims=False).asnumpy().tolist()
        absmax_list = [float(max(absmax[i:i+step])) for i in range(0, C, step)]
        return ALFeature(absmax_list)

    def get_range(self, prec):
        mrange = 2**(prec-1) - 1
        return -mrange, mrange

    def get_prec(self, val):
        return tutils.get_bit(val)

    def _quantize_parameter(self, W, oprec, oscale=None, **kwargs):
        """ Symmetric Quantization of weight (real value)
        """
        logger = logging.getLogger("log.mrt.realize")
        params, features = kwargs["params"], kwargs["features"]
        precs = kwargs['precs']
        wn = W.attr("name")
        wqn = N.n(wn)

        oprec = precs[wn].get(kwargs['oname'], oprec)
        ft = features[wn]
        absmax_list = ft.get()
        oscale_list, oprec_list = [], []

        for absmax in absmax_list:
            if absmax == 0:
                oscale_list.append(0 if oscale is None else oscale)
                oprec_list.append(1)
            else:
                oscale_list.append(
                    self.get_buffer(oprec, absmax) if oscale is None else oscale)
                oprec_list.append(oprec)

        ichannel = kwargs['ichannel']
        os_tensor = nd.array(oscale_list)
        shp = params[wqn].shape
        for i in range(len(shp)):
            if i != ichannel:
                os_tensor = nd.expand_dims(oscale_tensor, axis=i)
        params[wqn] = nd.broadcast_mul(params[wn], os_tensor)
        oscale_list = [1 if oscale == 0 else oscale for oscale in oscale_list]
        attr = {"precision": str(oprec_list)}
        W = mx.sym.var(wqn, shape=params[wqn].shape, attr=attr)

        return W, oprec_list, oscale_list

    def _quantize_operator(self, X, oprec, oscale=None, **kwargs):
        """ Symmetric Quantization of symbol expansion (int value)
        """
        logger = kwargs.get("logger", logging.getLogger("log.mrt.realize"))
        params, features = kwargs["params"], kwargs["features"]
        precs, buffers = kwargs["precs"], kwargs["buffers"]
        graph, shift_bits = kwargs["graph"], kwargs["shift_bits"]
        xn, xopn = X.attr("name"), X.attr("op_name")
        xqn = N.n(xn)

        oprec = precs[xn].get(kwargs['oname'], oprec)
        iscale, iprec = buffers[xn].get(), precs[xn][OUT_KEY]
        ft = features[xn]
        absmax = ft.get()
        if absmax == 0:
            return X, 1, 1 if oscale is None else oscale
        oscale = self._get_buffer(oprec, ft).get() \
            if oscale is None else oscale

        sb = iprec - oprec
        if sb > shift_bits:
            iprec -= sb
            X = tutils.realize(X, sb, iprec)
            iscale = iscale / (2**sb)

        if oscale is not None or iprec > oprec:
            rescale = oscale / iscale
            bits = MAX_BIT - iprec
            frac, exp = sim.cvm_float(rescale, bits)
            # TODO: compare n times
            sim_scale = frac * (2**exp)
            scale_err = abs((sim_scale - rescale) / rescale)
            if scale_err > 0.001:
                logger.warn(
                    "Operator  %-20s name=%-40s quantize with sb=%s" +
                    " scale=%s, error=%s",
                    xopn, xn, sb, iscale, scale_err)
            oscale = iscale * frac * (2**exp)
            if frac > 1:
                var = sutils.nd_const(frac, graph, params)
                X = mx.sym.broadcast_mul(
                    X, var, name=N.n("mrt_quantize_scale"))
            oprec = self.get_prec(oscale*absmax)
            X = tutils.realize(X, -exp, oprec)
            logger.debug(
                "Operator  %-20s name=%-40s requantize" +
                " with scale=%-16.8f<%d, %d>" +
                " iprec=%s, iscale=%-10.5f, oprec=%s, oscale=%-10.5f",
                xopn, xn, rescale, frac, exp, iprec, iscale, oprec, oscale)
        else:
            oprec, oscale = iprec, iscale
            logger.debug(
                "Operator  %-20s name=%-40s clip with iprec=%s, oprec=%s",
                xopn, xn, iprec, oprec)

        return X, oprec, oscale

    def int_realize(self, data, prec, **kwargs):
        logger = kwargs.get("logger", logging)

        out = data.round()
        lower, upper = self.get_range(prec)
        if out.abs().max() > upper:
            logger.warn(
                "quant out of range int%d with data=<%s,%s>",
                prec, out.max().asnumpy(), out.min().asnumpy())
        out = out.clip(a_min=lower, a_max=upper)
        return out, self.get_prec(out)


@register_quantizer("UniformAffine")
class UAQuantizer(Quantizer):
    """ Information data type for uniform affine quantizaton
    """
    def sample(self, data):
        minv = float(data.min().asscalar())
        maxv = float(data.max().asscalar())
        return MMFeature(minv, maxv)

    def get_range(self, prec):
        mrange = 2**prec - 1
        return 0, mrange

    def get_prec(self, data):
        if isinstance(data, nd.NDArray):
            data = data.max().asscalar()
        assert data > 0
        return math.ceil(math.log2(math.data+1))

    def _quantize_parameter(self, W, oprec, oscale=None, **kwargs):
        logger = logging.getLogger("log.mrt.realize")
        params, features = kwargs["params"], kwargs["features"]
        precs = kwargs['precs']
        wn = W.attr("name")
        wqn = N.n(wn)

        oprec = precs[wn].get(kwargs['oname'], oprec)
        minv, maxv = features[wn].get()
        oscale = (2**(oprec)-1) / (maxv-minv) if oscale is None else oscale
        zpoint = round(minv)
        Zp = sutils.nd_const(zpoint, graph, params)
        params[wqn], oprec = self.int_realize(
            (params[wn] - zpoint)*oscale, oprec, logger=logger)
        attr = {"precision": str(oprec)}
        # TODO: CVM precision update
        # attr = {"precision": "uint"+str(oprec)}
        W = mx.sym.var(wqn, shape=params[wqn].shape, attr=attr)
        return W, oprec, oscale, Zp

    def _quantize_operator(self, X, oprec, oscale=None, **kwargs):
        logger = kwargs.get("logger", logging.getLogger("log.mrt.realize"))
        params, features = kwargs["params"], kwargs["features"]
        precs, buffers = kwargs["precs"], kwargs["buffers"]
        graph, shift_bits = kwargs["graph"], kwargs["shift_bits"]
        xn, xopn = X.attr("name"), X.attr("op_name")
        xqn = N.n(xn)

        oprec = precs[xn].get(kwargs['oname'], oprec)
        iscale, iprec = buffers[xn].get(), precs[xn][OUT_KEY]
        minv, maxv = features[wn].get()
        oscale = (2**(oprec)-1) / (maxv-minv) if oscale is None else oscale
        zpoint = round(minv*iscale)

        sb = iprec - oprec
        if sb > shift_bits:
            iprec -= sb
            X = tutils.realize(X, sb, iprec)
            iscale = iscale / (2**sb)

        rescale = oscale / iscale
        bits = MAX_BIT - iprec
        frac, exp = sim.cvm_float(rescale, bits)
        sim_scale = frac * (2**exp)
        scale_err = abs((sim_scale - rescale) / rescale)
        if scale_err > 0.001:
            logger.warn(
                "Operator  %-20s name=%-40s quantize with sb=%s" +
                " scale=%s, error=%s",
                xopn, xn, sb, iscale, scale_err)
        oscale = iscale * frac * (2**exp)
        if frac > 1:
            var = sutils.nd_const(frac, graph, params)
            X = mx.sym.broadcast_mul(
                X, var, name=N.n("mrt_quantize_scale"))
        Zp = sutils.nd_const(zpoint, graph, params)
        X = mx.sym.broadcast_sub(X, Zp, name=N.n('minus_zp'))
        X = tutils.realize(
            X, -exp, USQuantizer().get_prec(oscale*(maxv-minv)))
        oprec = self.get_prec(oscale*(maxv-minv)-zpoint)
        logger.debug(
            "Operator  %-20s name=%-40s requantize" +
            " with scale=%-16.8f<%d, %d>" +
            " iprec=%s, iscale=%-10.5f, oprec=%s, oscale=%-10.5f",
            xopn, xn, rescale, frac, exp, iprec, iscale, oprec, oscale)

        return X, oprec, oscale, Zp

    def int_realize(self, data, prec, **kwargs):
        logger = kwargs.get("logger", logging)

        out = data.round()
        lower, upper = self.ger_range(prec)
        if out.max() > upper and out.min() < lower:
            logger.warn(
                "quant out of range int%d with data=<%s,%s>",
                prec, out.max().asnumpy(), out.min().asnumpy())
        out = out.clip(a_min=lower, a_max=upper)
        return out, self.get_prec(out)

DEFAULT_QUANT_TYPE = US_QUANT_TYPE
DEFAULT_QUANTIZER = USQuantizer()

QUANT_INSTANCES = {
    DEFAULT_QUANT_TYPE: DEFAULT_QUANTIZER,
    # "uniform_affine": UAQuantizer(),
}

def get_quantizer(quant_type):
    if quant_type not in QUANT_INSTANCES:
        QUANT_INSTANCES[quant_type] = QUANT_REG[quant_type]()
    return QUANT_INSTANCES[quant_type]

#----------------------------
# Optimizor Registration
#----------------------------

OPT_REG = {
    # "HistoricalValue": HVOptimizor,
    # "MovingAverage": MAOptimizor,
    # "KLDivergence": KLDOptimizor,
    # "OutlierRemoval": OROptimizor,
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

    def get_opt(self, raw_ft, out, **kwargs):
        raise NotImplementedError(
            "Derived " + self.name + " optimizor not override the" + \
            " base `get_opt` function defined in Optimizor")

    @staticmethod
    def list_supported_quant_types():
        raise NotImplementedError(
            "Derived " + self.name + " optimizor not override the" + \
            " base `list_supported_quant_types` function defined in Optimizor")

    @staticmethod
    def list_attr_types():
        return {}


@register_optimizor("HistoricalValue")
class HVOptimizor(Optimizor):
    """ Generalized historical value optimizor
    """
    lambd = None # hyperparameter for fine tuning

    def __init__(self, **attrs):
        super().__init__(**attrs)

    def get_opt(self, raw_ft, out, **kwargs):
        logger = kwargs.get("logger", logging.getLogger("optimize"))
        hist_ft = kwargs.get("hist_ft", None)
        name = kwargs.get("name", _NULL_NAME)

        if isinstance(raw_ft, AFeature):
            # hyperparameter 'lambd' for fine tuning
            absmax = raw_ft.get()
            if self.lambd is not None:
                mean = nd.mean(out).asscalar()
                sqrt_n = math.sqrt(np.product(out.shape))
                std = nd.norm(out-mean).asscalar() / sqrt_n
                alpha = abs(mean) + self.lambd*std
                absmax = alpha if alpha < 0.95*absmax else absmax
            if hist_ft is None:
                p = logger.debug if absmax < 30 else logger.warn
                p("collect symbol %-40s, out_shape=%-20s, opt: (%s)",
                  name, out.shape, absmax)
                opt = AFeature(absmax)
            else:
                opt = AFeature(max(habsmax, hist_ft.get()))
        elif isinstance(raw_ft, MMFeature):
            minv, maxv = raw_ft.get()
            if hist_ft is None:
                opt = MMFeature(minv, maxv)
            else:
                hminv, hmaxv = hist_ft.get()
                opt = MMFeature(min(minv, hminv), max(maxv, hmaxv))
        else:
            raise TypeError(
                "Unsupported feature type: %s for HVOptimizor" % \
                type(raw_ft))
        return opt

    @staticmethod
    def list_supported_quant_types():
        return ["UniformSymmetric", "UniformAffine"]

    @staticmethod
    def list_attr_types():
        return {"lambd": [_NONETYPE, float]}


@register_optimizor("MovingAverage")
class MAOptimizor(Optimizor):
    """ Generalized moving average optimizor
    """
    c = 0.01 # hyperparameter for moving average

    def __init__(self, **attrs):
        super().__init__(**attrs)

    def get_opt(self, raw_ft, out, **kwargs):
        hist_ft = kwargs.get("hist_ft", None)
        if isinstance(raw_ft, AFeature):
            absmax = raw_ft.get()
            if hist_ft is None:
                opt = AFeature(absmax)
            else:
                habsmax = hist_ft.get()
                opt = AFeature((1-self.c)*habsmax + self.c*absmax)
        elif isinstance(raw_ft, MMFeature):
            minv, maxv = raw_ft.get()
            if hist_val is None:
                opt = MMFeature(minv, maxv)
            else:
                hminv, hmaxv = hist_ft.get()
                opt = MMFeature(
                    (1-self.c)*hminv + self.c*minv,
                    (1-self.c)*hmaxv + self.c*maxv)
        else:
            raise TypeError(
                "Unsupported feature type: %s for MAOptimizor" % \
                type(raw_ft))
        return opt

    @staticmethod
    def list_supported_quant_types():
        return ["UniformSymmetric", "UniformAffine"]

    @staticmethod
    def list_attr_types():
        return {"c": [float]}


@register_optimizor("KLDivergence")
class KLDOptimizor(Optimizor):
    """ KL divergence optimizor for AFeature
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

    def get_opt(self, raw_ft, out, **kwargs):
        hist_ft = kwargs.get("hist_ft", None)
        if not isinstance(raw_ft, AFeature):
            raise TypeError(
                "KLDOptimizor do not support feature type: %s, " + \
                "only AFeature is supported" % type(raw_ft))

        absmax = raw_ft.get()
        kval = self._kldiverge(absmax, out)
        opt = AFeature()
        if hist_val is None:
            opt = AFeature(kval)
        else:
            opt = AFeature(max(kval, hist.ft.get()))
        return opt

    @staticmethod
    def list_supported_quant_types():
        return ["UniformSymmetric"]

    @staticmethod
    def list_attr_types():
        return {"bucket_bit": [int], "quant_bit": [int], "eps": [float]}


@register_optimizor("OutlierRemoval")
class OROptimizor(Optimizor):
    pass

DEFAULT_OPT_INFO = {
    "opt_type": "HistoricalValue",
    "lambd": None,
}
DEFAULT_OPTIMIZOR = HVOptimizor()

def make_key_opt(opt_info):
    infos = [opt_info["opt_type"]]
    for k, v in opt_info.items():
        if k == "opt_type":
            continue
        infos.append(k)
        infos.append(v)
    return tuple(infos)

OPT_INSTANCES = {
    make_key_opt(DEFAULT_OPT_INFO): DEFAULT_OPTIMIZOR,
    # ("HistoricalValue", "lambd", 25): HVOptimizor(lambd=25),
    # ("MovingAverage","c", 0.01): MAOptimizor(),
    # ("KLDivergence", "eps", 0.05): KLDOptimizor(eps=0.05),
}

def get_optimizor(opt_info):
    opt_type = opt_info[0]
    if opt_info not in OPT_INSTANCES:
        opt_attrs = {} if len(opt_info) == 1 else \
            {v[i]: v[i+1] for i in range(1, len(opt_info), 2)}
        OPT_INSTANCES[opt_info] = OPT_REG[opt_type](**opt_attrs)
    return OPT_INSTANCES[opt_info]


#----------------------------
# Expand Types Definition
#----------------------------

FT_TYPE_EXP = AFeature.name
BUF_TYPE_EXP = SBuffer.name
QUANT_TYPE_EXP = USQuantizer.name

#----------------------------
# Granularity Types
#----------------------------

LAYER_WISE_TYPE = "layer-wise"
CHANNEL_WISE_TYPE = "channel-wise"
DEFAULT_GN_INFO = {"gn_type": LAYER_WISE_TYPE}

GN_REG = {
    LAYER_WISE_TYPE,
    CHANNEL_WISE_TYPE,
}

