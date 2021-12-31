import argparse
from os import path
import logging

import mxnet as mx
from mxnet import ndarray as nd

from mrt.sym_utils import (
    topo_visit_transformer, sym_iter, get_entry_id, get_attr)
from mrt import utils
from mrt.tfm_base import N
from mrt.conf import MRT_MODEL_ROOT
from mrt.V3.utils import load_fname
from mrt import tfm_pass as tpass

utils.log_init()
parser = argparse.ArgumentParser()
parser.add_argument("--model-name", type=str, default="prediction_SCTF")
parser.add_argument(
    "--model-dir", type=str, default=MRT_MODEL_ROOT)
parser.add_argument("--no-reduce-dense", action="store_true")

@N.register_nm("reduce_dense")
def reduce_dense(sym, params, logger=logging.getLogger("reduce_dense")):
    def callback(op, **kwargs):
        name, op_name = op.attr('name'), op.attr('op_name')
        if op_name != 'FullyConnected':
            return op

        attr, childs = op.list_attr(), sym_iter(op.get_children())
        cns = [c.attr('name') for c in childs]
        X, W = childs[:2]
        infer_shapes = kwargs['infer_shapes']
        xshp = infer_shapes[cns[0]][get_entry_id(X)]
        if len(xshp) == 2:
            return op

        no_bias = get_attr(attr, 'no_bias')
        flatten = get_attr(attr, "flatten")
        num_hidden = get_attr(attr, "num_hidden")

        if flatten:
            shape = (-1,) + xshp[1:]
            rshp = mx.sym.reshape(X, shape=shape)
            if no_bias:
                op = mx.sym.FullyConnected(
                    rshp, W, no_bias=no_bias, flatten=flatten,
                    num_hidden=num_hidden)
            else:
                op = mx.sym.FullyConnected(
                    rshp, W, childs[2], no_bias=no_bias, flatten=flatten,
                    num_hidden=num_hidden)
        else:
            default_batch_axis = 0
            batch_axis = \
                kwargs.get("batch_axes", {}).get(name, default_batch_axis)
            assert batch_axis < len(xshp), \
                "invalid batch_axis: {}, length of xshp: {}".format(
                batch_axis, len(xshp))
            if batch_axis == len(xshp)-1:
                product = int(nd.prod(nd.array(xshp)).asscalar())
                res_shp = int(product/xshp[batch_axis])
                shape = (res_shp, -1)
            else:
                shape = (-1, xshp[-1])
            rshp = mx.sym.reshape(X, shape=shape)
            if no_bias:
                fc = mx.sym.FullyConnected(
                    rshp, W, no_bias=no_bias, flatten=flatten,
                    num_hidden=num_hidden)
            else:
                fc = mx.sym.FullyConnected(
                    rshp, W, childs[2], no_bias=no_bias, flatten=flatten,
                    num_hidden=num_hidden)
            if batch_axis == len(xshp)-1:
                shape = xshp[:-1] + (num_hidden,)
            else:
                shape = \
                    xshp[:batch_axis] + (-1,) + \
                    xshp[batch_axis+1:-1] + (num_hidden,)
            op = mx.sym.reshape(fc, shape=shape)

        logger.info(
            "{}-d dense name: {} has been reduced.".format(
            len(xshp), name))
        return op

    infer_shapes = tpass.infer_shape(sym, params, input_shape=(64,1,3))
    return topo_visit_transformer(
        sym, params, callback, logger=logger, infer_shapes=infer_shapes)

if __name__ == "__main__":
    args = parser.parse_args()

    model_name = args.model_name
    model_dir = args.model_dir
    if model_dir.startswith("~"):
        model_dir = path.expanduser(model_dir)
    prefix = path.join(model_dir, model_name)
    sym_file, prm_file = load_fname(prefix)
    sym = mx.sym.load(sym_file)
    params = nd.load(prm_file)

    suffixes = ["preprocess"]
    if not args.no_reduce_dense:
        suffixes.append("reduce_dense")
        sym, params = reduce_dense(sym, params)
    suffix = ".".join(suffixes)

    sym_json_str = sym.tojson()
    nsym_file, nprm_file = load_fname(prefix, suffix=suffix)
    with open(nsym_file, "w") as f:
        f.write(sym_json_str)
    nd.save(nprm_file, params)
