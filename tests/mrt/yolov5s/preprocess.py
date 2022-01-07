import argparse
import json
from os import path
import logging
from copy import deepcopy

import mxnet as mx
from mxnet import ndarray as nd

from mrt.sym_utils import topo_visit_transformer, sym_iter
from mrt import utils
from mrt.tfm_base import N
from mrt.conf import MRT_MODEL_ROOT
from mrt.V3.utils import load_fname

utils.log_init()
parser = argparse.ArgumentParser()
parser.add_argument("--model-name", type=str, default="yolov5s")
parser.add_argument(
    "--model-dir", type=str, default=MRT_MODEL_ROOT)
parser.add_argument("--no-unify", action="store_true")
parser.add_argument("--no-broadcastify", action="store_true")

@N.register_nm("unify")
def unify(sym, params, logger=logging.getLogger("unify")):

    # check symbol
    sym_json_str = sym.tojson()
    sym_json_dict = json.loads(sym_json_str)
    nodes = sym_json_dict["nodes"]

    name_cnts = {}
    nnodes = []
    for node in nodes:
        name = node["name"]
        if name in name_cnts:
            cur_cnt = name_cnts[name] = N.n(name)
            logger.info("duplicate name: {}".format(name))
            nnode = deepcopy(node)
            nnode["name"] = "{}_{}".format(name, cur_cnt)
            nnodes.append(nnode)
        else:
            name_cnts[name] = 1
            nnodes.append(node)

    sym_json_dict["nodes"] = nnodes
    sym_json_str = json.dumps(sym_json_dict)
    sym = mx.sym.load_json(sym_json_str)

    # check params
    # model exported from mxnet hybrid block compatibility,
    # remove the unnecessary prefix, hack
    param_keys = {}
    for k in params:
        if k.startswith("arg:") or k.startswith("aux:"):
            nk = k[4:]
        else:
            nk = k
        if nk in param_keys:
            assert False, nk
        param_keys[k] = nk
    params = {param_keys[k]: v for k, v in params.items()}

    return sym, params

@N.register_nm("broadcastify")
def broadcastify(sym, params, logger=logging.getLogger("broadcastify")):
    def callback(op, **kwargs):
        name, op_name = op.attr("name"), op.attr("op_name")
        if op_name != "elemwise_mul":
            return op
        childs = sym_iter(op.get_children())
        lhs, rhs = childs
        op = mx.sym.broadcast_mul(lhs, rhs, name=name)
        logger.info("op: {} has been broadcastified".format(name))
        return op

    return topo_visit_transformer(
        sym, params, callback, logger=logger)

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
    if not args.no_unify:
        suffixes.append("unify")
        sym, params = unify(sym, params)
    if not args.no_broadcastify:
        suffixes.append("broadcastify")
        sym, params = broadcastify(sym, params)
    suffix = ".".join(suffixes)

    sym_json_str = sym.tojson()
    nsym_file, nprm_file = load_fname(prefix, suffix=suffix)
    with open(nsym_file, "w") as f:
        f.write(sym_json_str)
    nd.save(nprm_file, params)
