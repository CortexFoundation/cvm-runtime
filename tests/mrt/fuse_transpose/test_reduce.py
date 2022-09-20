import unittest

import mxnet as mx
from mxnet import ndarray as nd
import numpy as np

from mrt import tfm_ops
from mrt.tfm_pass import fuse_transpose, attach_input_shape
from mrt.sym_utils import get_mxnet_op, sym_iter

ctx = mx.gpu(0)


class TestFuseTransposeReduce(unittest.TestCase):
    def assert_equal(self, a, b, places=10):
        self.assertEqual(a.shape, b.shape)
        nentry = int(nd.prod(nd.array(a.shape)).asscalar())
        res = (a-b).reshape(shape=(nentry,)).asnumpy()
        norm = np.linalg.norm(res)
        self.assertAlmostEqual(norm, 0.0, places=places)

    def cmp_sym(self, sym_1, sym_2):
        op_name_1 = sym_1.attr("op_name")
        op_name_2 = sym_2.attr("op_name")
        if op_name_1 != op_name_2:
            return False
        attr_1 = sym_1.list_attr()
        attr_2 = sym_2.list_attr()
        if attr_1 != attr_2:
            return False
        childs_1 = sym_iter(sym_1.get_children())
        childs_2 = sym_iter(sym_2.get_children())
        if childs_1 is None and childs_2 is not None or \
            childs_1 is not None and childs_2 is None:
            return False
        if childs_1 is None and childs_2 is None:
            return True
        if len(childs_1) != len(childs_2):
            return False
        for i in range(len(childs_1)):
            if not self.cmp_sym(childs_1[i], childs_2[i]):
                return False
        return True

    def get_random_data(self, shp, low=-1000.0, high=1000.0):
        data = nd.random.uniform(low=low, high=high, shape=shp)
        data = data.as_in_context(ctx)
        return data

    def test_fuse_transpose_reduce_keepdims(self):
        configs = [
            {
                "shp_cx": (2,3,4,2,5),
                "axes_1": [4,3,1,0,2],
                "axis": [0,3,4],
                "op_name": "sum",
                "axes_2": [2,4,0,1,3],
                "sym_ref": \
                    mx.sym.transpose(
                        mx.sym.sum(
                            mx.sym.var("cx", shape=(2,3,4,2,5)),
                            axis=[0,2,4], keepdims=True),
                        axes=[1,2,4,3,0]),
            },
            {
                "shp_cx": (4,2,4,2,5,7),
                "axes_1": [4,3,1,5,0,2],
                "axis": [0,2,5],
                "op_name": "mean",
                "axes_2": [2,5,4,0,1,3],
                "sym_ref": \
                    mx.sym.transpose(
                        mx.sym.mean(
                            mx.sym.var("cx", shape=(4,2,4,2,5,7)),
                            axis=[1,2,4], keepdims=True),
                        axes=[1,2,0,4,3,5]),
            },
        ]
        for config in configs:
            # generate data
            shp_cx = config["shp_cx"]
            data = self.get_random_data(shp_cx)
            # original graph
            cx = mx.sym.var("cx")
            axes_1 = config["axes_1"]
            tp1 = mx.sym.transpose(cx, axes=axes_1, name="var_tp1")
            axis = config["axis"]
            op_name = config["op_name"]
            op = get_mxnet_op(op_name)(
                tp1, axis=axis, name="var_"+op_name, keepdims=True)
            axes_2 = config["axes_2"]
            tp2 = mx.sym.transpose(op, axes=axes_2, name="var_tp2")
            # original output
            ex_1 = tp2.bind(ctx, {'cx': data})
            out_1 = ex_1.forward()
            # fused graph
            tp2n, _ = attach_input_shape(tp2, {}, {"cx":data.shape})
            tp2n, _ = fuse_transpose(tp2n, {})
            # fused output
            ex_2 = tp2n.bind(ctx, {'cx': data})
            out_2 = ex_2.forward()
            assert len(out_1) == len(out_2) == 1
            self.assert_equal(out_1[0], out_2[0], places=4)
            sym_ref = config["sym_ref"]
            self.assertEqual(self.cmp_sym(tp2n,sym_ref), True)

    def test_fuse_transpose_reduce(self):
        configs = [
            {
                "shp_cx": (2,3,4,2,5),
                "axes": [4,3,1,0,2],
                "axis": [0,3,4],
                "op_name": "sum",
                "sym_ref": \
                    mx.sym.transpose(
                        mx.sym.sum(
                            mx.sym.var("cx", shape=(2,3,4,2,5)),
                            axis=[0,2,4], keepdims=False),
                        axes=[1,0]),
            },
            {
                "shp_cx": (4,2,4,2,5,7),
                "axes": [4,0,1,3,5,2],
                "axis": [0,2,5],
                "op_name": "mean",
                "sym_ref": \
                    mx.sym.mean(
                        mx.sym.var("cx", shape=(4,2,4,2,5,7)),
                        axis=[1,2,4], keepdims=False),
            },
        ]
        for config in configs:
            # generate data
            shp_cx = config["shp_cx"]
            data = self.get_random_data(shp_cx)
            # original graph
            cx = mx.sym.var("cx")
            axes = config["axes"]
            tp = mx.sym.transpose(cx, axes=axes, name="var_tp")
            axis = config["axis"]
            op_name = config["op_name"]
            op = get_mxnet_op(op_name)(
                tp, axis=axis, name="var_"+op_name, keepdims=False)
            # original output
            ex_1 = op.bind(ctx, {'cx': data})
            out_1 = ex_1.forward()
            # fused graph
            op, _ = attach_input_shape(op, {}, {"cx":data.shape})
            opn, _ = fuse_transpose(op, {})
            # fused output
            ex_2 = opn.bind(ctx, {'cx': data})
            out_2 = ex_2.forward()
            assert len(out_1) == len(out_2) == 1
            self.assert_equal(out_1[0], out_2[0], places=4)
            sym_ref = config["sym_ref"]
            self.assertEqual(self.cmp_sym(opn,sym_ref), True)

if __name__ == "__main__":
    unittest.main()
