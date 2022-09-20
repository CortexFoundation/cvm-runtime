import unittest

import mxnet as mx
from mxnet import ndarray as nd
import numpy as np

from mrt import tfm_ops, cvm_op
from mrt.tfm_pass import rewrite, attach_input_shape
from mrt.sym_utils import get_mxnet_op, sym_iter

ctx = mx.gpu(0)


class TestRewriteMean(unittest.TestCase):
    def cal_err_rel(self, a, b):
        self.assertEqual(a.shape, b.shape)
        nentry = int(nd.prod(nd.array(a.shape)).asscalar())
        numerator = np.linalg.norm((a-b).reshape(shape=(nentry,)).asnumpy())
        denominator = max(
            np.linalg.norm(a.reshape(shape=(nentry,)).asnumpy()),
            np.linalg.norm(a.reshape(shape=(nentry,)).asnumpy())
        )
        err = numerator / denominator
        return err

    def assert_equal_rel_places(self, a, b, min_places=1, max_places=20):
        err = self.cal_err_rel(a, b)
        self.assertAlmostEqual(err, 0.0, places=min_places)
        # search
        places = min_places
        while True:
            places <<= 1
            try:
                self.assertAlmostEqual(err, 0.0, places=places)
            except AssertionError:
                break
            if places > max_places:
                return max_places
        l, r = min_places, places
        while l < r-1:
            m = (l+r) >> 1
            flag = True
            try:
                self.assertAlmostEqual(err, 0.0, places=m)
            except AssertionError:
                flag = False
            if flag:
                l = m
            else:
                r = m
        return l

    def get_random_data_round(self, shp, low=-10000.0, high=10000.0):
        data = nd.random.uniform(
            low=low, high=high, shape=shp).round().astype('int')
        data = data.as_in_context(ctx)
        return data

    def test_rewrite(self):
        configs = [
            {
                "shp_cx": (10,4,512,5,2),
                "low": -1000.0,
                "high": 1000.0,
                "axis": (1,2,4),
                "keepdims": False,
            },
            {
                "shp_cx": (10,4,512,5,2),
                "low": -10000.0,
                "high": 10000.0,
                "axis": (1,2,4),
                "keepdims": False,
            },
            {
                "shp_cx": (10,4,512,5,2),
                "low": -100000.0,
                "high": 100000.0,
                "axis": (1,2,4),
                "keepdims": False,
            },
            {
                "shp_cx": (10,4,512,5,2),
                "low": -1000000.0,
                "high": 1000000.0,
                "axis": (1,2,4),
                "keepdims": False,
            },
        ]
        for config in configs:
            # generate data
            shp_cx = config["shp_cx"]
            low = config["low"]
            high = config["high"]
            data = self.get_random_data_round(shp_cx, low=low, high=high)
            # original graph
            cx = mx.sym.var("cx")
            axis = config["axis"]
            keepdims = config["keepdims"]
            op = mx.sym.mean(
                cx, axis=axis, name="var_mean", keepdims=keepdims)
            # original output
            ex_1 = op.bind(ctx, {'cx': data})
            out_1 = [o.round() for o in ex_1.forward()]
            # rewritten graph
            nop, _ = attach_input_shape(op, {}, {"cx":data.shape})
            nop, _ = rewrite(nop, {})
            # fused output
            ex_2 = nop.bind(ctx, {'cx': data})
            out_2 = ex_2.forward()
            assert len(out_1) == len(out_2) == 1
            places = self.assert_equal_rel_places(out_1[0], out_2[0])
            print("config: {}, places: {}".format(config, places))

if __name__ == "__main__":
    unittest.main()
