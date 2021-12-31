import unittest

from mxnet import ndarray as nd
import numpy as np


class TestOpEquiv(unittest.TestCase):
    def assert_equal(self, a, b, places=10):
        self.assertEqual(a.shape, b.shape)
        nentry = int(nd.prod(nd.array(a.shape)).asscalar())
        res = (a-b).reshape(shape=(nentry,)).asnumpy()
        norm = np.linalg.norm(res)
        self.assertAlmostEqual(norm, 0.0, places=places)

    def test_elemmul_to_broadcast_mul(self):
        """
            ElemwiseMul --> BroadcastMul
        """
        def get_data(*x, low=-1000.0, high=1000.0):
            lhs = nd.random.uniform(low=low, high=high, shape=x)
            rhs = nd.random.uniform(low=low, high=high, shape=x)
            return lhs, rhs
        inputs = [
            {
                "x": [3,2,5],
                "low": -55000.0,
                "high": 55000.0,
            },
            {
                "x": [11,5,2],
                "low": -7767.0,
                "high": 9989.0,
            },
            {
                "x": [11,5,2,23],
                "low": -77672.0,
                "high": 99892.0,
            },
        ]
        for inp in inputs:
            x = inp["x"]
            low = inp["low"]
            high = inp["high"]
            lhs, rhs = get_data(*x, low=low, high=high)
            out = nd.elemwise_mul(lhs=lhs, rhs=rhs)
            out1 = nd.broadcast_mul(lhs=lhs, rhs=rhs)
            self.assert_equal(out, out1)

    def test_activation_to_sigmoid(self):
        """
            Activation(act_type=sigmoid) --> sigmoid
        """
        def get_data(*x, low=-1000.0, high=1000.0):
            data = nd.random.uniform(low=low, high=high, shape=x)
            return data
        inputs = [
            {
                "x": [3,2,5],
                "low": -55000.0,
                "high": 55000.0,
            },
            {
                "x": [11,5,2],
                "low": -7767.0,
                "high": 9989.0,
            },
            {
                "x": [11,5,2,23],
                "low": -77672.0,
                "high": 99892.0,
            },
        ]
        for inp in inputs:
            x = inp["x"]
            low = inp["low"]
            high = inp["high"]
            data = get_data(*x, low=low, high=high)
            out = nd.Activation(data=data, act_type="sigmoid")
            out1 = nd.sigmoid(data=data)
            self.assert_equal(out, out1, places=4)

    def test_dense_to_dense2d_flatten(self):
        """
            FullyConnected(flatten=True)
            -->
            reshape.FullyConnected(flatten=True)
        """
        def get_data(
            batch_size, num_hidden, *x,
            low=-1000.0, high=1000.0, no_bias=False):
            xshp = (batch_size,) + x
            product = int(nd.prod(nd.array(x)).asscalar())
            wshp = (num_hidden, product)
            data = nd.random.uniform(low=low, high=high, shape=xshp)
            weight = nd.random.uniform(low=low, high=high, shape=wshp)
            if no_bias:
                return data, weight, None, xshp
            bshp = (num_hidden,)
            bias = nd.random.uniform(low=low, high=high, shape=bshp)
            return data, weight, bias
        inputs = [
            {
                "batch_size": 16,
                "num_hidden": 8,
                "x": [3,2,5],
                "no_bias": True,
                "low": -55000.0,
                "high": 55000.0,
            },
            {
                "batch_size": 64,
                "num_hidden": 5,
                "x": [11,5,2],
                "no_bias": False,
                "low": -7767.0,
                "high": 9989.0,
            },
            {
                "batch_size": 13,
                "num_hidden": 3,
                "x": [11,5,2,23],
                "no_bias": True,
                "low": -77672.0,
                "high": 99892.0,
            },
        ]
        for inp in inputs:
            batch_size = inp["batch_size"]
            num_hidden = inp["num_hidden"]
            x = inp["x"]
            no_bias = inp["no_bias"]
            low = inp["low"]
            high = inp["high"]
            # for reference
            data, weight, bias = get_data(
                batch_size, num_hidden, *x, low=low, high=high)
            if no_bias:
                out = nd.FullyConnected(
                    data=data, weight=weight,
                    no_bias=no_bias, flatten=True, num_hidden=num_hidden)
            else:
                out = nd.FullyConnected(
                    data=data, weight=weight, bias=bias,
                    no_bias=no_bias, flatten=True, num_hidden=num_hidden)
            # for comparison
            xshp = data.shape
            shape = (-1,) + xshp[1:]
            data1 = nd.reshape(data=data, shape=shape)
            if no_bias:
                out1 = nd.FullyConnected(
                    data=data, weight=weight,
                    no_bias=no_bias, flatten=True, num_hidden=num_hidden)
            else:
                out1 = nd.FullyConnected(
                    data=data, weight=weight, bias=bias,
                    no_bias=no_bias, flatten=True, num_hidden=num_hidden)
            # validate
            self.assert_equal(out, out1)

    def test_dense_to_dense2d(self):
        """
            FullyConnected(flatten=False)
            -->
            reshape.FullyConnected(flatten=False).reshape
        """
        def get_data(
            input_dim, num_hidden, *x,
            low=-1000.0, high=1000.0, no_bias=False):
            xshp = x + (input_dim,)
            wshp = (num_hidden, input_dim)
            data = nd.random.uniform(low=low, high=high, shape=xshp)
            weight = nd.random.uniform(low=low, high=high, shape=wshp)
            if no_bias:
                return data, weight, None, xshp
            bshp = (num_hidden,)
            bias = nd.random.uniform(low=low, high=high, shape=bshp)
            return data, weight, bias
        inputs = [
            {
                "input_dim": 16,
                "num_hidden": 8,
                "x": [3,2,5],
                "no_bias": True,
                "low": -55000.0,
                "high": 55000.0,
                "batch_axis": 3,
            },
            {
                "input_dim": 64,
                "num_hidden": 5,
                "x": [11,5,2],
                "no_bias": False,
                "low": -7767.0,
                "high": 9989.0,
            },
            {
                "input_dim": 13,
                "num_hidden": 3,
                "x": [11,5,2,23],
                "no_bias": True,
                "low": -77672.0,
                "high": 99892.0,
                "batch_axis": 3,
            },
        ]
        for inp in inputs:
            input_dim = inp["input_dim"]
            num_hidden = inp["num_hidden"]
            x = inp["x"]
            no_bias = inp["no_bias"]
            low = inp["low"]
            high = inp["high"]
            # for reference
            data, weight, bias = get_data(
                input_dim, num_hidden, *x, low=low, high=high)
            if no_bias:
                out = nd.FullyConnected(
                    data=data, weight=weight,
                    no_bias=no_bias, flatten=False, num_hidden=num_hidden)
            else:
                out = nd.FullyConnected(
                    data=data, weight=weight, bias=bias,
                    no_bias=no_bias, flatten=False, num_hidden=num_hidden)
            # for comparison
            xshp = data.shape
            batch_axis = inp.get("batch_axis", 0)
            assert batch_axis < len(xshp), \
                "invalid batch_axis: {}, length of xshp: {}".format(
                batch_axis, len(xshp))
            if batch_axis == len(xshp)-1:
                product = int(nd.prod(nd.array(xshp)).asscalar())
                res_shp = int(product/xshp[batch_axis])
                shape = (res_shp, -1)
            else:
                shape = (-1, xshp[-1])
            data1 = nd.reshape(data=data, shape=shape)
            if no_bias:
                fc = nd.FullyConnected(
                    data=data1, weight=weight,
                    no_bias=no_bias, flatten=False, num_hidden=num_hidden)
            else:
                fc = nd.FullyConnected(
                    data=data1, weight=weight, bias=bias,
                    no_bias=no_bias, flatten=False, num_hidden=num_hidden)
            if batch_axis == len(xshp)-1:
                shape = xshp[:-1] + (num_hidden,)
            else:
                shape = \
                    xshp[:batch_axis] + (-1,) + \
                    xshp[batch_axis+1:-1] + (num_hidden,)
            out1 = nd.reshape(data=fc, shape=shape)
            # validate
            self.assert_equal(out, out1)

if __name__ == "__main__":
    unittest.main()
