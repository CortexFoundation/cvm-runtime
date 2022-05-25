import unittest
import mxnet as mx

if mx.__version__ > '1.5.1':
    print(f"[Warning] Unknow Version: {mx.__version__}")


class TestUniformAffineQuantizer(unittest.TestCase):
    """Test qweight.py"""

    @classmethod
    def setUpClass(cls):
        #  import site
        #  site.addsitedir('../')
        from mrt import yamrt
        pass

    @classmethod
    def tearDownClass(cls):
        pass

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_shape(self):
        weight = mx.sym.Variable('conv', shape=(64,3,7,7))
        bias = mx.sym.Variable('bias', shape=(64,))
        input_data = mx.sym.Variable('input_data', shape=(64, 3, 224, 224))
        zero_point = mx.sym.Variable('zero_point', shape=(1))
        delta = mx.sym.Variable('delta', shape=(1))
        qweight = mx.symbol.Custom(data=weight, delta=delta, zero_point=zero_point, name='conv1_wq', op_type='MRT_UniformAffineQuantizer', n_bits=32)
        conv = mx.sym.Convolution(input_data, weight, bias, kernel=(7,7), num_filter=64)
        qconv = mx.sym.Convolution(input_data, qweight, bias, kernel=(7,7), num_filter=64)

        self.assertEqual(conv.infer_shape()[0][1], weight.infer_shape()[1][0])
        self.assertEqual(conv.infer_shape()[0][1], qweight.infer_shape()[1][0])

        self.assertEqual(qconv.infer_shape()[0][1], qweight.infer_shape()[1][0])
        self.assertEqual(qconv.infer_shape()[0][1], qweight.infer_shape()[1][0])

    def test_forward(self):
        weight = mx.sym.Variable('weight', shape=(64,3,7,7))
        bias = mx.sym.Variable('bias', shape=(64,))
        input_data = mx.sym.Variable('input_data', shape=(64, 3, 224, 224))
        zero_point = mx.sym.Variable('zero_point', shape=(1))
        delta = mx.sym.Variable('delta', shape=(1))
        qweight = mx.symbol.Custom(data=weight, delta=delta, zero_point=zero_point, name='conv1_wq', op_type='MRT_UniformAffineQuantizer', n_bits=32)
        conv = mx.sym.Convolution(input_data, weight, bias, kernel=(7,7), num_filter=64)
        qconv = mx.sym.Convolution(input_data, qweight, bias, kernel=(7,7), num_filter=64)
        args = {"input_data": mx.nd.ones([43,3,224,224]), "weight": mx.nd.ones([64,3,7,7]), "delta": mx.nd.ones([1]), "zero_point": mx.nd.ones([1]), "bias": mx.nd.ones([64])}
        qargs = {"input_data": mx.nd.ones([43,3,224,224]), "weight": mx.nd.ones([64,3,7,7]), "delta": mx.nd.ones([1]), "zero_point": mx.nd.ones([1]), "bias": mx.nd.ones([64])}
        c = conv.bind(mx.cpu(), args=args, args_grad={})
        qc = qconv.bind(mx.cpu(), args=qargs, args_grad={})
        res = c.forward()
        qres = qc.forward()
        self.assertEqual(len(res), len(qres))
        self.assertTrue(res[0].shape == qres[0].shape)
    
    def test_backward(self): # TODO: 
        weight = mx.sym.Variable('weight', shape=(64,3,7,7))
        bias = mx.sym.Variable('bias', shape=(64,))
        input_data = mx.sym.Variable('input_data', shape=(64, 3, 224, 224))
        zero_point = mx.sym.Variable('zero_point', shape=(1))
        delta = mx.sym.Variable('delta', shape=(1))
        qweight = mx.symbol.Custom(data=weight, delta=delta, zero_point=zero_point, name='conv1_wq', op_type='MRT_UniformAffineQuantizer', n_bits=32)
        conv = mx.sym.Convolution(input_data, weight, bias, kernel=(7,7), num_filter=64)
        qconv = mx.sym.Convolution(input_data, qweight, bias, kernel=(7,7), num_filter=64)
        args = {"input_data": mx.nd.ones([43,3,224,224]), "weight": mx.nd.ones([64,3,7,7]), "bias": mx.nd.ones([64])}
        qargs = {"input_data": mx.nd.ones([43,3,224,224]), "weight": mx.nd.ones([64,3,7,7]), "delta": mx.nd.ones([1]) + 1, "zero_point": mx.nd.ones([1]) + 128, "bias": mx.nd.ones([64])}
        grad = {"weight": mx.nd.zeros([64,3,7,7]), "bias": mx.nd.zeros([64])}
        qgrad = {"weight": mx.nd.zeros([64,3,7,7]), "delta": mx.nd.zeros([1]), "zero_point": mx.nd.zeros([1]), "bias": mx.nd.zeros([64])}

        c = conv.bind(mx.cpu(), args=args, args_grad=grad)
        qc = qconv.bind(mx.cpu(), args=qargs, args_grad=qgrad)
        res = c.forward(is_train=True)
        qres = qc.forward(is_train=True)
        self.assertEqual(len(res), len(qres))
        self.assertTrue(res[0].shape == qres[0].shape)
        
        c.backward(res[0].detach())
        qc.backward(qres[0].detach())

        self.assertEqual(c.grad_dict['weight'].shape, qc.grad_dict['weight'].shape)
        self.assertGreater(qc.grad_dict['delta'], 0)


    #def test_minus(self):
    #    """Test method minus(a, b)"""
    #    self.assertEqual(1, minus(3, 2))
    #    self.assertNotEqual(1, minus(3, 2))

    #@unittest.skip("do't run as not ready")
    #def test_minus_with_skip(self):
    #    """Test method minus(a, b)"""
    #    self.assertEqual(1, minus(3, 2))
    #    self.assertNotEqual(1, minus(3, 2))
