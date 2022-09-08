from yamrt.model import MxnetModelHandler
import mxnet as mx
import numpy as np
import onnx
print(mx.__version__)
print(onnx.__version__)
model = MxnetModelHandler()
sym = './resnet-18-symbol.json'
params = './resnet-18-0000.params'

in_shapes = [(1, 3, 224, 224)]
in_types = [np.float32]

model.load(sym, params, in_shapes, in_types)
model.build()