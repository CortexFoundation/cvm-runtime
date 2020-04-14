import cvm
from cvm import nd
import numpy as np

npa = np.array([1,2,3,4])
nda = nd.array(npa, cvm.gpu())
print(nda.asnumpy(), nda)
npb = np.array([5,6,7,8])
ndb = nd.array(npb)

data = {'a':nda, 'b':ndb}
print(data.items())

ret = nd.save_param_dict(data)


import mxnet as mx
print (mx.nd.array([1,2,3], mx.gpu()))
