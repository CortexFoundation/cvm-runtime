from cvm._ctypes import ndarray as nd
import numpy as np

npa = np.array([1,2,3,4])
nda = nd.array(npa)
print(nda.asnumpy())
npb = np.array([5,6,7,8])
ndb = nd.array(npb)

data = {'a':nda, 'b':ndb}
print(data.items())

ret = nd.save(data)

