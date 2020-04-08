from cvm._ctypes import ndarray as nd
import numpy as np

npa = np.array([1,2,3,4])
nda = nd.array(npa)
print(nda.asnumpy())



