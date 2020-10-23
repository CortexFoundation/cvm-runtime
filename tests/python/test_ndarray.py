import cvm
from cvm import nd
import numpy as np
import time

npa = np.array([1,2,3,4])
nda = nd.array(npa, ctx=cvm.cpu())
print(nda.asnumpy(), nda)
npb = np.array([5,6,7,8])
ndb = nd.array(npb)

data = {'ac':nda, 'b':ndb}
print(data.items())

ret = nd.save_param_dict(data)
loaded_data = nd.load_param_dict(ret)


# print(len(data), len(loaded_data))
print(loaded_data.items())
