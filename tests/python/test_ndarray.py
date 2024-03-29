import cvm
from cvm import nd
import numpy as np
import time

# npa = np.array([1,2,3,4])
# nda = nd.array(npa, ctx=cvm.cpu())
# print(nda.asnumpy(), nda)
# npb = np.array([[5,6],[7,8]])
# ndb = nd.array(npb)

# print('testing save/load_param_dict')
# data = {'ac':nda, 'b':ndb}
# print(data.items())
# ret = nd.save_param_dict(data)
# loaded_data = nd.load_param_dict(ret)
# print(len(data), len(loaded_data))
# print(loaded_data.items())


npc = np.array(range(60)).reshape((3, 4, 5))
ndc = nd.array(npc, ctx=cvm.cpu())  # npc.dtype is int64 by default
print('testing ndarray setter')
ndc[0] = 1
print('ndc after setitem:', ndc.asnumpy())
print('testing fill all with a scalar:')
ndc[:] = 2
print('ndc after filling:', ndc)
print('testing fill all with a nested-list/ndarray')
ndc[:] = np.array(range(60, 120)).reshape((3, 4, 5))
print('ndc after filling:', ndc)


print('testing nd with proper shape')
ndc[0] = np.array(range(20, 40)).reshape((4, 5))
print('testing list with proper shape')
ndc[1] = np.array(range(40, 60)).reshape((4, 5)).tolist()
print('testing shape with need of broadcasting')
ndc[2] = np.array(range(60, 65))
print('ndc after filling:', ndc)

ndc[[0, 1], 1] = [2]
bool_array = np.array([[True, False, True, False],
                        [False, True, False, True],
                        [True, False, True, False]], dtype=np.bool)
ndc[bool_array, 0] = [4]
