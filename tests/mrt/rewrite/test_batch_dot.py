import numpy as np

import mxnet as mx
from mxnet import ndarray as nd

from mrt.tfm_base import N

def get_norm(data):
    data = data.asnumpy()
    ndims = np.product(data.shape)
    data = np.reshape(data, (ndims,))
    norm = np.linalg.norm(data)
    return norm, data

@N.register_nm("test")
def verify_batch_dot(ashp, bshp, transpose_a, transpose_b):
    A_np = np.random.uniform(size=ashp)
    B_np = np.random.uniform(size=bshp)
    A = nd.array(A_np)
    B = nd.array(B_np)

    # org op
    y = nd.batch_dot(A, B, transpose_a, transpose_b)

    # rewrite op
    andims, bndims = len(ashp), len(bshp)
    assert andims == 3 and bndims == 3, \
        "batch_dot currently only support 3D*3D array." + \
        "name: (%s), op_name: (%s)" % (name, op_name)

    if transpose_a:
        ashp = ashp[:-2] + (ashp[-1], ashp[-2])
        axes = tuple(range(andims-2)) + (andims-1, andims-2)
        A = nd.transpose(
            A, axes=axes, name=N.n("transpose_a"))

    if transpose_b:
        bshp = bshp[:-2] + (bshp[-1], bshp[-2])
        bndims = len(bshp)
        axes = tuple(range(bndims-2)) + (bndims-1, bndims-2)
        B = nd.transpose(
            B, axes=axes, name=N.n("transpose_b"))

    assert ashp[-1] == bshp[1]
    C, MATRIX_MAXIMUM_SIZE = ashp[-1], 4096
    if ashp[-1] <= MATRIX_MAXIMUM_SIZE:
        op = nd.batch_dot(A, B, name=N.n("batch_dot"))
    else:
        C, nodes, step, start = \
            ashp[-1], [], MATRIX_MAXIMUM_SIZE, 0
        while start < C:
            stop = min(start+step, C)

            begin, end = (0,0,start), (ashp[0],ashp[1],stop)
            Ak = nd.slice(
                A, begin=begin, end=end, name=N.n("slice_a"))

            begin, end = (0,start,0), (bshp[0],stop,bshp[2])
            Bk = nd.slice(
                B, begin=begin, end=end, name=N.n("slice_b"))

            tmp = nd.batch_dot(
                Ak, Bk, name=N.n("batch_dot"))
            nodes.append(tmp)
            start += step

        while len(nodes) > 1:
            A, B = nodes.pop(0), nodes.pop(0)
            tmp = nd.elemwise_add(A, B, name=N.n("elemwise_add"))
            nodes.append(tmp)

        op = nodes[0]

    z = op
    # compare
    assert z.shape == y.shape
    zn, zp = get_norm(z)
    yn, yp = get_norm(y)
    rn = np.linalg.norm(zp-yp)
    print(zn, yn, rn)

if __name__ == '__main__':
    verify_batch_dot(
        (4,16384,512),
        (4,16384,256),
        True,
        False)
    verify_batch_dot(
        (4,1024,512),
        (4,256,1024),
        True,
        True)
