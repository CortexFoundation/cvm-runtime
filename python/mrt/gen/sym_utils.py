import math

from mrt.sym_utils import nd_full

def nd_full_const(number, shape, graph, params):
    assert isinstance(shape, tuple)
    name = 'const_var_' + str(number) + '_' + str(shape)
    if name not in params and name not in graph:
        prec = math.ceil(math.log2(math.fabs(number)+1)) + 1
        attr = { 'precision': str(prec) }
        graph[name] = mx.sym.var(name, shape=shape, attr=attr)
        params[name] = nd_full(shape=shape, val=number)
    return graph[name]
