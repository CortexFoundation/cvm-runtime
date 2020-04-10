import cvm
from cvm import utils


def test_abs():
    x = cvm.sym.Variable('x')
    y = cvm.sym.Variable('y')
    sym = cvm.sym.Group(x, y)
    op_name = y.attr("op_name")
    print("op name = ", op_name)

    def _print(sym, params, graph):
        print (sym.attr('op_name'), sym.attr('name'), graph.keys())

    sym, params = utils.topo_visit(sym, {}, _print)


test_abs()


