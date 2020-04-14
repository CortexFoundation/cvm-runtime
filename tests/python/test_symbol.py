import cvm
from cvm import utils
from cvm import nd


def test_abs():
    x = cvm.sym.var('data', shape=(3), precision=8)
    y = cvm.sym.var('y', shape=(3), precision=8)
    x = x + y
    params = {'y': nd.array([-11, 5, 108])}
    x = cvm.sym.clip(x, a_min=-10, a_max=10)
    sym = cvm.sym.Group(x, y)
    op_name = y.attr("op_name")
    print("op name = ", op_name)

    def _print(sym, params, graph):
        print (sym.attr('op_name'), sym.attr('name'), graph.keys())

    sym, params = utils.topo_visit(sym, params, _print)
    return sym, params


if __name__ == "__main__":
    sym, params = test_abs()
    graph = cvm.graph.build(sym, params)
    print (graph.json())

    json_str = graph.json()
    param_bytes = nd.save_param_dict(params)

    model = cvm.runtime.CVMAPILoadModel(
        json_str, param_bytes, cvm.cpu())
    data = nd.array([-30, 0, 10], dtype="int8").as_runtime_input()
    print (len(data))
    out = cvm.runtime.CVMAPIInference(
        model, data)
    cvm.runtime.CVMAPIFreeModel(model)
    print (out)


