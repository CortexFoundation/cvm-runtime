from cvm import symbol as sym


def test_abs():
    x = sym.Variable('x')
    y = sym.abs(x)
    op_name = y.attr("op_name")
    print("op name = ", op_name)
    op_func = sym.__dict__[op_name]
    z = op_func(x)


test_abs()


