from mxnet.gluon.model_zoo import vision
import mxnet as mx

def load_inception_v3(ctx):
    return vision.inception_v3(pretrained=True, ctx=ctx, prefix="")
def save_inception_v3():
    graph = load_inception_v3(mx.cpu())
    sym = graph(mx.symbol.Variable('data'))
    with open('./data/inception_v3.json', 'w') as fout:
        fout.write(sym.tojson())
    graph.save_params('./data/inception_v3.params')

def load_mobilenet1_0(ctx):
    return vision.mobilenet1_0(pretrained=True, ctx=ctx, prefix="")
def save_mobilenet1_0():
    graph = load_mobilenet1_0(mx.cpu())
    sym = graph(mx.symbol.Variable('data'))
    with open('./data/mobilenet1_0.json', 'w') as fout:
        fout.write(sym.tojson())
    graph.save_params('./data/mobilenet1_0.params')

def load_mobilenet_v2_1_0(ctx):
    return vision.mobilenet_v2_1_0(pretrained=True, ctx=ctx, prefix="")
def save_mobilenet_v2_1_0():
    graph = load_mobilenet_v2_1_0(mx.cpu())
    sym = graph(mx.sym.var('data'))
    with open('./data/mobilenet_v2_1_0.json', 'w') as fout:
        fout.write(sym.tojson())
    graph.save_parameters('./data/mobilenet_v2_1_0.params')

# save_inception_v3()
# save_mobilenet1_0()
# save_mobilenet_v2_1_0()
