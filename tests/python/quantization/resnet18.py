from mxnet.gluon.model_zoo import vision

from resnet import *

SYMBOL_FILE = "./data/resnet18-symbol.json"
PARAMS_FILE = "./data/resnet18-0000.params"

def load_quant_graph(quant_flag):
    layers, channels = ([2, 2, 2, 2], [64, 64, 128, 256, 512])
    qsym_block = ResNetV1Q(BasicBlockV1Q, layers, channels, quant_flag=quant_flag, prefix="")
    return qsym_block

def load_graph(ctx):
    return vision.resnet18_v1(pretrained=True, ctx=ctx, prefix="")

def save_graph(ctx):
    resnet = load_graph(ctx)
    sym = resnet(mx.symbol.Variable('data'))
    with open(SYMBOL_FILE, 'w') as fout:
        fout.write(sym.tojson())
    resnet.save_params(PARAMS_FILE)
