from .rules import FuseRule
import torch.nn as nn
import torch.quantization as quant


def fuse_model(model, rules, inplace=True):
    """Fuse the model with a list of rules.
    Args:
        model: A nn.Module to be fused.
        rules: A list of rule object functions as FuseRule.
        inplace: Bool. If True, the model object will be modified.
    Return:
        A new fused model, if inplace is False.
    """
    assert isinstance(model, nn.Module)

    for m in model.named_modules():
        for rule in rules:
            rule.add_module(m)
    modules_to_fuse = list()
    for rule in rules:
        modules_to_fuse += rule.names_lists()
    print(modules_to_fuse)
    model = quant.fuse_modules(model, modules_to_fuse, inplace=inplace)
    return model


def post_training_quant(fused_model, data_loader=None, batches=None, inplace=True):
    """Quant a trained model (int8).
    Args:
        model: A fused nn.Module to be quanted.
        data_loader: A data loader provides input data iterations.
        batches: The limitation of iteration(batch) number.
        inplace: Bool. If True, the model object will be modified.
    Return:
        A new quanted model, if inplace is False.
    """
    if not hasattr(fused_model, "qconfig"):
        fused_model.qconfig = quant.get_default_qconfig('fbgemm')
    if inplace:
        quant.prepare(fused_model, inplace=True)
    else:
        fused_model = quant.prepare(fused_model, inplace=False)

    if data_loader is not None:
        for idx, _in in enumerate(data_loader):
            if batches is not None:
                print(f'\r{idx} / {batches}', end='')
            output = fused_model(_in)
            if batches is not None and idx >= batches:
                break
    if inplace:
        quant.convert(fused_model, inplace=True)
    else:
        fused_model = quant.convert(fused_model, inplace=False)

    return fused_model