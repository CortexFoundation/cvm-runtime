from __future__ import print_function  # only relevant for Python 2
import mxnet as mx
from mxnet import nd, gluon, autograd
from mxnet.gluon import nn

from mrt import conf, utils

import numpy as np
import argparse

parser = argparse.ArgumentParser(description='Mnist Traning')
parser.add_argument('--cpu', default=False, action='store_true',
                    help='whether enable cpu (default use gpu)')
parser.add_argument('--gpu-id', type=int, default=0,
                    help='gpu device id')
parser.add_argument('--net', type=str, default='',
                    help='choose available networks, optional: lenet, mlp')

args = parser.parse_args()

def load_fname(version, suffix=None, with_ext=False):
    suffix = "."+suffix if suffix is not None else ""
    version = "_"+version if version is not None else ""
    prefix = "{}/mnist{}{}".format(conf.MRT_MODEL_ROOT, version, suffix)
    return utils.extend_fname(prefix, with_ext)

def data_xform(data):
    """Move channel axis to the beginning, cast to float32, and normalize to [0, 1]."""
    return nd.moveaxis(data, 2, 0).astype('float32') / 255

train_data = mx.gluon.data.vision.MNIST(
        train=True).transform_first(data_xform)
val_data = mx.gluon.data.vision.MNIST(
        train=False).transform_first(data_xform)

batch_size = 4
train_loader = mx.gluon.data.DataLoader(train_data, shuffle=True, batch_size=batch_size)
val_loader = mx.gluon.data.DataLoader(val_data, shuffle=False, batch_size=batch_size)

version = args.net
print ("Training {} Mnist".format(version))

# Set the gpu device id
ctx = mx.cpu() if args.cpu else mx.gpu(args.gpu_id)
print ("Using device: {}".format(ctx))

def train_mnist():
    # Select a fixed random seed for reproducibility
    mx.random.seed(42)

    if version == '':
        net = nn.HybridSequential(prefix='DApp_')
        with net.name_scope():
            net.add(
                nn.Conv2D(channels=16, kernel_size=(3, 3), activation='relu'),
                nn.MaxPool2D(pool_size=(2, 2), strides=(1, 1)),
                nn.Conv2D(channels=32, kernel_size=(3, 3), activation='relu'),
                nn.MaxPool2D(pool_size=(2, 2), strides=(1, 1)),
                nn.Conv2D(channels=64, kernel_size=(3, 3), activation='relu'),
                nn.MaxPool2D(pool_size=(2, 2), strides=(2, 2)),
                nn.Conv2D(channels=128, kernel_size=(1, 1), activation='relu'),
                nn.MaxPool2D(pool_size=(2, 2), strides=(2, 2)),
                nn.Flatten(),
                nn.Dense(10, activation=None),
            )
    elif version == 'lenet':
        net = nn.HybridSequential(prefix='LeNet_')
        with net.name_scope():
            net.add(
                nn.Conv2D(channels=20, kernel_size=(5, 5), activation='relu'),
                nn.MaxPool2D(pool_size=(2, 2), strides=(2, 2)),
                nn.Conv2D(channels=50, kernel_size=(5, 5), activation='relu'),
                nn.MaxPool2D(pool_size=(2, 2), strides=(2, 2)),
                nn.Flatten(),
                nn.Dense(500, activation='relu'),
                nn.Dense(10, activation=None),
            )
    elif version == 'mlp':
        net = nn.HybridSequential(prefix='MLP_')
        with net.name_scope():
            net.add(
                nn.Flatten(),
                nn.Dense(128, activation='relu'),
                nn.Dense(64, activation='relu'),
                nn.Dense(10, activation=None)  # loss function includes softmax already, see below
            )
    else:
        assert False

    # Random initialize all the mnist model parameters
    net.initialize(mx.init.Xavier(), ctx=ctx)
    net.summary(nd.zeros((1, 1, 28, 28), ctx=ctx))

    trainer = gluon.Trainer(
	params=net.collect_params(),
	optimizer='adam',
	optimizer_params={'learning_rate': 1e-3},
    )
    metric = mx.metric.Accuracy()
    loss_function = gluon.loss.SoftmaxCrossEntropyLoss()
    num_epochs = 5

    for epoch in range(num_epochs):
        for inputs, labels in train_loader:
            inputs = inputs.as_in_context(ctx)
            labels = labels.as_in_context(ctx)

            with autograd.record():
                outputs = net(inputs)
                loss = loss_function(outputs, labels)

            loss.backward()
            metric.update(labels, outputs)

            trainer.step(batch_size=inputs.shape[0])

        name, acc = metric.get()
        print('After epoch {}: {} = {:5.2%}'.format(epoch + 1, name, acc))
        metric.reset()

    for inputs, labels in val_loader:
        inputs = inputs.as_in_context(ctx)
        labels = labels.as_in_context(ctx)
        metric.update(labels, net(inputs))
    print('Validaton: {} = {}'.format(*metric.get()))
    assert metric.get()[1] > 0.96

    sym = net(mx.sym.var('data'))
    sym_file, param_file = load_fname(version)
    print ("Dump model into ", sym_file, " & ", param_file)

    # dump the mxnet model
    with open(sym_file, "w") as fout:
        fout.write(sym.tojson())
    net.collect_params().save(param_file)

train_mnist()
