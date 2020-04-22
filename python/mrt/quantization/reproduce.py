import tensorflow as tf
from tensorflow.contrib import slim as contrib_slim

from .tfmodels.research.slim.datasets import dataset_factory
from .tfmodels.research.slim.nets import nets_factory

import numpy as np
import dataset as ds
import sys
import os
from os import path

slim = contrib_slim

tf.app.flags.DEFINE_integer('batch_size', 160)
tf.app.flags.DEFINE_integer('max_num_batches', 10)
tf.app.flags.DEFINE_string('eval_dir', '/home/test/tvm-cvm/data/tfouts/')
tf.app.flags.DEFINE_string('dataset_dir', '/home/test/.test_dataset_dir')
tf.app.flags.DEFINE_string('model_name', 'inception_v3')
tf.app.flags.DEFINE_integer('eval_image_size', 299)
tf.app.flags.DEFINE_integer('dataset_name', 'imagenet')
tf.app.flags.DEFINE_string('dataset_split_name', 'test')
tf.app.flags.DEFINE_string('labels_offset', 1)

FLAGS = tf.app.flags.FLAGS

def main(_):
    # select the dataset
    dataset = dataset_factory.get_dataset(
        FLAGS.dataset_name, FLAGS.dataset_split_name, FLAGS.dataset_dir)

    # create dataset provider
    provider = slim.dataset_data_provider.DatasetDataProvider(
        dataset,
        shuffle=False,
        common_queue_capacity=2 * FLAGS.batch_size,
        common_queue_min=FLAGS.batch_size)

    # select the model
    network_fn = nets_factory.get_network_fn(
        FLAGS.model_name,
        num_classes=(dataset.num_classes - FLAGS.labels_offset),
        is_training=False)



'''
def load_data_1(input_size=224, batch_size=1, layout='NHWC'):
    ds_name = 'imagenet'
    data_iter_func = ds.data_iter(ds_name, batch_size, input_size=input_size)
    data, label = data_iter_func()
    data = data.asnumpy()
    if layout == 'NHWC':
        data = np.transpose(data, axes=[0,2,3,1])
    print('data loaded with shape: ', data.shape)
    return data, label

def run_lite(modelname):
    lite = lite_path[modelname]
    interpreter = tf.lite.Interpreter(model_path=lite)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    # input_details[0]['shape'] = np.array([160, 299, 299, 3])
    output_details = interpreter.get_output_details()

    input_shape = input_details[0]['shape']
    _, input_size, _, _ = input_shape
    # input_data = np.array(np.random.random_sample(input_shape), dtype=np.float32)
    # input_data, label = load_data_1(input_size=input_size, batch_size=160, layout="NHWC")
    input_data, label = load_data_1(input_size=input_size, batch_size=1, layout="NHWC")
    interpreter.set_tensor(input_details[0]['index'], input_data)

    interpreter.invoke()

    output_data = interpreter.get_tensor(output_details[0]['index'])
    acc, acc_top = tf.metrics.accuracy(
        labels=tf.argmax(label, 1), predictions=tf.argmax(output_data, 1))
    print(output_data.shape)


lite_path = {
    'inception_v3': "/data/tfmodels/lite/Inception_V3/inception_v3.tflite",
}

if __name__ == '__main__':
    assert len(sys.argv) >= 2, "Please enter at least 2 python arguments."
    modelname = sys.argv[1]
    run_lite(modelname)
'''

