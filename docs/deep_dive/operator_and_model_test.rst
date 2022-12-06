
.. _operator-and-model-test:

***********************
Operator and Model Test
***********************

.. contents::

Test Method
===========

Operator
--------

First generate test data with test_ops.py, then verify the 
generated data with test_op.cc, execute the following code 
to test(test_op.cc is compiled with Makefile):

.. code-block::

    python tests/test_ops.py
    ./test_op 0/1/2 (0 is cpu, 1 is gpu, and 2 is formal)

Model
-----

First dump cvm model with main.py and corresponding yaml file, then 
verify the generated model with test_model.cc, execute the following 
code to test(test_model.cc is compiled with Makefile):

.. code-block::

    python main.py /tests/mrt/model_zoo/yaml_file
    ./test_model 

Test Results
============

Operator
--------

The following are the test results of the operator on different devices:

+-------------------+-------------------------+
|     operator      |          results        |
+===================+=========================+
| broadcast_sub     | cpu & formal & gpu pass |
+-------------------+-------------------------+
| broadcast_add     | cpu & formal & gpu pass |
+-------------------+-------------------------+
| broadcast_mul     | cpu & formal & gpu pass |
+-------------------+-------------------------+
| broadcast_max     | cpu & formal & gpu pass |
+-------------------+-------------------------+
| broadcast_div     | cpu & formal & gpu pass |
+-------------------+-------------------------+
| broadcast_greater | cpu & formal & gpu pass |
+-------------------+-------------------------+
| max_pool2d        | cpu & formal & gpu pass |
+-------------------+-------------------------+
| dense             | cpu & formal & gpu pass |
+-------------------+-------------------------+
| sum               | cpu & formal & gpu pass |
+-------------------+-------------------------+
| max               | cpu & formal & gpu pass |
+-------------------+-------------------------+
| slice_like        | cpu & formal & gpu pass |
+-------------------+-------------------------+
| tile              | cpu & formal & gpu pass |
+-------------------+-------------------------+
| repeat            | cpu & formal & gpu pass |
+-------------------+-------------------------+
| concatenate       | cpu & formal & gpu pass |
+-------------------+-------------------------+
| transpose         | cpu & formal & gpu pass |
+-------------------+-------------------------+
| take              | cpu & formal & gpu pass |
+-------------------+-------------------------+
| get_valid_counts  | cpu & formal & gpu pass |
+-------------------+-------------------------+
| strided_slice     | cpu & formal & gpu pass |
+-------------------+-------------------------+
| conv2d            | cpu & formal & gpu pass |
+-------------------|-------------------------|
| upsampling        | cpu & formal & gpu pass |
+-------------------|-------------------------|


Model
-----

The following are the test results of the model on different devices:

+---------------------------+-------------------------+
|           model           |          results        |
+===========================+=========================+
| alexnet                   | cpu & formal & gpu pass |
+---------------------------+-------------------------+
| cifar_resnet110_v1        | cpu & formal & gpu pass |
+---------------------------+-------------------------+
| cifar_resnet110_v2        | cpu & formal & gpu pass |
+---------------------------+-------------------------+
| cifar_resnet20_v1         | cpu & formal & gpu pass |
+---------------------------+-------------------------+
| cifar_resnet20_v2         | cpu & formal & gpu pass |
+---------------------------+-------------------------+
| cifar_resnet56_v1         | cpu & formal & gpu pass |
+---------------------------+-------------------------+
| cifar_resnet56_v2         | cpu & formal & gpu pass |
+---------------------------+-------------------------+
| cifar_resnext29_16x64d    | cpu & formal & gpu pass |
+---------------------------+-------------------------+
| cifar_wideresnet16_10     | cpu & formal & gpu pass |
+---------------------------+-------------------------+
| cifar_wideresnet28_10     | cpu & formal & gpu pass |
+---------------------------+-------------------------+
| cifar_wideresnet40_8      | cpu & formal & gpu pass |
+---------------------------+-------------------------+
| densenet121               | cpu & formal & gpu pass |
+---------------------------+-------------------------+
| densenet161               | cpu & formal & gpu pass |
+---------------------------+-------------------------+
| densenet169               | cpu & formal & gpu pass |
+---------------------------+-------------------------+
| densenet201               | cpu & formal & gpu pass |
+---------------------------+-------------------------+
| inceptionv1_kinetics400   | cpu & formal & gpu pass |
+---------------------------+-------------------------+
| mobilenet0.25             | cpu & formal & gpu pass |
+---------------------------+-------------------------+
| mobilenet0.5              | cpu & formal & gpu pass |
+---------------------------+-------------------------+
| mobilenet0.75             | cpu & formal & gpu pass |
+---------------------------+-------------------------+
| mobilenet1.0              | cpu & formal & gpu pass |
+---------------------------+-------------------------+
| mobilenetv2_0.25          | cpu & formal & gpu pass |
+---------------------------+-------------------------+
| mobilenetv2_0.5           | cpu & formal & gpu pass |
+---------------------------+-------------------------+
| mobilenetv2_0.75          | cpu & formal & gpu pass |
+---------------------------+-------------------------+
| mobilenetv2_1.0           | cpu & formal & gpu pass |
+---------------------------+-------------------------+
| mobilenetv3_large         | cpu & formal & gpu pass |
+---------------------------+-------------------------+
| resnet101_v1b             | cpu & formal & gpu pass |
+---------------------------+-------------------------+
| resnet101_v1b_kinetics400 | cpu & formal & gpu pass |
+---------------------------+-------------------------+
| resnet101_v1c             | cpu & formal & gpu pass |
+---------------------------+-------------------------+
| resnet101_v1              | cpu & formal & gpu pass |
+---------------------------+-------------------------+
| resnet101_v1d_0.73        | cpu & formal & gpu pass |
+---------------------------+-------------------------+
| resnet101_v1d_0.76        | cpu & formal & gpu pass |
+---------------------------+-------------------------+
| resnet101_v1d             | cpu & formal & gpu pass |
+---------------------------+-------------------------+
| resnet101_v1s             | cpu & formal & gpu pass |
+---------------------------+-------------------------+
| resnet101_v2              | cpu & formal & gpu pass |
+---------------------------+-------------------------+
| resnet152_v1b             | cpu & formal & gpu pass |
+---------------------------+-------------------------+
| resnet152_v1b_kinetics400 | cpu & formal & gpu pass |
+---------------------------+-------------------------+
| resnet152_v1c             | cpu & formal & gpu pass |
+---------------------------+-------------------------+
| resnet152_v1              | cpu & formal & gpu pass |
+---------------------------+-------------------------+
| resnet152_v1d             | cpu & formal & gpu pass |
+---------------------------+-------------------------+
| resnet152_v1s             | cpu & formal & gpu pass |
+---------------------------+-------------------------+
| resnet152_v2              | cpu & formal & gpu pass |
+---------------------------+-------------------------+
| resnet18_v1b_0.89         | cpu & formal & gpu pass |
+---------------------------+-------------------------+
| resnet18_v1b              | cpu & formal & gpu pass |
+---------------------------+-------------------------+
| resnet18_v1b_kinetics400  | cpu & formal & gpu pass |
+---------------------------+-------------------------+
| resnet18_v1               | cpu & formal & gpu pass |
+---------------------------+-------------------------+
| resnet18_v2               | cpu & formal & gpu pass |
+---------------------------+-------------------------+
| resnet34_v1b              | cpu & formal & gpu pass |
+---------------------------+-------------------------+
| resnet34_v1b_kinetics400  | cpu & formal & gpu pass |
+---------------------------+-------------------------+
| resnet34_v1               | cpu & formal & gpu pass |
+---------------------------+-------------------------+
| resnet34_v2               | cpu & formal & gpu pass |
+---------------------------+-------------------------+
| resnet50_v1b              | cpu & formal & gpu pass |
+---------------------------+-------------------------+
| resnet50_v1b_hmdb51       | cpu & formal & gpu pass |
+---------------------------+-------------------------+
| resnet50_v1b_kinetics400  | cpu & formal & gpu pass |
+---------------------------+-------------------------+
| resnet50_v1b_sthsthv2     | cpu & formal & gpu pass |
+---------------------------+-------------------------+
| resnet50_v1c              | cpu & formal & gpu pass |
+---------------------------+-------------------------+
| resnet50_v1               | cpu & formal & gpu pass |
+---------------------------+-------------------------+
| resnet50_v1d_0.11         | cpu & formal & gpu pass |
+---------------------------+-------------------------+
| resnet50_v1d_0.37         | cpu & formal & gpu pass |
+---------------------------+-------------------------+
| resnet50_v1d_0.48         | cpu & formal & gpu pass |
+---------------------------+-------------------------+
| resnet50_v1d_0.86         | cpu & formal & gpu pass |
+---------------------------+-------------------------+
| resnet50_v1d              | cpu & formal & gpu pass |
+---------------------------+-------------------------+
| resnet50_v1s              | cpu & formal & gpu pass |
+---------------------------+-------------------------+
| resnet50_v2               | cpu & formal & gpu pass |
+---------------------------+-------------------------+
| squeezenet1.0             | cpu & formal & gpu pass |
+---------------------------+-------------------------+
| squeezenet1.1             | cpu & formal & gpu pass |
+---------------------------+-------------------------+
| ssd_512_mobilenet1.0_voc  | cpu & formal & gpu pass |
+---------------------------+-------------------------+
| ssd_512_resnet50_v1_voc   | cpu & formal & gpu pass |
+---------------------------+-------------------------+
| vgg11_bn                  | cpu & formal & gpu pass |
+---------------------------+-------------------------+
| vgg11                     | cpu & formal & gpu pass |
+---------------------------+-------------------------+
| vgg13_bn                  | cpu & formal & gpu pass |
+---------------------------+-------------------------+
| vgg13                     | cpu & formal & gpu pass |
+---------------------------+-------------------------+
| vgg16_bn                  | cpu & formal & gpu pass |
+---------------------------+-------------------------+
| vgg16                     | cpu & formal & gpu pass |
+---------------------------+-------------------------+
| vgg16_ucf101              | cpu & formal & gpu pass |
+---------------------------+-------------------------+
| vgg19_bn                  | cpu & formal & gpu pass |
+---------------------------+-------------------------+
| vgg19                     | cpu & formal & gpu pass |
+---------------------------+-------------------------+
| yolo3_darknet53_voc       | cpu & formal & gpu pass |
+---------------------------+-------------------------+
| yolo3_mobilenet1.0_voc    | cpu & formal & gpu pass |
+---------------------------+-------------------------+
