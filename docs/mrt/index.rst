
*****************
MRT Documentation
*****************

.. contents::

MRT, short of **Model Representation Tool**, aims at transforming floating model into a deterministic and non-data-overflow(containging upflow and downflow) format that CVM defines. Currently the tool only supported AI framework - MxNet, and we have developed the transformation program from TensorFlow into MxNet for google developer. PyTorch is also in our consideration, just wait for some time.

Pre-train Model
===============

There have been many pre-trained models in `Gluon Model Zoo <https://gluon-cv.mxnet.io/model_zoo/index.html>`_, these models are wide-spread used and regarded as an reference to various situations. Thus we do the quantization for those common floating models and expose an whole sequence of MRT execution as the mnist tutorial.

.. _mrt_model_list:

Pre-quantized Model List
------------------------

All the available models, which have benn quantized and tested accuracy in *MxNet Gluon Zoo*, are located in the ``python/mrt/model_zoos`` directory for reference.

.. note::
  The pre-quantized models are executed via MRT's configuration file settings. Refer the
  :ref:`mrt_conf_file` for more details please.

.. toctree::
  :maxdepth: 3

  model.md

.. _mrt_mnist_tutorial:

Mnist Tutorial
--------------

.. toctree::
  :maxdepth: 3

  mnist_tutorial.md

Quantization Documentation
==========================

.. toctree::
  :maxdepth: 3

  quantize.md

V2 Documentation
================

.. toctree::
  :maxdepth: 3

  V2.rst

API Documentation
=================

Core python API briefly involves model quantization, compilation and evaluation. For standardized implementation of MRT stages, please refer to :ref:`MRT Main2 API <mrt_main2_api>`.

For other detailed classes and methods, please refer to the API links.

.. toctree::
  :maxdepth: 1

  Configuration File Main <api/main2>
  Dataset Loader <api/dataset>
  Transformer <api/transformer>
  Operator List <api/operator>
  Graph API <api/graph>
  Helper Utils <api/utils>

Deprecated Docs
===============

.. toctree::
  :maxdepth: 2

  MRT Main2 API <api/main2>
  MRT Transformer API <api/transformer>
  MRT Operator API <api/operator>
  MRT Graph API <api/graph>
  MRT Dataset API <api/dataset>
  MRT Utils API <api/utils>
  MRT Generalized Quantization API <api/gen>
  
An Example
===============

.. toctree::
  :maxdepth: 3

  example.md
  
