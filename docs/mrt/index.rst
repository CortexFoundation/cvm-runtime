
*****************
MRT Documentation
*****************

Introduction
============

MRT, short of Model Representation Tool, aims at quantizing floating model into a deterministic format that CVM expositions. Currently the tool only supported AI framework - MxNet, and we have developed the transformation program from TensorFlow into MxNet for google developer. PyTorch is also in our consideration, just wait for some time.

.. toctree::
  :maxdepth: 3

  README.md
  mrt.md
  model.md
  mnist_tutorial.md


Python API
==========

Core python API briefly involves model quantization, compilation and evaluation. For standardized implementation of MRT stages, please refer to :ref:`MRT Main2 API <mrt_main2_api>`

For other detailed classes and methods, please refer to the API links.

`MRT` Module will preprocess the MxNet model and transform the it into CVM format. Inference process can be performed using `CVM` python API.

.. toctree::
  :maxdepth: 2

  MRT Main2 API <api/main2>
  MRT Transformer API <api/transformer>
  MRT base API <api/tfm_base>
  MRT operators API <api/tfm_ops>
  MRT utils API <api/tfm_utils>
  MRT helper API <api/sim_quant_helper>
  MRT pass API <api/tfm_pass>
