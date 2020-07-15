
*****************
MRT Documentation
*****************

Introduction
============

MRT, short of Model Representation Tool, aims at quantizing floating model into a deterministic format that CVM expositions. Currently the tool only supported AI framework - MxNet, and we have developed the transformation program from TensorFlow into MxNet for google developer. PyTorch is also in our consideration, just wait for some time.

.. toctree::
  :maxdepth: 2

  README.md
  mrt.md
  model.md
  mnist_tutorial.md


Python API
==========

.. toctree::
  :maxdepth: 2

  MRT main2 API <api/main2>
  MRT Transformer API <api/transformer>
  MRT base API <api/tfm_base>
  MRT operators API <api/tfm_ops>
