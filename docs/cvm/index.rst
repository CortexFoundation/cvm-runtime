
*************************
CVM Runtime Documentation
*************************

Introduction
============

The CVM Runtime library is used in @CortexLabs full-node project: `CortexTheseus <https://github.com/CortexFoundation/CortexTheseus/>`_, working for pure deterministic AI model inference.

.. toctree::
  :maxdepth: 2

  Installation <install.md>
  OPs Table <ops.md>

Python API
==========

Core python API briefly involves inference and graph model.

Inference package contains the model load, inference, and free pass etc. A simple inference code likes this:
::

  import cvm
  from cvm.runtime import *

  ctx = cvm.cpu()

  json_str, param_bytes = cvm.utils.load_model(json_path, param_path)
  net = CVMAPILoadModel(json_str, param_bytes, ctx=ctx)

  ...
  data = cvm.ndarray.array(shape, ctx=ctx)
  res = CVMAPIInference(net, data)
  ...

  CVMAPIFreeModel(net)

And the detail methods refer to the API link.

Model graph generally used to construct the network like `keras` does. This module is invoked by `MRT` to transform the MxNet model into CVM format.

.. toctree::
  :maxdepth: 2

  Inference API <api/inference>
  Symbol&Graph <api/graph>
