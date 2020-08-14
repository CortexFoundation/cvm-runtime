
***************
Acknowledgement
***************

.. contents::

The ``cvm-runtime`` project contains two important part,
``CVM Runtime`` and ``MRT``, contributing to the whole community
for ``CortexLabs`` off-chain and on-chain ecosystem.

CVM Runtime
===========

The CVM Runtime library is used in @CortexLabs' full-node project: `CortexTheseus <https://github.com/CortexFoundation/CortexTheseus/>`_, working for pure deterministic AI model inference.

CVM Runtime is a AI model inference framework with strict 
consistency, that is all the final inference result must
be the same, non of the tolerance for any uncertain factor.

Motivation & Application
------------------------

The CVM Runtime is referenced by ``CortexTheseus``, a block-chain
project, to combile the ability for open, decentralization and
artificial intelligence.

As the blockchain requests, all the deployed node should maintain
a consistent state, including memory state, smart contract and
the block chain etc. **And the consistent inference result for any
model execution is also natural.** Ideally, keep the same output
for the same model and input at virous devices and system environments is our first target, and after that, keep all the 
operators output deterministic is another goal to be perform incidentally. Besides, we keep an ambiguous attitude for the 
flexibility and scalability of model graph (or model structure), 
like the sequence alteration or fuse operation for operators in 
a model.

Design & Architecture
---------------------

The CVM Runtime project starts from inheriting the NNVM project.
and the mainstream AI models are represented via the network
structure graph, stacking with some pre-defined, basic operators.
A model usually is organized with the ``tree`` construction.

.. note::
  NNVM is an early deep learning backend project, it's still 
  used and as an intergral part by the ``MxNet`` framework.
  In last few years, NNVM is also the backend for the deep
  learning stack: ``TVM``, it has been replaced by the ``ir``
  intermediate layer.

So we propose an stable, extensible and powerful AI operator set,
that can apply and run most application models, specifically,
the AI models which can well combile with blockchain, such as
image classification, object detection, and NLP etc.

The graph runtime mantains a **full integer** memory state,
and all the operators process the integer data flow and output 
integer data to avoid the non-determinism in floating point,
such as different summation order and multi-threading.
Also, we brought an **data precision schema** to get out of
the overflow or underflow problem.

More design and architecture introduction refers to the section:
:ref:`Design and Architecture <design-and-arch>` please.

Supported Operators
-------------------

One noticable thing to be indicate is that **all the CVM Runtime
operators process logic are deterministic**. We strictly define
the :ref:`operator-math-formalization` for each operator
according to the common AI framework definition.

CVM Runtime have defined the total **36 operators**, and please 
refer to :ref:`CVM Operator List <op_list>` for more 
details.

Model Representation Tool
=========================

MRT, short for **Model Representation Tool**, aims to convert floating model into a deterministic and non-data-overflow network. MRT links the off-chain AI developer community to the on-chain ecosystem, from Off-chain deep learning to MRT transformations, and then uploading to Cortex Blockchain for on-chain deterministic inference.

As the above CVM Runtime section points out, the model that goes
under MRT transformation can be accepted by CVM Runtime, which
we called it on-chain model. MRT propose approaches to transform 
floating model to on-chain model, mainly include:

- do transformation from floating to full integer with minimum 
  accuracy drop.
- certify the process data to be non-flow over INT32.

Design & Architecture
---------------------

MRT is based on the MXNet symbol, doing operations on the whole 
operators with topological order in models. Besides, for 
scalability, we've researched the model transformation from 
TensorFlow into MXNet, models such as mobilenet, inception_v3
have been successfully converted and more operators will be 
supported in the future. Other deep learning frameworks like 
PyTorch and Caffe is in the roadmap of our plan.

MRT transformation usage is simple to model-training programmer 
since we have separated model quantization procedures from
source code. One can invoke MRT via programming or configuring
the settings file, more detail usage refers to
:ref:`dev-guide` please.

Supported Operators
-------------------

MRT adopts a wide range of MXNET operator support
(include custom operators) which can be compiled into CVM 
operators. Please refer to :ref:`MRT Operator 
List <mrt_tfm_ops_api>` for more details.

Reference
---------

Some effective links to MxNet libraries are listed here:

1. `Mxnet Symbol <https://mxnet.apache.org/versions/1.6/api/python/docs/api/symbol/>`_
2. `Gluon Model Zoo <https://gluon-cv.mxnet.io/model_zoo/index.html>`_
