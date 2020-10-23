
***********
Get Started
***********

.. contents::


CVM Python Package
==================

CVM C++ Interface
=================

MRT Quantization
================

For a whole glance at the MRT execution, refer to the simple mnist model's quantization 
tutorial: :ref:`mrt_mnist_tutorial`.

Generally, MRT supports two kind of quantization methods,
including ``Configuration File`` and ``Code Program``.
The configuration file is achieved to wrapper complicate
interface and execute with limited quantize settings listed
at the ``.ini`` file. As for the code program, refer to 
the public API please.

.. _mrt_conf_file:

Configuration File
------------------

Run the following command to execute quantization:

.. code-block::

  python python/mrt/main2.py config/file/path

The sample configure file is located at ``python/mrt/model_zoo/config.example.ini`` (TODO: add link),
copy and set your model's quantization settings locally.
And more details about the configure keys have added 
neccessary comments in the example file.

Many pre-quantized models and it's corresponding configures
are also deployed in the same directory. We have quantized 
and tested accuracy for some availble models in MxNet gluon
zoo. These accuracies are organized into a chart for analysis
at :ref:`Model Testing <mrt_model_list>`.

Besides, the unify quantization procedure is defined in file:
``python/mrt/main2.py``, refer to the source code (TODO: add link) for more details.

API Relative
-----------------

The Main public quantization API is located at cvm/quantization/transformer.py. And the main quantization procedure is: 

    Model Load >>> Preparation >>> [Optional] Model Split >>>
    
    Calibration >>> Quantization >>> [Optional] Model Merge >>> Compilation to CVM,

which maps the specific class methods: 

    Model.load >>> Model.prepare >>> [Optional] Model.split >>> 
    
    MRT.calibrate >>> MRT.quantize >>> [Optional] ModelMerger.merge >>> Model.to_cvm.

The Calibration and Quantization pass is achieved in class MRT.

Split & Merge
~~~~~~~~~~~~~

MRT supports for most of MXNet operators while there still exists some unsupported. We advise splitting the model into two sub-graph if there are some unsupported operators and only quantizing the half model (named base_model, indicating the input nodes to split operators generally). In other words, it's the user's responsibility to select the split keys of splitting the original model, while the output-half model is ignored to quantization pass. 

