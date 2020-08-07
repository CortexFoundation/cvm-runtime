
*******************
Developer Guideline
*******************

Introduce the develop relative module.

Such as op registry, compiler pass add, extra device(TPU) support.

Another thing is CVM Codebase Walkthrough.

.. contents::

.. _dev_guide:

Supported Operators
===================

MRT Operators
-------------

MRT adopts a wide range of MXNET operator support (include custom operators) which can be compiled into CVM operators. Run MRT to check whether the model is supported:

.. code-block:: bash

  python python/mrt/main2.py python/mrt/model_zoo/<your_model_name>.ini

If there exist operators that are not supported by MRT, the process will be interupted and throw the followling info:

.. code-block:: bash

  NotImplementedError: Transformer <unknown operator name> has not been registered

For more details, please refer to :ref:`MRT Operators API <mrt_tfm_ops_api>`.


CVM Operators
-------------

CVM operators are compiled from MRT operators that are equivalently transformed, thus CVM supports a relatively bounded set of operators.

For more details, please refer to :ref:`CVM Operator List<op_list>`.
