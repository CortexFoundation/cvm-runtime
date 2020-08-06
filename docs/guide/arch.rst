
***********************
Design and Architecture
***********************

This document is intended for developers who want to understand the architecture of cvm-runtime and/or actively develop on the project. This page is organized as follows:

- The `System Architecture`_ introduces all the important details
  of system design. To get started, please read this section first.

- The `Example Inference Flow`_ gives an overview of the steps
  that CVM takes to load a hard-disk model into CPU/GPU memory
  for running the deterministic result. 

- The `Operator Math Formalization`_ section describes the ideal
  process logic for specific input data with strict mathematical
  description, guiding the source code.

.. contents::
  :depth: 3

System Architecture
===================

Assertion
---------

- source code: `include/utils/logging.h`.

The assertion do runtime checks and may generate two exceptions,
which respectively stands for runtime error inherited from
`std::runtime_error` and logic error inherited from `std::logic_error`.
The corresponding exception classes declared refer to
:cpp:class:`utils::LogMessageFatal` and
:cpp:class:`utils::ValueVerifyFatal` for more details.

Developer need to assert conditions with pre-defined macros,
such as :c:func:`CHECK` and :c:func:`VERIFY`. :c:func:`CHECK`
macro will throw runtime exception and :c:func:`VERIFY` will
throw logic error. A example usage likes this:
  
.. code-block:: C

  CHECK(condition) << "error information";
  VERIFY(condition) << "error information";

Now it's important to understand the difference between the two
exceptions. One should know the cvm-runtime project is intergral
to the CVM in Cortex Foundation's full-node: CortexTheasus. A
inference call in the cortex blockchain will cost the Endophin,
A calculation unit for model inference takes up, including
memory, time-spending, etc. And then according to the Endophin
cost, the logic error will consume the invoker's CTXC token 
even if the inference fails, whereas the runtime error won't.

**Briefly, a logic error is caused by model supplier or invoker
usually, so it's user's responsibility to take the failure.
And the generic situation that a runtime error occurs is out of
source code bug.**

And one another noticable thing is that cvm-runtime uses exception
to record errors, and it's a big offense to segement fault or dump.
Try your best to avoid core dump and use CHECK macro to check if
you are uncertain to some conditions.

Example Inference Flow
======================

This section will guide the developer to walk the codebase of an
inference flow, from C library API.

Library API
-----------

**cvm-runtime** defines the inference interface as below:

#. :cpp:func:`CVMAPILoadModel`: load model of network handle

#. :cpp:func:`CVMAPIInference`: inference via network handle

#. :cpp:func:`CVMFreeModel`: free model space by handle

- declaration: `include/cvm/c_api.h`
- definition: `src/core/c_api.cc`

And the relative NDArray data API functions are:

#. :cpp:func:`CVMArrayAlloc`: allocate the NDArray space
#. :cpp:func:`CVMArrayFree`: free the NDArray space

- declaration: `include/cvm/runtime/c_runtime_api.h`
- definition: `src/runtime/ndarray.cc`

A simple inference code blocks refer to:

.. code-block:: CPP

  void *handle;
  CVMAPILoadModel(..., &net, ...);

  CVMArrayHandle data;
  CVMArrayAlloc(..., &data);

  CVMArrayHandle output;
  CVMArrayAlloc(..., &output);

  CVMAPIInference(handle,
                  data->data, data_len, 
                  &output->data, output_len);

  ... // process output data

  CVMAPIFreeModel(handle);
  CVMArrayFree(data);
  CVMArrayFree(output);

CVM Model
---------

- declaration: `inclulde/cvm/model.h`
- definition: `src/core/model.cc`

The API above invokes the lower level
:cpp:class:`cvm::runtime::CVMModel` class, which exposes the 
mainly inference interface.

And the :cpp:class:`cvm::runtime::CVMModel` wrappered a pure 
on-device graph runtime, and set extra post-process logic on 
output Tensor.
More details refer to source code please.

Graph Runtime
-------------

- declaration: `src/runtime/graph_runtime.h`
- definition: `src/runtime/graph_runtime.cc`

The :cpp:class:`cvm::runtime::CVMRuntime` manages all the 
resources allocated via CPU, CUDA, OPENCL, ..., etc.
It will pre-allocate the model memory space on device and 
recognize the right order that operators execute.

:cpp:func:`cvm::runtime::CvmRuntime::Init()` initialize the
model network and do some neccessary checks such as shape infer,
precision check, ..., etc.

:cpp:func:`cvm::runtime::CvmRuntime::Setup()` will prepare
all the resources the model needs. Set up procedure has three
steps:

1. Plan Storage

  Using optimizor to pre-calculate the operator memory size.

2. Setup Storage

  Allocate needed memory space and create NDArray node.

3. Setup Op Execs

  Connect all the NDArray node via operators and return the
  lambda function to execute. The operators are registered
  in some other places.

The :cpp:func:`cvm::runtime::CvmRuntime::Run()` will trigger
the execution for model inference.

Operators
---------

Operators are some abstract mathematical process logic set.
Currently cvm-runtime has supportted all 33 operators, grouped
by similar properties. Specific operators name refer to
:ref:`the op list <op_list>`.

All operators are the instance for class :cpp:class:`cvm::Op`, 
with different attribites, inputs, parameters, outputs.

Generally, we register an new opertors using the pre-defined
macro :c:func:`CVM_REGISTER_OP`. And all the operators
registry are located at directory `src/top`.

Besides, all the above registry function won't set the specific
operators' inference logic. It's a device-relative code
implementation, so we put the forward function code in the
`src/runtime/{device_type}/ops` directory according to the
device type. Currently we have achieved three on-device version:

- CPU
- CUDA
- FORMAL

The OPENCL is still in progress and not the latest version.
Please refer to the source code for more detail process logic
and the next section will abstract the mathematical logic in
the formalization format.


.. _operator_math_formalization:

Operator Math Formalization
===========================

.. note::

  Write this section document refer to the doc:
  :ref:`Math Format <write_math_formalization>` please.

This will give a full exhaustive explanation to CVM operators.
The FORMAL version source code has a strong correlation
with this mathematical description, while other versions like
CPU, CUDA, will only promise the consistent inference result,
with arbitrary process logic.

All the operators' formalization obeys the unify format:

.. math::

  Y[y_\text{indices}] = X[x_\text{indices}], \\

  \forall \text{given range}, \\

  \text{where } \text{condition}_1 \text{ and } \text{condition}_2 \text{ and } 
  \cdots \text{condition}_n

which means that for given value range, the forluma in the first 
line is always true, subjecting to the constraints listed as the 
condition variable.

The quick operators reference is listed as below:

.. _op_list:

.. contents::
  :local:


Reduce Operator
---------------

Reduce operators perform the reduction function to input data based on the parameters, and the process logic over all the type-based operators is consistent. We abstract the common reduce
logic as formalization here and specific the reduce function
for each operators respectively.

- Input: :math:`X`, whose shape is :math:`N` dimension, declared
  as :math:`(n_0, n_1, \cdots, n_{N-1})`
- Output: :math:`Y`
- Attribute:

  + `axes` (TShape), which is :math:`M`-length vector,
    where :math:`M \in [0, N+1)`
  + `keepdims` (bool)
  + `exclude` (bool)

.. math::

  T = \left\{x \mid i \in \text{axes} \wedge 
  x = \begin{cases}
    i, & \text{if } i\geqslant 0 \\
    i + N, & \text{otherwise}
  \end{cases} \right\}, \\

  \text{where } card\{T\} = M \text{ and }
    j \in [0, N), \forall j \in \text{T}

.. math::
  
  U = \{ 0, 1, \cdots, N-1 \}

.. math::

  R = \begin{cases}
    U - T, & \text{if exclude is true} \\
    T, & \text{otherwise}
  \end{cases}

.. math::

  r = card\{R\}


1. Case `exclude` is true and :math:`M = N`

  .. math::
    Y = X


2. Case `exclude` is false and :math:`M = 0`

  .. math::
    Y[\underbrace{0, 0, \cdots, 0}_{K}] = 
      \text{REDUCE_FUNC}(X)

    \text{where } K = \begin{cases}
      1, & \text{if keepdims is false} \\
      N, & \text{otherwise}
    \end{cases}

3. Case `keepdims` is false

  .. math::
    Y[d_{I(0)}, d_{I(1)}, \cdots, d_{I(N-r-1)}] =
      \text{REDUCE_FUNC}(Z)

    \forall d_{I(i)} \in [0, n_{I(i)}) \wedge i \in [0, N-r)

    \text{where } 
    I: [0, N-r) \to U-R, \text{s.t. }
      \forall i < j, I(i) < I(j) \text{ and} \\
    J: [0, r) \to R, \text{s.t. }
      \forall i < j, J(i) < J(j) \text{ and} \\
    Z = \{ X[d_0, d_1, \cdots, d_{N-1}] \mid
      d_i \in [0, n_i) \wedge i \in J \}

4. Otherwise

  .. math::
    Y[d_0, d_1, \cdots, d_{N-1}] =
      M[d_{I(0)}, d_{I(1)}, \cdots, d_{I(N-r-1)}], \\

    \forall d_i \in [0, n_i) \wedge i \in U - R
      \wedge d_j = 0 \wedge j \in R \\

    \text{where } 
    I: [0, N-r) \to U-R, \text{s.t. }
      \forall i < j, I(i) < I(j) \text{ and} \\
    J: [0, r) \to R, \text{s.t. }
      \forall i < j, J(i) < J(j) \text{ and} \\
    M = \text{OP_NAME}(X,
      \text{axes=axes, keepdims=false, exclude=exclude})

sum
~~~

- Set :math:`\text{OP_NAME}` as `sum`
- Set :math:`\text{REDUCE_FUNC}` as

  .. math::
    \text{REDUCE_FUNC}(Z) = \sum Z,

*Example*

.. code-block:: Python

  data = [[[1, 2], [2, 3], [1, 3]],
          [[1, 4], [4, 3], [5, 2]],
          [[7, 1], [7, 2], [7, 3]]]

  sum(data, axis=(1))
  [[  4.   8.]
   [ 10.   9.]
   [ 21.   6.]]

  sum(data, axis=[1,2])
  [ 12.  19.  27.]

max
~~~

- Set :math:`\text{OP_NAME}` as `max`
- Set :math:`\text{REDUCE_FUNC}` as

  .. math::
    \text{REDUCE_FUNC}(Z) = \max Z,


Broadcast Operators
~~~~~~~~~~~~~~~~~~~

broadcast_add
~~~~~~~~~~~~~

broadcast_sub
~~~~~~~~~~~~~

broadcast_mul
~~~~~~~~~~~~~

broadcast_div
~~~~~~~~~~~~~

broadcast_max
~~~~~~~~~~~~~

NN Operators
------------

conv2d
~~~~~~

dense
~~~~~

relu
~~~~

max_pool2d
~~~~~~~~~~

upsampling
~~~~~~~~~~

Elemwise Operators
------------------

abs
~~~

cvm_precision
~~~~~~~~~~~~~

elemwise_add
~~~~~~~~~~~~

elemwise_sub
~~~~~~~~~~~~

negative
~~~~~~~~

clip
~~~~

cvm_cilp
~~~~~~~~

cvm_right_shift
~~~~~~~~~~~~~~~

cvm_left_shift
~~~~~~~~~~~~~~

Transform Operators
-------------------

repeat
~~~~~~

tile
~~~~

flatten
~~~~~~~

concatenate
~~~~~~~~~~~

transpose
~~~~~~~~~

slice
~~~~~

take
~~~~

cvm_lut
~~~~~~~

slice_like
~~~~~~~~~~

expand_dims
~~~~~~~~~~~

reshape
~~~~~~~

squeeze
~~~~~~~

Vision Operators
----------------

get_valid_count
~~~~~~~~~~~~~~~

non_max_suppression
~~~~~~~~~~~~~~~~~~~








