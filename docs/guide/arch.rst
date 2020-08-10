
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


Reduce Operators
----------------

A reduce operator performs the reduction function to input data based on the parameters, and the process logic over all kind of operators are the same.
Reduction is performed on the given axes, other dimensions remains the same and the result are stored in those places.
We abstract the common reduce logic as formalization here and specify the reduce function
for each operators respectively.

- Input: :math:`X`, a tensor of :math:`N` dimensions, namely :math:`(n_0, n_1, \cdots, n_{N-1})`
- Output: :math:`Y`
- Attribute:

  + ``axes`` (TShape), which is :math:`M`-length vector,
    where :math:`M \in [0, N+1)`
  + ``keepdims`` (bool)
  + ``exclude`` (bool)

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


1. Case ``exclude`` is true and :math:`M = N`

  .. math::
    Y = X


2. Case ``exclude`` is false and :math:`M = 0`

  .. math::
    Y[\underbrace{0, 0, \cdots, 0}_{K}] =
      \text{REDUCE_FUNC}(X),

    \text{where } K = \begin{cases}
      1, & \text{if keepdims is false} \\
      N, & \text{otherwise}
    \end{cases}

3. Case ``keepdims`` is false

  .. math::
    Y[d_{I(0)}, d_{I(1)}, \cdots, d_{I(N-r-1)}] =
      \text{REDUCE_FUNC}(Z),

    \forall d_{I(i)} \in [0, n_{I(i)}) \wedge i \in [0, N-r),

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
      \wedge d_j = 0 \wedge j \in R, \\

    \text{where }
    I: [0, N-r) \to U-R, \text{s.t. }
      \forall i < j, I(i) < I(j) \text{ and} \\
    J: [0, r) \to R, \text{s.t. }
      \forall i < j, J(i) < J(j) \text{ and} \\
    M = \text{OP_NAME}(X,
      \text{axes=axes, keepdims=false, exclude=exclude})

sum
~~~

- Set :math:`\text{OP_NAME}` as ``sum``
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

- Set :math:`\text{OP_NAME}` as ``max``
- Set :math:`\text{REDUCE_FUNC}` as

  .. math::
    \text{REDUCE_FUNC}(Z) = \max Z,


Broadcast Operators
-------------------

A broadcast operator performs the broadcast function to input data, and the process logic over all kinds of operators are the same.

- Input: There are 2 inputs.

 + :math:`A`, a tensor of :math:`M` dimensions, namely :math:`(m_0, m_1, \cdots, m_{M-1})`
 + :math:`B`, a tensor of :math:`N` dimensions, namely :math:`(n_0, n_1, \cdots, n_{N-1})`

- Output: :math:`Y`, a tensor with :math:`max(M, N)` dimensions, the higher dimension of the two inputs, and it's shape is identical to the input with higher dimension.

The lower :math:`min(M, N)` dimensions of the two inputs must be the same. The input with lower dimension is expanded with 1 so that the two inputs can have the same dimension.

Then the elementwise opertaion is performed to the inputs with broadcast.

We abstract the formalization here and introduce the details as below:

.. math::

  Y[d_0, d_1, \cdots, d_{K-1}] = \begin{cases}
    A[d_{N-M}, d_1, \cdots, d_{N-1}] \text{ OP } B[d_0, d_1, \cdots, d_{N-1}], & M \leq N \\
    A[d_0, d_1, \cdots, d_{M-1}] \text{ OP } B[d_{M-N}, d_1, \cdots, d_{M-1}], & M > N
  \end{cases}, \\

  \forall d_i \in [0, n_i) \text{ if } N \geq M \text{ or } d_i \in [0, m_i) \text{ otherwise} \\

  \text{where } i \in [0, max(M, N))\\
  


broadcast_add
~~~~~~~~~~~~~

set :math:`\text{OP}` to :math:`\text{add}`.

*Example*

.. code-block:: Python

  x = [[ 1.,  1.,  1.],
       [ 1.,  1.,  1.]]

  y = [[ 0.],
       [ 1.]]

  broadcast_add(x, y)
  [[ 1.,  1.,  1.],
   [2.,  2.,  2.]]


broadcast_sub
~~~~~~~~~~~~~
set :math:`\text{OP}` to :math:`\text{sub}`.
Note that there's no need to make sure that the dimension of the minuend :math:`A` is higher than subtractor :math:`B`

broadcast_mul
~~~~~~~~~~~~~
set :math:`\text{OP}` to :math:`\text{mutiply}`.

broadcast_div
~~~~~~~~~~~~~
set :math:`\text{OP}` to :math:`\text{divide}`.

broadcast_max
~~~~~~~~~~~~~
set :math:`\text{OP}` to :math:`\text{max}`.

NN Operators
------------
We provide NN operators for users.
In fact, NN operators stand for neural network operators, the core of neural network learning mechanism.
NN operators have parameters to be trained and logic for linear or non-linear transformation in a model graph.

conv2d
~~~~~~
We only supported 2-D convolution operator. Also alias *Group-wise Convolution*.

*Math Formalization*

- Input: there are 2 or 3 inputs.

  + :math:`X`, input data to be calculated whose shape is :math:`(N, C, H, W)`
  + :math:`W`, convolution kernel weight whose shape is :math:`(OC, IC, KH, KW)`
  + :math:`B`, bias, of type ``Optional<DLTensor>``. If :math:`B` is not None, it's shape is :math:`(\text{OC},)`.

- Output: :math:`Y`
- Attributes:

  + ``padding``, a ``TShape`` of length 2, namely :math:`(PH, PW), PH,PW \in [min\_attr, max\_attr)`, indicating padding size.
  + ``stride``, a ``TShape`` of length 2, namely :math:`(SH, SW) \in [1, max\_attr)`, indicating strides.
  + ``dilation``, a ``TShape`` of length 2, namely :math:`(DH, DW) \in [1, max\_attr)`, parameter used in dilation convolution.
  + ``groups``, an ``int`` in :math:`\text{range} [1, C]`, indicating group number. :math:`C = IC \cdot \text{groups} \wedge OC \text{ mod } \text{groups} = 0`

.. math::

  Y[n,oc,p,q]= \sum_{ic = 0}^{IC-1} \text{kernel}(n,(oc \div OPG) * IC + ic, p, q, oc,ic) + \begin{cases}
  0, & \text{if B is None}\\
  B[oc], & \text{otherwise}
  \end{cases}, \\

  \forall n \in [0, N) \wedge oc\in [0, OC) \wedge
  p \in \left[0, \text{Y_HMAX} \right) \wedge 
  q \in \left[0, \text{Y_WMAX} \right),

  \text{where } \text{Y_HMAX} = \left\lfloor{H+2 \cdot \text{PH}-\text{DH} \cdot (\text{KH}-1)-1\over\text{SH}}\right\rfloor+1\wedge\\
  \text{Y_WMAX} = \left\lfloor{W+2 \cdot \text{PW}-\text{DW} \cdot (\text{KW}-1)-1 \over \text{SW}}\right\rfloor+1 \wedge\\
  OPG = OC / \text{groups, }  OPG \in \mathbb N^+ \text{ since } OC \text{ mod } \text{groups} = 0\\

where :math:`\text{kernel}` function does the 2D image convolution calculation, and the formulation is

.. math::

  \text{kernel}(n, j, p, q, o, i) = \sum_{k_i=0}^{\text{KH}} \sum_{k_j = 0}^{\text{KW}} \text{pad}(p'+k_i*\text{DH},q'+k_j*\text{DW}) \cdot W[o, i, k_i, k_j], \\

  \text{where } p' = p \cdot \text{SH} -\text{PH} \text{ and }
  q' = q \cdot \text{SW}-\text{PW} \text{ and } \\
  \text{pad}(p, q) = \begin{cases}
  X[n, j, p, q], & \text{ if } p \in [0, H) \wedge q \in [0, W) \\
  0, & \text{otherwise}
  \end{cases}


dense
~~~~~
Dense operator provides a full connected layer.

*Math Formalization*

- Input: there 2 or 3 inputs.

  + :math:`X`, a matrix of shape :math:`(M, K)`
  + :math:`W`, a matrix of shape :math:`(N, K)`
  + :math:`B`, bias, of type ``Optional<DLTensor>``, If :math:`B` is not ``NONE``, it's shape is :math:`(N,)`.

- Output: :math:`Y`, a matrix of shape :math:`(M, N)`

.. math::

  Y=X W^T + \begin{cases}
  0, & \text{if B is None} \\
  B, & \text{otherwise}
  \end{cases}

relu
~~~~

Relu performs elementwise rectified linear unit function.

*Math Formalization*

- Input: :math:`X`, a tensor of :math:`N` dimensions, namely :math:`(n_0, n_1, \cdots, n_{n-1})`
- Output: :math:`Y`, the same shape as :math:`X`

.. math::
  Y[d_0, d_1, \cdots, d_{n-1}] = max(0, X[d_0, d_1, \cdots, d_{n-1}]) \\
  \forall i \in [0, N), d_i \in [0, n_i)

max_pool2d
~~~~~~~~~~
Max_pool2d performs max pooling over every plane for each batch and channel.

*Math Formalization*

- Input: :math:`X`, of shape :math:`(N, C, H, W)`, indicating there are :math:`N` batches, :math:`C` channels and all the planes are of size :math:`(H, W)`
- Output: :math:`Y`
- Attributes:

  + ``pool_size``, a ``TShape`` of length 2, namely :math:`(PSH, PSW)`
  + ``padding``, either a ``TShape`` of length 2, namely :math:`(PH, PW) \in [min\_attr, max\_attr)`, or an int indicating :math:`PH=PW`
  + ``strides``, a ``TShape`` of length 2, namely :math:`(SH, SW)`
  + ``ceil_mode``, ``boolean``.

.. math::
  PSH \in [0, H + 2PH + 1), \\
  PSW \in [0, W + 2PW + 1), \\
  PSH > PH \wedge PSW > PW

.. math::

  Y[n,i,p,q] = \max\{\text{pad}(n, i, p', q') \\
  \mid p' \in [p \cdot \text{SH} -\text{PH}, p \cdot \text{SH} -\text{PH}+\text{PSH}),
  q' \in [q \cdot \text{SW}-\text{PW}, q \cdot \text{SW}-\text{PW}+\text{PSW})\}, \\

  \forall n \in [0, N) \wedge i \in [0, C) \wedge \\
  p \in \left[0, \text{ceil_func}\left({H+2 \cdot \text{PH}-  \text{PSH}\over\text{SH}}\right)+1 \right) \wedge \\
  q \in \left[0, \text{ceil_func}\left({W+2 \cdot \text{PW}- \text{PSW} \over \text{SW}}\right)+1 \right), \\

  \text{where } \text{ceil_func(val)} = \begin{cases}
  \lceil \text{val} \rceil, & \text{if ceil_mode is true} \\
  \lfloor \text{val} \rfloor, & \text{otherwise}
  \end{cases} \text{ and } \\
  \text{pad}(n, i, p, q) = \begin{cases}
  X[n, i, p, q], & \text{ if } p \in [0, H) \wedge q \in [0, W) \\
  INT32_MIN, & \text{otherwise}
  \end{cases}


upsampling
~~~~~~~~~~
Upsampling operator performs upsampling to the input data by copying the value in each position serveral times.

*Math Formalization*

- Input :math:`X`, whose shape is :math:`(N, C, H, W)`
- Output :math:`Y`
- Attributes: ``scale``, in range :math:`[1, max\_attr)`.

.. math::
  Y[n, i, h, w] = X[n, i, \left\lfloor {h \over \text{scale}}\right\rfloor, \left\lfloor {w \over \text{scale}}\right\rfloor], \\
  \forall n \in [0, N) \wedge i \in [0, C) \wedge
  h \in [0, H \cdot \text{scale}) \wedge w \in [0, W \cdot \text{scale})

Elemwise Operators
------------------

An elemwise operator performs the elementwise function to input data and the process logic over all kinds of operators are alike.
There might be 1 or 2 input tensors and the logic might be complicated for someone. That's way we don't abstract them but describe each one.

abs
~~~
This operator calculates absolute value of input data.

*Math Formalization*

- Input: :math:`X`, a tensor of :math:`N` dimensions, namely :math:`(n_0, n_1, \cdots, n_{N-1})`.
- Output: :math:`Y`, a tensor whose shape is same as :math:`X`

.. math::

  Y[d_0, d_1, \cdots, d_{N-1}] = \begin{cases}
  x, &  x \geqslant 0  \\
  -x, & x < 0
  \end{cases},\\

  \forall i \in [0, N), d_i \in [0, n_i),\\

  \text{, where }x \text{ denotes } X[d_0, d_1, \cdots, d_{N-1}]

cvm_precision
~~~~~~~~~~~~~

The precision operator gives how many bits the absolute value of a number takes. 1 takes 1 bit. 2, 3 take 2 bits, etc. A special case is that 0 always takes at least 1 bit.

*Math Formalization*

- Input :math:`X`, a tensor of :math:`N` dimensions, namely :math:`(n_0, n_1, \cdots, n_{N-1})`.
- Output :math:`Y`, a tensor whose shape is same as :math:`X`

.. math::

  Y[d_0, d_1, \cdots, d_{N-1}] = \begin{cases}
  \lceil log_2(abs(x)+1) \rceil, & x \neq 0\\
  1, & x = 0
  \end{cases},\\
  
  \forall i \in [0, N), d_i \in [0, n_i),\\

  \text{ where } x \text{ denotes } X[d_0, d_1, \cdots, d_{N-1}]


elemwise_add
~~~~~~~~~~~~
This operator performs elementwise add to the 2 input tensors.

*Math Formalization*

- Input: there are 2 inputs, of the same shape.

  + :math:`A`, a tensor of :math:`N` dimensions, namely :math:`(n_0, n_1, \cdots, n_{N-1})`.
  + :math:`B`, whose shape is same as :math:`A`.

- Output: :math:`Y`, a tensor whose shape is same as :math:`A` and :math:`B`.

.. math::
  Y = A + B

elemwise_sub
~~~~~~~~~~~~
This operator performs elementwise subtraction to the 2 input tensors.

*Math Formalization*

- Input: there are 2 inputs, of the same shape.

  + :math:`A`, a tensor of :math:`N` dimensions, namely :math:`(n_0, n_1, \cdots, n_{N-1})`.
  + :math:`B`, whose shape is same as :math:`A`.

- Output: :math:`Y`, a tensor whose shape is same as :math:`A` and :math:`B`.

.. math::
  Y = A - B

negative
~~~~~~~~
This operator performs elementwise negative to the input tensor.

*Math Formalization*

- Input: :math:`X`, a tensor of :math:`N` dimensions, namely :math:`(n_0, n_1, \cdots, n_{N-1})`.
- Output: :math:`Y`, a tensor whose shape is same as :math:`X`.

.. math::
  Y = -X

clip
~~~~
This operator performs clip, cutting the data into a range, to the input tensor.

*Math Formalization*

- Input: :math:`X`, a tensor of :math:`N` dimensions, namely :math:`(n_0, n_1, \cdots, n_{N-1})`.
- Output: :math:`Y`, a tensor whose shape is same as :math:`X`.
- Attributes:

  + ``a_min``
  + ``a_max``

.. math::
  Y[d_0, d_1, \cdots, d_{N-1}] = \begin{cases}
  \text{a_max}, & x \geqslant \text{a_max} \\
  x, & x \in (\text{a_min}, \text{a_max}) \\
  \text{a_min}, & x \leqslant \text{a_min}
  \end{cases},\\

  \forall i \in [0, N) \wedge d_i \in [0, n_i),

  \text{ where } x \text{ denotes } X[d_0, d_1, \cdots, d_{N-1}]

cvm_cilp
~~~~~~~~
This operator clips the input data into a certain CVM precision.

*Math Formalization*

- Input: :math:`X`, a tensor of :math:`N` dimensions, namely :math:`(n_0, n_1, \cdots, n_{N-1})`.
- Output: :math:`Y`, a tensor whose shape is same as :math:`X`.
- Attribute: ``precision``, an int in range :math:`[1, 33)`

.. math::
  Y = clip(X, \text{a_min}=-\alpha, \text{a_max}=\alpha), \\
  \text{where } \alpha = 2^\text{precision-1}-1


cvm_right_shift
~~~~~~~~~~~~~~~
This operator performs right shift. Slightly different from C right shift, the result of this operator would be rounded to nearest integer. A special case is that negative half number will be rounded up, -1.5 rounded to -1 for example.

*Math Formalization*

- Input: :math:`X`, a tensor of :math:`N` dimensions, namely :math:`(n_0, n_1, \cdots, n_{N-1})`.
- Output: :math:`Y`, a tensor whose shape is same as :math:`X`.
- Attribute:

  + ``precision``, an int in range :math:`[1, 33)`
  + ``shift_bit``, an int in range :math:`[1, 33)`

.. math::
  Y = clip(T, \text{a_min} = -\alpha, \text{a_max}=\alpha), \\
  \text{where } T = {\left\lfloor
  \left(\left\lfloor \frac{X}{2^{\text{shift_bit} - 1}} \right\rfloor + 1 \right)
  \div 2 \right\rfloor} \text{ and } \alpha = 2 ^ {\text{precision} - 1} - 1


cvm_left_shift
~~~~~~~~~~~~~~
This operator performs left shift to the input tensor, same as C left shift operator.

*Math Formalization*

- Input: :math:`X`, a tensor of :math:`N` dimensions, namely :math:`(n_0, n_1, \cdots, n_{N-1})`.
- Output: :math:`Y`, a tensor whose shape is same as :math:`X`.
- Attribute:

  + ``precision``, an int in range :math:`[1, 33)`
  + ``shift_bit``, an int in range :math:`[1, 33)`

.. math::
  Y = clip(T, \text{a_min} = -\alpha, \text{a_max}=\alpha), \\
  \text{where } T = X * 2^\text{shift_bit} \text{ and } \alpha = 2 ^ {\text{precision} - 1} - 1


Transform Operators
-------------------

transform operator do not do the calculation on  the data but simply reshape, copy or select part of it. The process logic over all kinds of operators are quite different.

repeat
~~~~~~
.. _repeat_operator:


This operator repeats the input data by ``repeats`` times along the given ``axis``. Each element is repeated right after itself.

*Math Formalization*

- Input: :math:`X`, a tensor of :math:`N` dimensions, namely :math:`(n_0, n_1, \cdots, n_{\text{axis}}, \cdots, n_{N-1})`
- Output: :math:`Y`, a tensor of :math:`N` dimensions, namely :math:`(n_0, n_1, \cdots, n_{\text{axis}} \cdot repeats, \cdots, n_{N-1})`
- Attribute:

  + ``axis``, an int in range :math:`[0, N)`, indicating on which axis is the repeatation performed.
  + ``repeats``, an int in range :math:`[1, +\infty)`, indicating how many times the data is repeated.

.. math::
  Y[d_0, d_1, \cdots, d_\text{axis}, \cdots, d_{N-1}] =
  X[d_0, d_1, \cdots, \left\lfloor{d_\text{axis} \over \text{repeats}}\right\rfloor, \cdots, d_{N-1}], \\
  \forall d_0 \in [0, n_0) \wedge \cdots \wedge d_{axis-1} \in [0, n_{axis-1}) \wedge
  d_{axis} \in [0, n_{axis} \cdot \text{repeats}) \wedge \\
  d_{axis+1} \in [0, n_{axis+1}) \wedge \cdots \wedge d_{N-1} \in [0, n_{N-1})


tile
~~~~
This operator tiles the input data serveral times on serveral dimensions. Different from :ref:`Repeat <repeat_operator>` operator repeating each element right after itself, this operator tiles the whole data after the whole data.

*Math Formalization*

- Input: :math:`X`, a tensor of :math:`N` dimensions, namely :math:`(n_0, n_1, \cdots, n_{N-1})`,
- Output: :math:`Y`
- Attribute: ``reps``, a tensor of :math:`M` dimensions, namely :math:`(m_0, m_1, \cdots, m_{M-1})`.

.. math::
  r \in [1, max\_attr), \forall r \in \text{reps}

.. math::
  Y[k_0, \cdots, k_{K-N-1}, k_{K-N}, k_{K-N+1}, \cdots, k_{K-1}] = \\
  X[k_{K-N+0} \text{ mod } n_0, k_{K-N+1} \text{ mod } n_1, \cdots, k_{K-N+N-1} \text{ mod } n_{N-1}], \\
  \forall k_0 \in [0, S_0) \wedge \cdots \wedge k_{K-1} \in [0, S_{K-1}), \\

  \text{where } K = \max\{M, N\} \text{ and } S_i = SX_i \cdot SR_i \text{ and } \\
  SX_p = \begin{cases}
  n_{p-K+N}, & p \in [K-N, K-1) \\
  1, & p \in [0, K-N)
  \end{cases} \text{ and } \\
  SR_q = \begin{cases}
  m_{q-K+M}, & q \in [K-M, K-1) \\
  1, & q \in [0, K-M)
  \end{cases}


flatten
~~~~~~~

This operator flattens the input tensor data to an array in a row-major order.

*Math Formalization*

- Input: :math:`X`, a tensor of :math:`N` dimensions, namely :math:`(n_0, n_1, \cdots, n_{N-1})`.
- Output: :math:`Y`.

.. math::
  Y[\text{flatten_index}(d_0, d_1, \cdots, d_{N-1}, n_0, n_1, \cdots, n_{N-1})]  =  \\
  X[d_0, d_1, \cdots, d_{N-1}], \\
  \forall d_0 \in [0, n_0) \wedge d_1 \in [0, n_1) \wedge \cdots \wedge
  d_{N-1} \in [0, n_{N-1})

where :math:`\text{flatten_index}` is

.. math::
  \text{flatten_index}(d_0, d_1, \cdots, d_{N-1}, n_0, n_1, \cdots, n_{N-1}) = \\
  d_0 \cdot \prod_{i = 1}^{N-1} n_i +
  d_1 \cdot \prod_{i = 2}^{N-1} n_i +
  \cdots + d_{N-2} * n_{N-1} + d_{N-1}

concatenate
~~~~~~~~~~~

This operator concatenates tensors along a given axis.

*Math Formalization*

- Inputs: :math:`M` tensors, namely :math:`I^0, I^1, \cdots, I^{M-1}`. They all have :math:`N` dimensions, namely :math:`I^i`'s shape is :math:`(n^i_0, n^i_1, \cdots, n^i_{N-1})`. All dimensions except the axis to be concatenated along must have the same length.
- Output: :math:`Y`
- Attribute: ``axis``, an int in range :math:`[0, N)`.

.. math::
  n^i_j = n^0_j, \forall i \in [1, M) \wedge j \in [0, N) \wedge j \neq \text{axis}

.. math::
  Y[d_0, d_1, \cdots, d_\text{axis-1}, \text{new_idx}, d_\text{axis+1}, \cdots, d_{N-1}] = I^i[d_0, d_1, \cdots, d_{N-1}], \\
  \forall d_0 \in [0, n^i_0) \wedge \cdots \wedge d_{N-1} \in [0, n^i_{N-1})
  \wedge i \in [0, M), \\

  \text{where new_idx} = \sum_{j=0}^{i-1} n^j_\text{axis} + d_\text{axis}

transpose
~~~~~~~~~

This operator transposes the input data with a certain sequence.

*Math Formalization*

- Input: :math:`X`, a tensor of :math:`N` dimensions, namely :math:`(n_0, n_1, \cdots, n_{N-1})`
- Output: :math:`Y`,
- Attribute: ``axes``, a TShape of length :math:`M \in \{0, N\}`, which means ``axes`` is either empty or a permutation of :math:`\{0, 1, \cdots, N-1\}`

.. math::
  \text{axis} \in [-N, N), \forall \text{axis} \in \text{axes}

.. math::
  Y[d_{\text{real_axes}_0}, d_{\text{real_axes}_1}, \cdots, d_{\text{real_axes}_{N-1}}] =
  X[d_0, d_1, \cdots, d_{N-1}], \\

  \forall d_0 \in [0, n_0) \wedge \cdots \wedge d_{N-1} \in [0, n_{N-1}), \\

  \text{where real_axes}_i = \begin{cases}
  \text{axes}_i, & M = N \wedge \text{axes}_i \geqslant 0 \\
  \text{axes}_i + N, & M = N \wedge \text{axes}_i < 0 \\
  N-1-i, & M = 0
  \end{cases} \text{ and } \\
  card \; \{\text{real_axes}_i \mid i \in [0, N) \} = N

slice
~~~~~

This operator slices an input array with given attribute. For each dimension, strides (default to be 1), start (default to be 0) and end (default to be length of this dimension) coordinates can be assigned by user.

*Math Formalization*

- Input: :math:`X`, a tensor of :math:`N` dimensions, namely :math:`(n_0, n_1, \cdots, n_{N-1})`
- Output: :math:`Y`,
- Attributes:

  + ``begin``, :math:`B` dimensions.
  + ``end``, :math:`E` dimensions.
  + ``strides``: :math:`S` dimensions.

  :math:`B`, :math:`E`, :math:`S` can be different.

.. math::
  \text{b_arr}[b] = \begin{cases}
  \text{begin}[b] + n_i, & b \in [0, N) \wedge b < B \wedge begin[b] < 0 \\
  \text{begin}[b], & b \in [0, N) \wedge b < B \wedge begin[b] \geqslant 0 \\
  0, & b \in [0, N) \wedge b \geqslant B
  \end{cases}, b \in [0, N) \\

  \text{e_arr}[e] = \begin{cases}
  \text{end}[e] + n_i, & e \in [0, N) \wedge e < E \wedge end[e] < 0\\
  \text{end}[e], & e \in [0, N) \wedge e < E \wedge end[e] \geqslant 0\\
  n_{e}, & e \in [0, N) \wedge e \geqslant E
  \end{cases}, e \in [0, N) \\

  \text{s_arr}[s] = \begin{cases}
  \text{stride}[s], & s \in [0, N) \wedge s < S \\
  1, & s \in [0, N) \wedge s \geqslant S
  \end{cases}, s \in [0, N) \\

  \forall \{i \mid i \in [0, N)\}: \text{s_arr}[i] \ne 0 \\
  \text{b_range}(i) = \begin{cases}
  -1, & \text{s_arr}[i] < 0 \\
  0, & \text{s_arr}[i] \geqslant 0
  \end{cases} \\
  \text{e_range}(i) = \begin{cases}
  n_i - 1, & \text{s_arr}[i] < 0 \\
  n_i, & \text{s_arr}[i] \geqslant 0
  \end{cases} \\

  \text{b_vec}[b] =
  clip(\text{b_arr}[b], \text{a_min}=\text{b_range}(b), \text{a_max}=\text{e_range}(b)-1), b \in [0, N) \\
  \text{e_vec}[e] =
  clip(\text{e_arr}[e], \text{a_min}=\text{b_range}(e), \text{a_max}=\text{e_range}(e)-1), e \in [0, N) \\
  \forall \{i \mid i \in [0, N) \}:
  \begin{cases}
  \text{b_vec}[i] < \text{e_vec}[i], & \text{s_arr}[i] > 0 \\
  \text{e_vec}[i] < \text{b_vec}[i], & \text{s_arr}[i] < 0
  \end{cases} \\

  Y[d_0, d_1, \cdots, d_{N-1}] = \\
  X[\text{b_vec}[0] + \text{s_arr}[0] * d_0,
  \text{b_vec}[1] + \text{s_arr}[1] * d_1,
  \cdots, \text{b_vec}[N-1] + \text{s_arr}[N-1] * d_{N-1}] \\

  \forall d_j \in [0, \left\lceil{\text{e_vec}[j] - \text{b_vec}[j] \over \text{s_arr}[j]}\right\rceil),

  \text{where } j \in [0, N)


slice_like
~~~~~~~~~~

This operator slices the input :math:`X` to a shape that looks like the other given input ``shape_like``.

*Math Formalization*

- Input: there are 2 inputs
  
  + :math:`X`, a tensor of :math:`N` dimensions, namely :math:`(n_0, n_1, \cdots, n_{N-1})`
  + :math:`\text{shape_like}`, a tensor of :math:`M` dimensions, namely :math:`(m_0, m_1, \cdots, m_{M- 1 })`,

- Output: :math:`Y`
- Attribute: ``axes``, a ``TShape`` with length :math:`K`. If ``axes`` is not empty, only those axes mentioned will be sliced on and others in ``shape_like`` will also be ignored. :math:`M \ne N` is allowed only if non-empty ``axes`` are given. If :math:`M>N`, those dimensions higher than :math:`N` will be ignored and if :math:`M<N`, only the first :math:`M` dimensions are sliced while those dimensions higher than :math:`M` will stay the same.

.. math::
  \text{sliced_axes} = \begin{cases}
  \{j \mid j \in axes \wedge j \geqslant 0\} \bigcup
  \{j + N \mid j \in axes \wedge j < 0\}, & K > 0\\
  \{0, 1, \cdots, M-1\}, & K = 0
  \end{cases}, \\

  \text{where } \forall j \in \text{sliced_axes}: j < \min(M, N) \text{ and } m_j \leqslant n_j\\

.. math::
  Y[d_0, d_1, \cdots, d_{N-1}] = X[d_0, d_1, \cdots, d_{N-1}], \\
  
  \forall d_j \in \begin{cases}
  [0, m_j), & j \in \text{sliced_axes} \\
  [0, n_j), & j \notin \text{sliced_axes}
  \end{cases},\\

  \text{where } j \in [0, N)

take
~~~~

This operator takes some elements from the input data. If axis is not given, it flattens the input data and takes elements; if axis is given, takes the input data on this axis for every coordinates in other axes.

*Math Formalization*

- Input: there 2 inputs

  + :math:`X`, a tensor of :math:`N` dimensions, namely :math:`(n_0, n_1, \cdots, n_{N-1})`
  + :math:`indices`, a tensor of :math:`M` dimensions, namely :math:`(m_0, m_1, \cdots, m_{M- 1})`

- Output: :math:`Y`,
- Attribute: ``axis`` an ``Optional<int>``.

1. Case axis is ``None`` :

.. math::
  T = flatten(X) \\
  Y[d_0, d_1, \cdots, d_{M-1}] = T[clip(\text{xidx}, \text{a_min}=0, \text{a_max}=|T|-1)],\\

  \forall (d_0, d_1, \cdots, d_{M-1}), \\

  \text{ where } d_j \in [0, m_j) \wedge j \in [0, M) \text{ and }\\
  \text{xidx} = \text{indices}[d_0, d_1, \cdots, d_{M-1}]

2. Case axis is ``int``:

.. math::
  \text{axis} \in [-N, N) \\
  \text{real_axis} = \begin{cases}
  \text{axis}, & \text{axis} \geqslant 0 \\
  \text{axis} + N, & \text{axis} < 0
  \end{cases} \\
  Y[d_0, d_1, \cdots, d_{M+N-2}] = X[d_0, \cdots, d_{\text{real_axis}-1}, \text{xdix}, d_{\text{real_axis}+M}, \cdots, d_{M+N-2}], \\


  \forall d_j \in \begin{cases}
  [0, n_j), & j < \text{real_axis} \\
  [0, m_{j-\text{real_axis}}), & j \in [\text{real_axis}, \text{real_axis}+M) \\
  [0, n_{j-M+1}), & j \in [\text{real_axis} + M, M+N-1)
  \end{cases},\\
  \text{where } \text{xidx}{} = clip(\text{indices}[d_{\text{real_axis}}, d_{\text{real_axis}+1}, \cdots, d_{\text{real_axis}+M-1}],\\
  \text{a_min}=0, \text{a_max}=n_{\text{real_axis}}-1)


cvm_lut
~~~~~~~

This operator is a alias for a take where axis is None.

*Math Formalization*

- Input: there are 2 inputs.

  + :math:`X`, a tensor of :math:`N` dimensions, namely :math:`(n_0, n_1, \cdots, n_{N-1})`
  + :math:`indices`, a tensor of :math:`M` dimensions, namely :math:`(m_0, m_1, \cdots, m_{M- 1})`

- Output: :math:`Y`.


.. math::
  Y=take(X, \text{indices}, \text{axis}=\text{None})


expand_dims
~~~~~~~~~~~

This operator expands some new dimensions right from the given axis.

*Math Formalization*

- Input: :math:`X`, a tensor of :math:`N` dimensions, namely :math:`(n_0, n_1, \cdots, n_{N-1})`,
- Output: :math:`Y`
- Attributes:

  + ``axis``, an int in range :math:`[-N-1, N+1)`, indicating where the new dimensions starts. Note that :math:`N+1`, instead of :math:`N`, will be added to negative axis.
  + ``num_newaxis``, an int in range :math:`[min\_attr, max\_attr)`

.. math::
  Y[d_0,d_1, \cdots, d_{\text{real_axis}-1},
  \underbrace{0, 0, \cdots, 0}_{\text{num_newaxis}},
  d_\text{real_axis}, \cdots, d_{N-1}] = X[d_0, d_1, \cdots, d_{N-1}], \\

  \forall d_0 \in [0, n_0) \wedge \cdots \wedge d_{N-1} \in [0, n_{N-1}), \\

  \text{where real_axis} =
  \begin{cases}
  \text{axis},& \text{axis} \geqslant 0 \\
  \text{axis} + N,& \text{axis} < 0
  \end{cases}


reshape
~~~~~~~

This operator reshapes the input data.

*Math Formalization*

- Input: :math:`X`, a tensor of :math:`N` dimensions, namely :math:`(n_0, n_1, \cdots, n_{N-1})`,
- Output: :math:`Y`,
- Attribute: ``target_shape``, a ``TShape`` of length ``M``,  namely :math:`(m_0, m_1, \cdots,  m_{M-1})` , s.t. :math:`m_0 * m_1 * \cdots * m_{M-1} = n_0 * n_1 * \cdots * n_{N-1}`.

.. math::
  Y[d_0, d_1, \cdots, d_{M-1}] = T[\text{flatten_index}(d_0, d_1, \cdots, d_{M-1}, m_0, m_1, \cdots, m_{N-1})], \\

  \forall d_0 \in [0, m_0) \wedge \cdots \wedge d_{N-1} \in [0, m_{N-1}), \\

  \text{where } T = \text{flatten}(X)


squeeze
~~~~~~~

This operator removes dimensions of length 1.

*Math Formalization*

- Input: :math:`X`, a tensor of :math:`N` dimensions, namely :math:`(n_0, n_1, \cdots, n_{N-1})`
- Output: :math:`Y`.
- Attribute: ``axes``, a ``TShape`` of length M.

.. math::
  \text{axis} \in [-N, N), \forall \text{axis} \in \text{axes}

.. math::
  \text{real_axes} =
  \begin{cases}
  \{\text{axis} \mid \text{axis} \geqslant 0 \wedge \text{axis} \in \text{axes} \} \bigcup
  \{\text{axis} + N \mid \text{axis} < 0 \wedge \text{axis} \in \text{axis}\},
  & M > 0 \\
  \{\text{axis} \mid n_\text{axis} = 1 \wedge \text{axis} \in [0, N) \}, & M = 0
  \end{cases} \\

.. math::
  n_\text{axis} = 1, \forall \text{axis} \in \text{real_axis}

.. math::
  Y[d_{I(0)}, d_{I(1)}, \cdots, d_{I(N-K-1)}] = X[d_0, d_1, \cdots, d_{N-1}], \\

  \forall d_0 \in [0, n_0) \wedge d_1 \in [0, n_1)
  \wedge \cdots \wedge d_{N-1} \in [0, n_{N-1}), \\

  \text{where } K = card \; \text{\{real_axes\}} \text{ and } \\

  I: \{i \mid i \in [0, N-K) \} \to
  \{i \mid i \in [0, N) \wedge i \notin \text{real_axes} \}, \\
  \text{s.t. } I(i) < I(j), \forall 0 \leqslant i < j < N-K


Vision Operators
----------------

We provide 2 operators for vision scenario. Since the scenario is narrow, regulation of the input data is stricter than other operators. If there's no other specification, the input data should be 3D, namely :math:`(B, N, K)`, indicating number of batches, number of result for each batch and the length (:math:`K`) of a result. The first value should be ID and the second should be the score.

get_valid_count
~~~~~~~~~~~~~~~

This operator counts how many results are more than a threshold and what are they.

*Math Formalization*

- Input: :math:`X`, a 3D tensor of shape :math:`(B, N, K), 2 \leqslant K \leqslant 32`
- Output:

  + :math:`\text{valid_count}`,
  + :math:`Y`,

- Attribute: ``score_threshold``, an ``int``.


.. math::
  \text{valid_count}[b] = card\{ q \mid q \in [0, N) \wedge
  X[b, q, 1] > \text{score_threshold} \}, \\
  \quad \forall b \in [0, B)

.. math::
  Y[b, \text{idx}, k] = X[b, n, k], \\
  \quad \forall b \in [0, B) \wedge n \in [0, N) \wedge
  k \in [0, K) \wedge X[b, n, 1] > \text{score_threshold}, \\

  \quad \text{where idx = }
  card \{q \mid q \in [0, n) \wedge
  X[b, q, 1] > \text{score_threshold} \}

.. math::
  Y[b,n, k] = -1, \forall b \in [0, B) \wedge
  n \in [\text{valid_count}[b], N) \wedge k \in [0, K)


non_max_suppression
~~~~~~~~~~~~~~~~~~~

This operator implements the nms algorithm, finding valid bounding boxes.

*Math Formalization*

- Input: there are 2 inputs.

  + :math:`X`, a 3D tensor of shape :math:`(B, N, K), K = 6`
  + ``valid_count``, a 1D tensor of length :math:`B`

- Output: :math:`Y`
- Attributes:

  + ``iou_threshold``, an ``int``, representing the percentage of intersection over union with value in range :math:`(0, +\infty)` where 101 stands for bounding box full-overlap specifically and larger integer is equivalent to that.
  + ``max_output_size``, an ``int``
  + ``force_suppress``, a ``boolean``
  + ``top_k``, an ``int``.

.. math::
  R[b, i, k] = X[b, I(i), k], \\

  \forall b \in [0, B) \wedge i \in [0, T) \wedge k \in [0, K), \\

  \text{where } T = \text{max}\{
  \text{min}(N, \text{valid_count}[b]), 0\} \text{ and } \\
  I: \{ i \mid i \in [0, T) \} \to \{ i \mid i \in [0, T) \}, \\
  \text {s.t. } X[b, I(i), 1] > X[b, I(j), 1] \vee \\
  (X[b, I(i), 1] = X[b, I(j), 1] \wedge I(i) < I(j)),
  \forall 0 \leqslant i < j < T

.. math::
  Y[b, n, k] = R[b, \text{IDX}(n), k], \\

  \forall b \in [0, B) \wedge n \in [0, \min\{T, \text{MOS}, card\{U\}\}) \wedge
  k \in [0, K), \\

  \text{where } \text{TK} = \begin{cases}
    +\infty, & \text{if top_k < 0} \\[1ex]
    \text{top_k}, & \text{otherwise}
  \end{cases} \text{ and } \\
  \text{MOS} =
  \begin{cases}
    +\infty, & \text{if max_output_size < 0} \\[1ex]
    \text{max_output_size}, & \text{otherwise}
  \end{cases} \text{ and } \\
  \text{iou}(p, q) = \begin{cases}
    \text{overlap_ratio}(R[b, p, :], R[b, q, :]), &
    \begin{array}{}
    \text{force_suppress is true}\\
    \vee R[b, p, 0] = R[b, q, 0]
    \end{array} \\[1ex]
    0, & \text{otherwise}
  \end{cases} \\
  \text{ and } \\
  \text{U} = \{p \mid p \in [0, min\{TK, T\}) \wedge
  R[b,p,0] >= 0 \wedge \text{iou}(p, q) < \text{iou_threshold}, \\
  \forall q \in U \wedge q < p\}
   \text{ and } \\
  \text{IDX}: \{i \mid i \in [0, card\{U\})\} \to U, \text{s.t. }
  \text{IDX}(i) < \text{IDX}(j), \forall  i < j

.. math::
  Y[b, n, k] = -1, \\
  \forall b \in [0, B) \wedge k \in [0, K) \wedge
  n \in [min\{T, \text{MOS},card\{U\}\}, N)

:math:`\text{overlap_ratio}` calculates the overlap ratio of two rectangles: area of their intersection over the area of their union.
