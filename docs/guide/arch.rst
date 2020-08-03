
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


Operator Math Formalization
===========================

.. note::

  Write this section document refer to the doc:
  :ref:`Math Format <write_math_formalization>` please.

This doc gives a full exhaustive explanation to CVM operators
which are defined with the macro function `CVM_REGISTER_OP`.
The formalization version source code has a strong correlation
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

.. contents::
  :local:


Reduce Operators
---------------

A reduce operator performs the reduction function to input data based on the parameters, and the process logic over all kind of operators are the same.
Reduction is performed on the given axes, other dimensions remains the same and the result are stored in those places.
We abstract the common reduce logic as formalization here and specify the reduce function
for each operators respectively.

- Input: :math:`X`, a tensor of :math:`N` dimensions, namely :math:`(n_0, n_1, \cdots, n_{N-1})`
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
    U - T, & \text{if 5exclude is true} \\
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
-------------------
A broadcast operator performs the broadcast function to input data, and the process logic over all kinds of operators are the same. 

- Input: There are 2 inputs.
 + :math:`A`, a tensor of :math:`M` dimensions, namely :math:`(m_0, m_1, \cdots, m_{M-1})`
 + :math:`B`, a tensor of :math:`N` dimensions, namely :math:`(n_0, n_1, \cdots, n_{N-1})`
- Output: :math:`Y`, a tensor with :math:`max(M, N)` dimensions, the higher dimension of the two inputs, and it's shape is identical to the input with higher dimension.

The lower :math:`min(M, N)` dimensions of the two inputs must be the same and the remaining higher dimensions of the input of lower dimension is expanded to the higher dimension with 1.

Then the elementwise opertaion is performed to the inputs with broadcast.

We abstract the formalization here and introduce the details as below:

.. math::

  Y[d_0, d_1, \cdots, d_{K-1}] = \begin{cases}
   & A[d_{N-M}, d_1, \cdots, d_{M-1}] OP B[d_0, d_1, \cdots, d_{N-1}], M \leq N\\
   & A[d_0, d_1, \cdots, d_{M-1}] OP B[d_{M-N}, d_1, \cdots, d_{K-1}], M > N
   \end{cases}
.. math::

  \forall i \in [0, K), \text{where } K = max(M, N), d_i \in [0, n_i) if N \geq M or d_i \in [0, m_i) \text{otherwise}


broadcast_add
~~~~~~~~~~~~~
set :math:`\text{BROADCAST_OP}` to :math:`\text{add}`.
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
set :math:`\text{BROADCAST_OP}` to :math:`\text{sub}`.
Note that there's no need to make sure that the dimension of the minuend :math:`A` is higher than subtractor :math:`B`

broadcast_mul
~~~~~~~~~~~~~
set :math:`\text{BROADCAST_OP}` to :math:`\text{mutiply}`.

broadcast_div
~~~~~~~~~~~~~
set :math:`\text{BROADCAST_OP}` to :math:`\text{divide}`.

broadcast_max
~~~~~~~~~~~~~
set :math:`\text{BROADCAST_OP}` to :math:`\text{max}`.

NN Operators
------------
We provide NN operators for users. Unlike reduce operators or broadcast operators, the logic of each operators are different but usage scenario may be the same. In this way, we discribe them together.

conv2d
~~~~~~
We only supported 2-D convolution operator. Also alias *Group-wise Convolution*.

*Math Formalization*

- Input: there are 2 or 3 inputs.
  + `X`, input data to be calculated whose shape is :math:`(N, C, H, W)`
  + `W`, convolution kernel weight whose shape is :math:`(OC, IC, KH, KW)`, :math:`C = IC \cdot \text{groups} \wedge OC \text{ mod } \text{groups} = 0`
  + `B`, bias, of type `Optional<DLTensor>`. If `B` is not None, it's shape is :math:`(\text{OC},)`.
- Output: :math:`Y`
- Attributes:
  + `padding`, a `TShape` of length 2, namely :math:`(PH, PW), PH,PW \in [min\_attr, max\_attr)`, indicating padding size.
  + `stride`, a `TShape` of length 2, namely :math:`(SH, SW) \in [1, max\_attr)`, indicating strides.
  + `dilation`, a `TShape` of length 2, namely :math:`(DH, DW) \in [1, max\_attr)`, parameter used in dilation convolution.
  + `groups`, an `int` in :math:`\text{range} [1, C]`, indicating group number.

.. math::

  OC = \text{groups} * OPG, \text{where } OPG \in \mathbb N^+ \\
  C = IC * \text{groups}

.. math::

  Y[n,oc,p,q]= \sum_{ic = 0}^{IC-1} \text{kernel}(n,(oc \div OPG) * IC + ic, p, q, oc,ic) + \begin{cases}
  0, & \text{if B is None}\\
  B[oc], & \text{otherwise}
  \end{cases}, \\
  \forall n \in [0, N) \wedge oc\in [0, OC) \wedge\\
  p \in \left[0, \left\lfloor{H+2 \cdot \text{PH}-\text{DH} \cdot (\text{KH}-1)-1\over\text{SH}}\right\rfloor+1 \right) \wedge \\
  q \in \left[0, \left\lfloor{W+2 \cdot \text{PW}-\text{DW} \cdot (\text{KW}-1)-1 \over \text{SW}}\right\rfloor+1 \right)

where :math:`\text{kernel}` function does the 2D image convolution calculation, and the formulation is

.. math::

  \text{kernel}(n, j, p, q, o, i) = \sum_{k_i=0}^{\text{KH}} \sum_{k_j = 0}^{\text{KW}} \text{pad}(p'+k_i*\text{DH},q'+k_j*\text{DW}) \cdot W[o, i, k_i, k_j], \\
  \text{where } p' = p \cdot \text{SH} -\text{PH} \text{ and }
  q' = q \cdot \text{SW}-\text{PW} \text{ and } \\
  \text{pad}(p, q) = \begin{cases}
  X[n, j, p, q], & \text{ if } p \in [0, H) \wedge q \in [0, W) \\
  0, & \text{otherwise}a
  \end{cases}


dense
~~~~~
Dense operator provides a full connected layer.
* Math Formalization*

- Input: there 2 or 3 inputs.
  + `X`, a matrix of shape :math:`(M, K)`
  + `W`, a matrix of shape :math:`(N, K)`
  + `B`, bias, of type `Optional<DLTensor>`, If `B` is not `NONE`, it's shape is :math:`(N,)`.
- Output: `Y`


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








