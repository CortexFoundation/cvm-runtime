
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
----------------

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
  + :math:`X`, input data to be calculated whose shape is :math:`(N, C, H, W)`
  + :math:`W`, convolution kernel weight whose shape is :math:`(OC, IC, KH, KW)`, :math:`C = IC \cdot \text{groups} \wedge OC \text{ mod } \text{groups} = 0`
  + :math:`B`, bias, of type `Optional<DLTensor>`. If `B` is not None, it's shape is :math:`(\text{OC},)`.
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
  0, & \text{otherwise}
  \end{cases}


dense
~~~~~
Dense operator provides a full connected layer.
* Math Formalization*

- Input: there 2 or 3 inputs.
  + :math:`X`, a matrix of shape :math:`(M, K)`
  + :math:`W`, a matrix of shape :math:`(N, K)`
  + :math:`B`, bias, of type `Optional<DLTensor>`, If `B` is not `NONE`, it's shape is :math:`(N,)`.
- Output: :math:`Y`, a matrix of shape :math:`(M, N)`

.. math::

  Y=X W^T + \begin{cases}
  0, & \text{if B is None} \\
  B, & \text{otherwise}
  \end{cases}

relu
~~~~

Relu performs elementwise rectified linear unit function.
* Math Formalization*

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
  + `pool_size`, a `TShape` of length 2, namely :math:`(PSH, PSW)`
  + `padding`, either a `TShape` of length 2, namely :math:`(PH, PW) \in [min\_attr, max\_attr)`, or an int indicating :math:`PH=PW`
  + `strides`, a `TShape` of length 2, namely :math:`(SH, SW)`
  + `ceil_mode`, `boolean`.

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
  0, & \text{otherwise}
  \end{cases}


upsampling
~~~~~~~~~~
Upsampling operator performs upsampling to the input data by copying the value in each position serveral times.

*Math Formalization*

- Input :math:`X`, whose shape is :math:`(N, C, H, W)`
- Output :math:`Y`
- Attributes: `scale`, in range :math:`[1, max\_attr)`.

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
  \end{cases},
  \forall i \in [0, N), d_i \in [0, n_i) where x denotes X[d_0, d_1, \cdots, d_{N-1}]

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
  \end{cases},
  \forall i \in [0, N), d_i \in [0, n_i) where x denotes X[d_0, d_1, \cdots, d_{N-1}]


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
  + `a_min`
  + `a_max`

.. math::
  Y[d_0, d_1, \cdots, d_{N-1}] = \begin{cases}
  \text{a_max}, & x \geqslant \text{a_max} \\
  x, & x \in (\text{a_min}, \text{a_max}) \\
  \text{a_min}, & x \leqslant \text{a_min}
  \end{cases},
  \forall i \in [0, N), d_i \in [0, n_i) where x denotes X[d_0, d_1, \cdots, d_{N-1}]

cvm_cilp
~~~~~~~~
This operator clips the input data into a certain CVM precision.

* Math Formalization*

- Input: :math:`X`, a tensor of :math:`N` dimensions, namely :math:`(n_0, n_1, \cdots, n_{N-1})`.
- Output: :math:`Y`, a tensor whose shape is same as :math:`X`.
- Attribute: `precision`, an int in range :math:`[1, 33)`

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
  + `precision`, an int in range :math:`[1, 33)`
  + `shift_bit`, an int in range :math:`[1, 33)`

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
  + `precision`, an int in range :math:`[1, 33)`
  + `shift_bit`, an int in range :math:`[1, 33)`

.. math::
  Y = clip(T, \text{a_min} = -\alpha, \text{a_max}=\alpha), \\
  \text{where } T = X * 2^\text{shift_bit} \text{ and } \alpha = 2 ^ {\text{precision} - 1} - 1


Transform Operators
-------------------

A transform operator do not do the calculation on the data but simply reshape, copy or select part of it. The process logic over all kinds of operators are quite different.

repeat
~~~~~~
This operator repeats the input data by `repeats` times along the given `axis`. Each element is repeated right after itself.


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








