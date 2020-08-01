
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

This will give a full exhaustive explanation to CVM operators,
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

Example
^^^^^^^

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











