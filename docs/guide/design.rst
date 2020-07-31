
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

Reduce Operator
---------------

.. math::

  T = \left\{i \mid \text{axis} \in \text{axes} \wedge
  i = \begin{cases}
  \text{axis}, & \text{if axis } \geqslant 0 \\
  \text{axis} + N, & \text{otherwise}
  \end{cases} \right\}, \\
  \text{where } card \; \text{T} = M \text{ and }
  j \in [0, N), \forall j \in \text{T}
