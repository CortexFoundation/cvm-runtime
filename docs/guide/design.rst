
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

And developer only need to assert conditions with pre-defined
macros, such as :c:func:`VERIFY` and :c:func:`CHECK`. A example
usage likes:
  
.. code-block:: C

  VERIFY(condition) << "error information";
  CHECK(condition) << "error information";

Now it's important to understand the difference between the two
exceptions. One should know the cvm-runtime project is intergral
to the CVM in Cortex Foundation's full-node: CortexTheasus. A
inference call in the cortex blockchain will cost the Endophin,
A calculation unit for model inference takes up, including
memory, time-spending, etc. And then the logic error will still
consume the invoker's CTXC token according to the Endophin cost 
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

Reduce Operator
---------------

.. toctree::
  :maxdepth: 2

  formalization

.. math::

  T = \left\{i \mid \text{axis} \in \text{axes} \wedge
  i = \begin{cases}
  \text{axis}, & \text{if axis } \geqslant 0 \\
  \text{axis} + N, & \text{otherwise}
  \end{cases} \right\}, \\
  \text{where } card \; \text{T} = M \text{ and }
  j \in [0, N), \forall j \in \text{T}
