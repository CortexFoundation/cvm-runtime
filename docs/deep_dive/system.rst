
.. _system-design:

*******************
System Architecture
*******************

.. contents::


Integer Inference
-----------------

All the inference data flow is pure integer in `Graph Runtime`_.
As the `Graph Runtime`_ points out, we will alloc the necessary
memory space and prepare packed functions in order before
real inference. In the memory space allocment, we invoke the
`NDArray Module`_ API to create and hold an memory manager(that
is the ``NDArray`` instance or the internal ``DLTensor*`` pointer) 
with reference count.

The inference have the two basic design idea:

  - Parameters stored in disk must be ``INT`` type.
  - Memory space that inference need must be ``INT32`` type.

First, we will assert all the parameters loaded, data input and
pre-alloced memory that operators need, are the ``INT`` type.

Here are some reference code:

.. code-block::

  auto dtype = data_in->dtype;
  VERIFY((dtype.code == kDLInt) &&
         ((dtype.bits == 32) || (dtype.bits == 8)) &&
         (dtype.lanes == 1))
    << "cvm runtime only supported INT8 or INT32 NDArray, but ("
    << dtype.code << ", " << dtype.bits << ", "
    << dtype.lanes << ")";

Moreover, the operators' forward funtions assume the input data's
type as the ``INT32`` type and then do specific process logic
corresponding with the :ref:`operator-math-formalization`
definition. So there are some places to convert the ``INT`` type
to ``INT32`` type, here are some reference code:

.. code-block:: C
  
  NDArray nd_in(reinterpret_cast<NDArray::Container*>(data_in));
  if (data_in->dtype.bits == 8) {
    NDArray nd32 = NDArray::Empty(
        std::vector<int64_t>(dshp, dshp+ndim),
        DLDataType{.code=kDLInt, .bits=32, .lanes=1},
        ctx);
    int8_t *data8 = static_cast<int8_t*>(data_in->data);
    int32_t *data32 = static_cast<int32_t*>(nd32->data);
    int64_t num_elems = 1;
    for (int i = 0; i < data_in->ndim; ++i)
      num_elems *= data_in->shape[i];
    for (int i = 0; i < num_elems; i++)
      data32[i] = static_cast<int32_t>(data8[i]);

    nd_in.swap(nd32);
  }

More detail source code refers to the ``SetInput`` function
in file: ``src/runtime/graph_runtime.cc`` please.

And the operators' forward funtions get the neccessary
``DLTensor*`` pointer to convert to ``INT32`` type pointer
based on the assumption 
that verify above when processing the mathmatical logic. 


Assertion and Log
-----------------

- source code: ``include/utils/logging.h``.

The assertion do runtime checks and may generate two exceptions,
which respectively stands for runtime error inherited from
``std::runtime_error`` and logic error inherited from 
``std::logic_error``.
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

NDArray Module
--------------

Graph Runtime
-------------

Precision Schema
----------------

Shape Inference
---------------

Operator Gas Table
------------------

.. toctree::
  :maxdepth: 2

  OPs Table <../cvm/ops.md>

