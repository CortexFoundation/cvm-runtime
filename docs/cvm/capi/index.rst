.. _c-backend-api-label:

C Backend API
=============

This is a reference to :cpp:class:`utils::ValueVerifyFatal`.
:c:func:`CVMAPILoadModel`.

.. doxygenfunction:: CVMAPILoadModel
.. doxygenfunction:: CVMAPIFreeModel
.. doxygenfunction:: CVMAPIInference

.. doxygenfunction:: CVMAPIGetInputLength
.. doxygenfunction:: CVMAPIGetInputTypeSize
.. doxygenfunction:: CVMAPIGetOutputLength
.. doxygenfunction:: CVMAPIGetOutputTypeSize

.. doxygenfunction:: CVMArrayAlloc
.. doxygenfunction:: CVMArrayFree

.. doxygenstruct:: cvm::runtime::CVMModel
  :members:
  :undoc-members:

.. doxygenclass:: cvm::runtime::CvmRuntime
  :members:

.. doxygenfunction:: cvm::runtime::CvmRuntime::Setup
.. doxygenfunction:: cvm::runtime::CvmRuntime::Run

.. doxygenclass:: cvm::Op
  :members:

.. doxygendefine:: CVM_REGISTER_OP

.. doxygenclass:: utils::LogMessageFatal
  :members:

.. doxygenclass:: utils::ValueVerifyFatal
  :members:

.. doxygendefine:: VERIFY
.. doxygendefine:: VERIFY_LT
.. doxygendefine:: VERIFY_GT
.. doxygendefine:: VERIFY_LE
.. doxygendefine:: VERIFY_GE
.. doxygendefine:: VERIFY_EQ
.. doxygendefine:: VERIFY_NE
.. doxygendefine:: VERIFY_NOTNULL

.. doxygendefine:: CHECK
.. doxygendefine:: CHECK_LT
.. doxygendefine:: CHECK_GT
.. doxygendefine:: CHECK_LE
.. doxygendefine:: CHECK_GE
.. doxygendefine:: CHECK_EQ
.. doxygendefine:: CHECK_NE
.. doxygendefine:: CHECK_NOTNULL
