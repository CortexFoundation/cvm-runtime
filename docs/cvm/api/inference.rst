.. _cvm_inference_api:

*******************
Model Inference API
*******************

cvm.runtime
-----------
.. automodule:: cvm.runtime

.. autofunction:: cvm.runtime.CVMAPILoadModel
.. autofunction:: cvm.runtime.CVMAPIFreeModel
.. autofunction:: cvm.runtime.CVMAPIInference

.. autofunction:: cvm.runtime.CVMAPIGetInputLength
.. autofunction:: cvm.runtime.CVMAPIGetInputTypeSize
.. autofunction:: cvm.runtime.CVMAPIGetOutputLength
.. autofunction:: cvm.runtime.CVMAPIGetOutputTypeSize

cvm.ndarray
-----------
.. automodule:: cvm.ndarray

.. autoclass:: cvm.ndarray.NDArray
    :members:
    :inherited-members:

.. autofunction:: cvm.ndarray.array
.. autofunction:: cvm.ndarray.empty
.. autofunction:: cvm.ndarray.save_param_dict


.. autoclass:: cvm.CVMContext

.. autofunction:: cvm.cpu
.. autofunction:: cvm.gpu
.. autofunction:: cvm.formal
.. autofunction:: cvm.opencl
.. autofunction:: cvm.context

