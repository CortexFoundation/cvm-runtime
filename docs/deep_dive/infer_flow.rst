
.. _example-infer-flow:

**********************
Example Inference Flow
**********************

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

- declaration: ``include/cvm/runtime/c_runtime_api.h``
- definition: ``src/runtime/ndarray.cc``

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

- declaration: ``inclulde/cvm/model.h``
- definition: ``src/core/model.cc``

The API above invokes the lower level
:cpp:class:`cvm::runtime::CVMModel` class, which exposes the
mainly inference interface.

And the :cpp:class:`cvm::runtime::CVMModel` wrappered a pure
on-device graph runtime, and set extra post-process logic on
output Tensor.
More details refer to source code please.

Graph Runtime
-------------

- declaration: ``src/runtime/graph_runtime.h``
- definition: ``src/runtime/graph_runtime.cc``

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
registry are located at directory ``src/top``.

Besides, all the above registry function won't set the specific
operators' inference logic. It's a device-relative code
implementation, so we put the forward function code in the
``src/runtime/{device_type}/ops`` directory according to the
device type. Currently we have achieved three on-device version:

- CPU
- CUDA
- FORMAL

The OPENCL is still in progress and not the latest version.
Please refer to the source code for more detail process logic
and the next section will abstract the mathematical logic in
the formalization format.

