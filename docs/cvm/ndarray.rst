NDArray
=============

This is the document of ``NDArray``, multi-dimension array containing data, a
core module in CVM Runtime. This part of documentation helps with understanding
the design and APIs of ``NDArray`` and related classes or methods.

Overview
--------

``NDArray``, as discussed above, is the multi-dimension array containing data.
This class provides users with memory management and persistence. The core
function is implemented with C++ and there are APIs for other languages.

Memory Layout
-------------

Understanding the memory layout will help understanding the function.

``NDArray`` has a, and only has a, pointer to a ``Container`` class.

The ``Container`` class is a wrapper for the core data structure ``DLTensor``,
containing a ``DLTensor`` instance instead of a pointer, together with some
other variables, providing useful features like reference count and lazy copy.

The memory layout is as below:
::
                 
  +-NDArray----+        +-Container--+
  | Container*-|------->| +DLTensor+ |      +-----------+
  +------------+        | | data*--|-|----->|  Memory   |
                        | | ndim   | |      |           |
                        | | context| |      |           |
                        | |  ...   | |      +-----------+
                        | +--------+ |
                        |  ref_cnt   |
                        |  type_code |
                        |    ...     |
                        +------------+

This graph shows the memory layout of NDArray.

A ``DLTensor`` is a data structure that holds the data, data context, shape of
the data and other information. Considering that the ``DLTensor`` is just a 
simple struct for data containing no functions, it should be wrapped in a container 
class providing operations
and optimizations like reference count and lazy copy, that's why ``Container``
class is introduced -- to store other necessary data right after a ``DLTensor``.

The memory management logic in C++ provides some convenient but tricky ways to
initialize an ``NDArray`` or a ``DLTensor`` object. There's an interface called
``CVMArrayAlloc`` that creates a ``DLTensor`` and returns a pointer to it.
However, **what the interface actually returned is a pointer to a** ``Container``.
The memory layout princilpe of C++ promises ``DLTensor`` is right at the
beginning of a ``Container`` and thus, a ``Container*`` can be treated as a
``DLTensor*`` and can be converted back to a ``Container*`` if necessary.

The benifit of such method is that memory memagement procedure is totally
transparent to API callers, which means all it can see is a ``DLTensor`` and all it
need to do is call an API to get this tensor and call another API to destruct it.
Destructing the tensor needs the data in its ``Container``.

All in all, remember that **users can only create a** ``DLTensor`` **with**
``CVMArrayAlloc`` **, must destruct it with calling** ``CVMArrayFree`` **and should
NEVER write into the memory after it.**


NDArray & API
-------------

CVMArrayAlloc
~~~~~~~~~~~~~

As mentioned above, this API creats a pointer to a ``DLTensor`` in users' view. What
it actually returns, however, is a point to a ``Container`` and the content outside
the ``DLTensor`` but inside the ``Container`` should never be modified.

For detailed information, you can refer to ``CVMArrayAlloc`` in :ref:`c-backend-api-label`.

CVMArrayFree
~~~~~~~~~~~~

If a ``DLTensor`` is created manually created by calling ``CVMArrayAlloc``, the user
then must call ``CVMArrayFree`` to free the memory. Otherwise memory leakage may
happen.

For detailed information, you can refer to ``CVMArrayFree`` in :ref:`c-backend-api-label`.


NDArray & Model
---------------

A ``CvmRuntime`` represents a model. The components of a model can be classified
into two types: either data or parameters. Data comes from users' input while
parameters are deterministic for a certain model and should be loaded from some
storage. Parameters are usually stored as in a dictionary: the key is the name
indicating its place in the model and the value is, of course, a ``DLTensor``.

That's why persistence of such dictionary is necessary in this project. A model,
aka ``CvmRuntime``, calls ``LoadParams`` to load the dictionary into itself and
there's an API ``CVMLoadParamsDict`` for other languages to get the dictionary.
What's more, models trained by other frameworks should be converted to integer
models and the converter needs an API to save the converter model:
``CVMSaveParamsDict`` is then introduced.

Such APIs save/load names and data. Saving and loading names is easy while
saving and loading a ``DLTensor`` is not that trivial so related functions are
needed. As mentioned above, ``DLTensor`` doesn't provide management methods of
itself so we usually wrap it with an ``NDArray`` to do the management.
``NDArray`` provides ``Save`` and ``Load`` APIs.

The layout of a saved parameter dictionary is:

- the first 64 bits is a magic number ``0xF7E58D4F05049CB7``
- the following 64 bits is reserved.
- the following is the names, aka keys, of parameters, whose layout is

  + the first 64 bits is the number of keys
  + for each key, 64 bits indicating the length and followed by the content of the key.

- the values, whose layout is

  + the first 64 bits is the number of values
  + for each value, 64 bits magic number ``0xDD5E40F096B4A13F``, 64 bits reserved, followed by content of ``DLTensor``. ``DLTensor`` is POD type so it is easy to store.

For detailed information for the APIs, you can refer to :ref:`c-backend-api-label`.
