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

A ``DLTensor`` is a data structure that holds the data, running context, shape
of the data and other information. Considering that the ``DLTensor`` is still
very simple, it should be wrapped in a container class providing operations
and optimizations like reference count and lazy copy, that's why ``Container``
class is introduced -- to store other necessary data right after a ``DLTensor``.

The memory management logic in C++ provides some convenient but tricky ways to
initialize an ``NDArray`` or a ``DLTensor`` object. There's an interface called
``CVMArrayAlloc`` that creates a ``DLTensor`` and returns a pointer to it.
However, **what the interface actually returned is a pointer to a** ``Container``.
The memory layout princilpe of C++ promises ``DLTensor`` is right at the
beginning of a ``Container`` and thus, a ``Container*`` can be treated as a
``DLTensor*``.

The benifit of such method is that memory memagement procedure is totally
transparent to API callers, which means all it can see is a ``DLTensor`` and all it
need to do is call an API to get this tensor and call another API to destruct it.
Destructing the tensor needs the data in its ``Container``.

All in all, remember that **users can only create a** ``DLTensor`` ** with**
``CVMArrayAlloc`` **, must destruct it with calling** ``CVMArrayFree`` and should
NEVER write into the memory after it.


NDArray & API
-------------

CVMArrayAlloc
~~~~~~~~~~~~~

As mentioned above, this API creats a pointer to a ``DLTensor`` in users' view. What
it actually returns, however, is a point to a ``Container`` and the content outside
the ``DLTensor`` but inside the ``Container`` should never be modified.

For detailed information, you can refer to ?????????

CVMArrayFree
~~~~~~~~~~~~

If a ``DLTensor`` is created manually created by calling ``CVMArrayAlloc``, the user
then must call ``CVMArrayFree`` to free the memory. Otherwise memory leakage may
happen.


NDArray & Model
---------------

In present CVM version, the most useful use case of NDArray is using it in a model.
To use it in a model, persistence is necessary. ``CVMSaveParamsDict`` and
``CVMLoadParamsDict`` do the persistence work, saving and loading a string-value
dictionary.

For detailed information, you can refer to ????????
