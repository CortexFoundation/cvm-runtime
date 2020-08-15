NDArray
=============

This is the document of ``NDArray``, multi dimension array containing data, a
core module in CVM Runtime. This core function is implemented with C++ and
there are APIs for other languages.

Overview
--------

Understanding the memory layout will help understanding the function.

``NDArray`` has a, and only has a, pointer to a ``Container`` class.

The ``Container`` class is a wrapper for the core data structure ``DLTensor``,
together with some other variables, providing useful features like reference
count and lazy copy. The memory layout is as below: 
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

This shows how the NDArray works.

A ``DLTensor`` is a data structure that holds the data, running context, shape
of the data and other informations. Considering that the ``DLTensor`` is still
very simple, it should be wrapped in a container class providing operations
and optimizations like reference count and lazy copy, that's why ``Container``
class is introduced.

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

All in all, remember that **user can create a** ``DLTensor`` **only with**
``CVMArrayAlloc`` **and must destruct it with calling** ``CVMArrayFree``
