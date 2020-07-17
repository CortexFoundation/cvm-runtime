.. _mrt_tfm_ops_api:

*****************
MRT operators API
*****************

mrt.tfm_ops
___________
.. automodule:: mrt.tfm_ops


.. autoclass:: mrt.tfm_ops.Transpose
  :members: fuse_transpose


.. autoclass:: mrt.tfm_ops.Convolution
  :members: rewrite, quantize


.. autoclass:: mrt.tfm_ops.FullyConnected
  :members: rewrite, quantize


.. autoclass:: mrt.tfm_ops.LeakyReLU
  :members: validate, rewrite


.. autoclass:: mrt.tfm_ops.Activation
  :members: validate


.. autoclass:: mrt.tfm_ops.Pad
  :members: compile


.. autoclass:: mrt.tfm_ops.Embedding
  :members: quantize


.. autoclass:: mrt.tfm_ops.SliceLike
  :members: quantize


.. autoclass:: mrt.tfm_ops.SliceAxis
  :members: rewrite


.. autoclass:: mrt.tfm_ops.Sigmoid
  :members: quantize


.. autoclass:: mrt.tfm_ops.Exp
  :members: quantize


.. autoclass:: mrt.tfm_ops.BroadcastAdd
  :members: quantize


.. autoclass:: mrt.tfm_ops.BroadcastSub
  :members: quantize


.. autoclass:: mrt.tfm_ops.Concat
  :members: fuse_transpose, quantize


.. autoclass:: mrt.tfm_ops.ElemwiseAdd
  :members: fuse_transpose, quantize


.. autoclass:: mrt.tfm_ops.ElemwiseSub
  :members: fuse_transpose, quantize


.. autoclass:: mrt.tfm_ops.Softmax
  :members: quantize

.. autofunction:: mrt.tfm_ops._quantize_scale

.. autofunction:: mrt.tfm_ops._quantize_xwb

.. autofunction:: mrt.tfm_ops._quantize_table

.. autofunction:: mrt.tfm_ops.reverse_sequence
