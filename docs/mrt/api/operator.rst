
****************
MRT Operator API
****************

.. contents::

mrt.tfm_base
____________
.. _mrt_tfm_base_api:

.. automodule:: mrt.tfm_base


.. autoclass:: mrt.tfm_base.Transformer
  :members:


mrt.tfm_ops
___________
.. _mrt_tfm_ops_api:

.. automodule:: mrt.tfm_ops


.. autoclass:: mrt.tfm_ops.Transpose
  :members: fuse_transpose


.. autoclass:: mrt.tfm_ops.Convolution
  :members: rewrite, quantize


.. autoclass:: mrt.tfm_ops.FullyConnected
  :members: rewrite, quantize


.. autoclass:: mrt.tfm_ops.LeakyReLU
  :members: validate, rewrite, fuse_transpose


.. autoclass:: mrt.tfm_ops.Relu
  :members: fuse_transpose


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


.. autoclass:: mrt.tfm_ops.Pooling
  :members: validate, rewrite


.. autoclass:: mrt.tfm_ops.BroadcastMul
  :members: quantize, prepare_for_compile

.. autoclass:: mrt.tfm_ops.Sum
  :members: fuse_transpose, quantize


.. autoclass:: mrt.tfm_ops.BatchNorm
  :members: rewrite


.. autoclass:: mrt.tfm_ops.Custom
  :members: validate


.. autoclass:: mrt.tfm_ops.Clip
  :members: fuse_transpose, quantize


.. autoclass:: mrt.tfm_ops.Dropout
  :members: fuse_transpose


.. autoclass:: mrt.tfm_ops.SwapAxis
  :members: rewrite


.. autoclass:: mrt.tfm_ops.ZerosLike
  :members: rewrite


.. autoclass:: mrt.tfm_ops.OnesLike
  :members: rewrite


.. autoclass:: mrt.tfm_ops.GreaterScalar
  :members: validate


.. autoclass:: mrt.tfm_ops.L2Normalization
  :members: quantize


.. autoclass:: mrt.tfm_ops.Null
  :members: quantize


.. autoclass:: mrt.tfm_ops.MulScalar
  :members: rewrite


.. autoclass:: mrt.tfm_ops.DivScalar
  :members: rewrite


.. autoclass:: mrt.tfm_ops.PlusScalar
  :members: rewrite


.. autoclass:: mrt.tfm_ops.Sqrt
  :members: quantize


.. autoclass:: mrt.tfm_ops.InstanceNorm
  :members: rewrite


.. autoclass:: mrt.tfm_ops.BatchDot
  :members: rewrite, quantize


.. autoclass:: mrt.tfm_ops.BroadcastLike
  :members: rewrite


.. autoclass:: mrt.tfm_ops.ReshapeLike
  :members: rewrite

.. autofunction:: mrt.tfm_ops._quantize_scale

.. autofunction:: mrt.tfm_ops._quantize_xwb

.. autofunction:: mrt.tfm_ops._quantize_table

.. autofunction:: mrt.tfm_ops.reverse_sequence

.. autofunction:: mrt.tfm_ops.reverse_transpose


mrt.cvm_op
__________
.. _mrt_cvm_op_api:

.. automodule:: mrt.cvm_op


.. autoclass:: mrt.cvm_op.Clip
  :members: forward


.. autoclass:: mrt.cvm_op.LeftShift
  :members: forward


.. autoclass:: mrt.cvm_op.RightShift
  :members: forward


.. autoclass:: mrt.cvm_op.LUT
  :members: forward


.. autoclass:: mrt.cvm_op.ClipProp


.. autoclass:: mrt.cvm_op.LeftShiftProp


.. autoclass:: mrt.cvm_op.RightShiftProp


.. autoclass:: mrt.cvm_op.LUTProp
