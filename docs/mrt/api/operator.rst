
****************
MRT Operator API
****************

.. contents::

mrt.tfm_base
------------
.. _mrt_tfm_base_api:

.. automodule:: mrt.tfm_base


.. autoclass:: mrt.tfm_base.Transformer
  :members:


mrt.tfm_ops
-----------

.. automodule:: mrt.tfm_ops

.. _mrt_tfm_ops_api:

MxNet Supported Operators are listed as below:

- NN Operators

  + :py:class:`Convolution <mrt.tfm_ops.Convolution>`
  + :py:class:`BatchNorm <mrt.tfm_ops.BatchNorm>`
  + :py:class:`Pooling <mrt.tfm_ops.Pooling>`
  + :py:class:`FullyConnected <mrt.tfm_ops.FullyConnected>`
  + :py:class:`BatchDot(not compiled) <mrt.tfm_ops.BatchDot>`
  + :py:class:`Activation <mrt.tfm_ops.Activation>`
  + :py:class:`LeakyReLU <mrt.tfm_ops.LeakyReLU>`
  + :py:class:`Relu <mrt.tfm_ops.Relu>`
  + :py:class:`Pad(not compiled) <mrt.tfm_ops.Pad>`
  + :py:class:`UpSampling <mrt.tfm_ops.UpSampling>`

- Transform Operators

  + :py:class:`Transpose <mrt.tfm_ops.Transpose>`
  + :py:class:`Embedding <mrt.tfm_ops.Embedding>`
  + :py:class:`SliceLike <mrt.tfm_ops.SliceLike>`
  + :py:class:`SliceAxis <mrt.tfm_ops.SliceAxis>`
  + :py:class:`SliceChannel <mrt.tfm_ops.SliceChannel>`
  + :py:class:`SwapAxis <mrt.tfm_ops.SwapAxis>`
  + :py:class:`Concat <mrt.tfm_ops.Concat>`
  + :py:class:`ReshapeLike <mrt.tfm_ops.ReshapeLike>`
  + :py:class:`tile <mrt.tfm_ops.Tile>`
  + :py:class:`expand_dims <mrt.tfm_ops.ExpandDims>`
  + :py:class:`repeat <mrt.tfm_ops.Repeat>`
  + :py:class:`Flatten <mrt.tfm_ops.Flatten>`
  + :py:class:`floor <mrt.tfm_ops.Floor>`
  + :py:class:`ceil <mrt.tfm_ops.Ceil>`
  + :py:class:`round <mrt.tfm_ops.Round>`
  + :py:class:`fix <mrt.tfm_ops.Fix>`
  + :py:class:`Cast <mrt.tfm_ops.Cast>`
  + :py:class:`slice <mrt.tfm_ops.Slice>`
  + :py:class:`Reshape <mrt.tfm_ops.Reshape>`
  + :py:class:`squeeze <mrt.tfm_ops.Squeeze>`

- Mathematic Operators

  + :py:class:`Sigmoid <mrt.tfm_ops.Sigmoid>`
  + :py:class:`Exp <mrt.tfm_ops.Exp>`
  + :py:class:`Softmax <mrt.tfm_ops.Softmax>`
  + :py:class:`Dropout <mrt.tfm_ops.Dropout>`
  + :py:class:`argmax <mrt.tfm_ops.Argmax>`
  + :py:class:`argmin <mrt.tfm_ops.Argmin>`

- Arrange Operators

  + :py:class:`ZerosLike <mrt.tfm_ops.ZerosLike>`
  + :py:class:`OnesLike <mrt.tfm_ops.OnesLike>`
  + :py:class:`_arange <mrt.tfm_ops.Arange>`

- Broadcast Operators

  + :py:class:`broadcast_add <mrt.tfm_ops.BroadcastAdd>`
  + :py:class:`broadcast_sub <mrt.tfm_ops.BroadcastSub>`
  + :py:class:`broadcast_mul <mrt.tfm_ops.BroadcastMul>`
  + :py:class:`broadcast_like <mrt.tfm_ops.BroadcastLike>`
  + :py:class:`broadcast_to <mrt.tfm_ops.BroadcastTo>`
  + :py:class:`broadcast_greater <mrt.tfm_ops.BroadcastGreater>`
  + :py:class:`_minimum <mrt.tfm_ops.Minimum>`
  + :py:class:`_maximum <mrt.tfm_ops.Maximum>`

- Elemwise Operators

  + :py:class:`ElemwiseAdd <mrt.tfm_ops.ElemwiseAdd>`
  + :py:class:`ElemwiseSub <mrt.tfm_ops.ElemwiseSub>`
  + :py:class:`Clip <mrt.tfm_ops.Clip>`
  + :py:class:`negative <mrt.tfm_ops.Negative>`
  + :py:class:`abs <mrt.tfm_ops.Abs>`

- Scalar Operators

  + :py:class:`MulScalar <mrt.tfm_ops.MulScalar>`
  + :py:class:`DivScalar <mrt.tfm_ops.DivScalar>`
  + :py:class:`PlusScalar <mrt.tfm_ops.PlusScalar>`
  + :py:class:`GreaterScalar <mrt.tfm_ops.GreaterScalar>`

- Reduce Operators

  + :py:class:`Sum <mrt.tfm_ops.Sum>`
  + :py:class:`max <mrt.tfm_ops.Max>`
  + :py:class:`min <mrt.tfm_ops.Min>`

- Custom Operators

  + :py:class:`Custom <mrt.tfm_ops.Custom>`

- Vision Operators

  + :py:class:`_contrib_box_nms <mrt.tfm_ops.BoxNms>`
  + :py:class:`where <mrt.tfm_ops.Where>`


.. autoclass:: mrt.tfm_ops.Null
  :members: quantize

.. autoclass:: mrt.tfm_ops.Transpose
  :members: fuse_transpose

.. autoclass:: mrt.tfm_ops.Relu
  :members: fuse_transpose

.. autoclass:: mrt.tfm_ops.LeakyReLU
  :members: validate, rewrite, fuse_transpose

.. autoclass:: mrt.tfm_ops.MulScalar
  :members: rewrite

.. autoclass:: mrt.tfm_ops.DivScalar
  :members: rewrite

.. autoclass:: mrt.tfm_ops.Activation
  :members: validate

.. autoclass:: mrt.tfm_ops.Convolution
  :members: rewrite, quantize

.. autoclass:: mrt.tfm_ops.Pad
  :members: compile

.. autoclass:: mrt.tfm_ops.ExpandDims
  :members:

.. autoclass:: mrt.tfm_ops.Embedding
  :members: 

.. autoclass:: mrt.tfm_ops.Repeat
  :members: 

.. autoclass:: mrt.tfm_ops.BoxNms
  :members: 

.. autoclass:: mrt.tfm_ops.SliceLike
  :members:

.. autoclass:: mrt.tfm_ops.SliceAxis
  :members:

.. autoclass:: mrt.tfm_ops.SliceChannel
  :members:

.. autoclass:: mrt.tfm_ops.UpSampling
  :members:

.. autoclass:: mrt.tfm_ops.FullyConnected
  :members: rewrite, quantize

.. autoclass:: mrt.tfm_ops.Sigmoid
  :members: quantize

.. autoclass:: mrt.tfm_ops.Exp
  :members: quantize

.. autoclass:: mrt.tfm_ops.Softmax
  :members: quantize

.. autoclass:: mrt.tfm_ops.Pooling
  :members: validate, rewrite

.. autoclass:: mrt.tfm_ops.BroadcastMul
  :members: quantize, prepare_for_compile

.. autoclass:: mrt.tfm_ops.BroadcastAdd
  :members: quantize

.. autoclass:: mrt.tfm_ops.BroadcastSub
  :members: quantize

.. autoclass:: mrt.tfm_ops.BroadcastTo
  :members:

.. autoclass:: mrt.tfm_ops.BroadcastGreater
  :members:

.. autoclass:: mrt.tfm_ops.Concat
  :members: fuse_transpose, quantize

.. autoclass:: mrt.tfm_ops.Sum
  :members: fuse_transpose, quantize

.. autoclass:: mrt.tfm_ops.BatchNorm
  :members: rewrite

.. autoclass:: mrt.tfm_ops.Flatten
  :members:

.. autoclass:: mrt.tfm_ops.Floor
  :members:

.. autoclass:: mrt.tfm_ops.Ceil
  :members:

.. autoclass:: mrt.tfm_ops.Round
  :members:

.. autoclass:: mrt.tfm_ops.Fix
  :members:

.. autoclass:: mrt.tfm_ops.Cast
  :members:

.. autoclass:: mrt.tfm_ops.Slice
  :members:

.. autoclass:: mrt.tfm_ops.Reshape
  :members:

.. autoclass:: mrt.tfm_ops.Custom
  :members: validate

.. autoclass:: mrt.tfm_ops.Clip
  :members: fuse_transpose, quantize

.. autoclass:: mrt.tfm_ops.Minimum
  :members:

.. autoclass:: mrt.tfm_ops.Maximum
  :members:

.. autoclass:: mrt.tfm_ops.Max
  :members:

.. autoclass:: mrt.tfm_ops.Min
  :members:

.. autoclass:: mrt.tfm_ops.Argmax
  :members:

.. autoclass:: mrt.tfm_ops.Argmin
  :members:

.. autoclass:: mrt.tfm_ops.Abs
  :members:

.. autoclass:: mrt.tfm_ops.ElemwiseAdd
  :members: fuse_transpose, quantize


.. autoclass:: mrt.tfm_ops.ElemwiseSub
  :members: fuse_transpose, quantize

.. autoclass:: mrt.tfm_ops.Dropout
  :members: fuse_transpose

.. autoclass:: mrt.tfm_ops.Arange
  :members:

.. autoclass:: mrt.tfm_ops.Tile
  :members:

.. autoclass:: mrt.tfm_ops.Negative
  :members:

.. autoclass:: mrt.tfm_ops.SwapAxis
  :members: rewrite

.. autoclass:: mrt.tfm_ops.PlusScalar
  :members:

.. autoclass:: mrt.tfm_ops.ZerosLike
  :members: rewrite

.. autoclass:: mrt.tfm_ops.OnesLike
  :members: rewrite

.. autoclass:: mrt.tfm_ops.GreaterScalar
  :members: validate

.. autoclass:: mrt.tfm_ops.Where
  :members:

.. autoclass:: mrt.tfm_ops.Squeeze
  :members:

.. .. autoclass:: mrt.tfm_ops.L2Normalization
..   :members: quantize

.. no need to quantize the ``sqrt`` operator.
.. .. autoclass:: mrt.tfm_ops.Sqrt
..   :members: quantize

.. .. autoclass:: mrt.tfm_ops.InstanceNorm
..   :members: rewrite

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
----------
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
