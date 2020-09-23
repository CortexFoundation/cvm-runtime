
********************************
MRT Generalized Quantization API
********************************

.. contents::

.. _mrt_gen_api:


Quantizer API
-------------
.. automodule:: mrt.gen.tfm_types


.. autoclass:: mrt.gen.tfm_types.Feature
  :members:


.. autoclass:: mrt.gen.tfm_types.AFeature


.. autoclass:: mrt.gen.tfm_types.MMFeature


.. autoclass:: mrt.gen.tfm_types.Buffer
  :members:


.. autoclass:: mrt.gen.tfm_types.SBuffer


.. autoclass:: mrt.gen.tfm_types.SZBuffer


.. autoclass:: mrt.gen.tfm_types.Quantizer
  :members:


.. autoclass:: mrt.gen.tfm_types.USQuantizer


.. autoclass:: mrt.gen.tfm_types.UAQuantizer


.. autoclass:: mrt.gen.tfm_types.Optimizor
  :members:


.. autoclass:: HVOptimizor


.. autoclass:: MAOptimizor


.. autoclass:: KLDOptimizor


Graph API
---------
.. automodule:: mrt.gen.tfm_pass


.. autofunction:: mrt.gen.tfm_pass.sym_config_infos


.. autofunction:: mrt.gen.tfm_pass.deserialize


.. autofunction:: mrt.gen.tfm_pass.sym_calibrate


.. autofunction:: mrt.gen.tfm_pass.sym_separate_pad


.. autofunction:: mrt.gen.tfm_pass.sym_separate_bias


.. autofunction:: mrt.gen.tfm_pass.sym_slice_channel


.. autofunction:: mrt.gen.tfm_pass.quantize


Transformer API
---------------
.. automodule:: mrt.gen.tfm_base


.. autoclass:: mrt.gen.tfm_base.Transformer
  :members: quantize, slice_channel
