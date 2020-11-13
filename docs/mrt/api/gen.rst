
********************************
MRT Generalized Quantization API
********************************

.. contents::

.. _mrt_gen_api:


Quantizer API
-------------
.. automodule:: mrt.V2.tfm_types


.. autoclass:: mrt.V2.tfm_types.Feature
  :members:


.. autoclass:: mrt.V2.tfm_types.AFeature


.. autoclass:: mrt.V2.tfm_types.MMFeature


.. autoclass:: mrt.V2.tfm_types.Buffer
  :members:


.. autoclass:: mrt.V2.tfm_types.SBuffer


.. autoclass:: mrt.V2.tfm_types.SZBuffer


.. autoclass:: mrt.V2.tfm_types.Quantizer
  :members:


.. autoclass:: mrt.V2.tfm_types.USQuantizer


.. autoclass:: mrt.V2.tfm_types.UAQuantizer


.. autoclass:: mrt.V2.tfm_types.Optimizor
  :members:


.. autoclass:: HVOptimizor


.. autoclass:: MAOptimizor


.. autoclass:: KLDOptimizor


Graph API
---------
.. automodule:: mrt.V2.tfm_pass


.. autofunction:: mrt.V2.tfm_pass.sym_config_infos


.. autofunction:: mrt.V2.tfm_pass.deserialize


.. autofunction:: mrt.V2.tfm_pass.sym_calibrate


.. autofunction:: mrt.V2.tfm_pass.sym_separate_pad


.. autofunction:: mrt.V2.tfm_pass.sym_separate_bias


.. autofunction:: mrt.V2.tfm_pass.sym_slice_channel


.. autofunction:: mrt.V2.tfm_pass.quantize


Transformer API
---------------
.. automodule:: mrt.V2.tfm_base


.. autoclass:: mrt.V2.tfm_base.Transformer
  :members: quantize, slice_channel
