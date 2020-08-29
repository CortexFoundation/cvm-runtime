
********************************
MRT Generalized Quantization API
********************************

.. contents::

.. _mrt_gen_api:


Transformer Helper Types
------------------------
.. automodule:: mrt.gen.tfm_types


.. autoclass:: Feature


.. autoclass:: AFeature


.. autoclass:: MMFeature


.. autoclass:: Buffer


.. autoclass:: SBuffer


.. autoclass:: SZBuffer


.. autoclass:: Quantizer


.. autoclass:: USQuantizer


.. autoclass:: UAQuantizer


.. autoclass:: Optimizor


.. autoclass:: HVOptimizor


.. autoclass:: MAOptimizor


.. autoclass:: KLDOptimizor


Derived Transformer Class
-------------------------
.. autoclass:: mrt.gen.tfm_base.Transformer
  :members: slice_channel


Derived MRT Class
-----------------
.. autoclass:: mrt.gen.transformer.MRT
  :members: set_cfg_dict

Interface Passes
----------------
.. autofunction:: mrt.gen.tfm_pass.sym_slice_channel

.. autofunction:: mrt.gen.tfm_pass.sym_config_infos

.. autofunction:: mrt.gen.tfm_pass.deserialize


Operator Helper Functions
-------------------------
.. autofunction:: mrt.gen.tfm_ops.separate_bias

.. autofunction:: mrt.gen.tfm_ops.separate_pad
