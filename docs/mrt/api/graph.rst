
*************
MRT Graph API
*************

.. contents::

mrt.tfm_pass
____________
.. _mrt_tfm_pass_api:

.. automodule:: mrt.tfm_pass

.. autofunction:: mrt.tfm_pass.calculate_ops

.. autofunction:: mrt.tfm_pass.fuse_transpose

.. autofunction:: mrt.tfm_pass.rewrite

.. autofunction:: mrt.tfm_pass.quantize

.. autofunction:: mrt.tfm_pass.prepare_for_compile

.. autofunction:: mrt.tfm_pass.to_cvm

.. autofunction:: mrt.tfm_pass.fuse_multiple_inputs

.. autofunction:: mrt.tfm_pass.name_duplicate_check

.. autofunction:: mrt.tfm_pass.params_unique

.. autofunction:: mrt.tfm_pass.input_name_replace

.. autofunction:: mrt.tfm_pass.fuse_constant

.. autofunction:: mrt.tfm_pass.attach_input_shape

.. autofunction:: mrt.tfm_pass.infer_shape

.. autofunction:: mrt.tfm_pass._collect_attribute

.. autofunction:: mrt.tfm_pass.collect_op_names

.. autofunction:: mrt.tfm_pass.fuse_multiple_outputs

.. autofunction:: mrt.tfm_pass._get_opt

.. autofunction:: mrt.tfm_pass.sym_calibrate

.. autofunction:: mrt.tfm_pass.convert_params_dtype


mrt.sym_utils
_____________
.. _mrt_sym_utils_api:

.. automodule:: mrt.sym_utils

.. autofunction:: mrt.sym_utils.is_op

.. autofunction:: mrt.sym_utils.is_var

.. autofunction:: mrt.sym_utils.is_params

.. autofunction:: mrt.sym_utils.is_inputs

.. autofunction:: mrt.sym_utils.nd_array

.. autofunction:: mrt.sym_utils.nd_arange

.. autofunction:: mrt.sym_utils.nd_full

.. autofunction:: mrt.sym_utils.nd_zeros

.. autofunction:: mrt.sym_utils.nd_ones

.. autofunction:: mrt.sym_utils.check_graph

.. autofunction:: mrt.sym_utils.get_attr

.. autofunction:: mrt.sym_utils.get_nd_op

.. autofunction:: mrt.sym_utils.get_mxnet_op

.. autofunction:: mrt.sym_utils.get_nnvm_op

.. autofunction:: mrt.sym_utils.sym_iter

.. autofunction:: mrt.sym_utils.nd_const

.. autofunction:: mrt.sym_utils.topo_sort

.. autofunction:: mrt.sym_utils.get_entry_id

.. autofunction:: mrt.sym_utils.get_node

.. autofunction:: mrt.sym_utils.topo_visit

.. autofunction:: mrt.sym_utils.topo_visit_transformer
