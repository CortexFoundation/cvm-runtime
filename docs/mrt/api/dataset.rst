
***************
MRT Dataset API
***************

.. contents::

.. _mrt_dataset_api:

.. automodule:: mrt.dataset


Abstract Dataset
----------------
.. autoclass:: mrt.dataset.Dataset
  :members: _load_data, iter_func
  :undoc-members: metrics, validate


Customized Datasets
-------------------
.. autoclass:: COCODataset
  :members: _load_data, metrics, validate

.. autoclass:: VOCDataset
  :members: _load_data, metrics, validate


.. autoclass:: VisionDataset
  :members: metrics, validate


.. autoclass:: ImageNetDataset
  :members: _load_data


.. autoclass:: Cifar10Dataset 
  :members: _load_data


.. autoclass:: QuickDrawDataset 
  :members: _load_data


.. autoclass:: MnistDataset
  :members: _load_data


.. autoclass:: TrecDataset 
  :members: _load_data, validate