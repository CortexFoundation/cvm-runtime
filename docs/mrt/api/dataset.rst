
.. _mrt_dataset_api:

********************
MRT Dataset Deveplop
********************

.. contents::

.. py:module:: mrt.dataset

Dataset to be loaded by MRT has been abstracted as an
python base class named: :class:`Dataset`.
The ``Dataset`` class defines many interface function to
be implemented in the concrete derived dataset class. And
the details about that refer to the following sections please.

.. note::
  **Usage**: The dataset class is mainly invoked via the
  ``main2.py`` program and developers may also add extra
  custom derived dataset class and invoke the unify API.

We have achieved some common datasets in MRT dataset module,
including image classification, detection and NLP models.


Module Infomation
-----------------

module path: ``mrt.dataset``
module exports: ``DS_REG``, ``Dataset``

**DS_REG**

``DS_REG`` contains all the implemented and registered
dataset obeying the pair format of dataset name and concrete
class.

The supported(have implemented and registered into ``DS_REG``)
datasets are listed as below:

- :class:`COCODataset`
- :class:`VOCDataset`
- :class:`ImageNetDataset`
- :class:`Cifar10Dataset`
- :class:`QuickDrawDataset`
- :class:`MnistDataset`
- :class:`TrecDataset`

``register_dataset`` is a convinent function to decorate
the concrete dataset class, which can auto-register into
export variable: ``DS_REG``.

.. _abstract_dataset:
 
Abstract Dataset
----------------
.. autoclass:: mrt.dataset.Dataset
  :members: _load_data, iter_func
  :undoc-members: metrics, validate


Common Datasets
---------------
.. autoclass:: mrt.dataset.COCODataset
  :members: _load_data, metrics, validate

.. autoclass:: mrt.dataset.VOCDataset
  :members: _load_data, metrics, validate


.. autoclass:: mrt.dataset.VisionDataset
  :members: metrics, validate


.. autoclass:: mrt.dataset.ImageNetDataset
  :members: _load_data


.. autoclass:: mrt.dataset.Cifar10Dataset 
  :members: _load_data


.. autoclass:: mrt.dataset.QuickDrawDataset 
  :members: _load_data


.. autoclass:: mrt.dataset.MnistDataset
  :members: _load_data


.. autoclass:: mrt.dataset.TrecDataset 
  :members: _load_data, validate


Customize Dataset
-----------------

One may want to add implemantary dataset into MRT framework,
and there are two situations: the extra dataset format is
compatible with the existed dataset such as another imagenet
dataset with the same ``MxNet Record Binary`` file format,
or the dataset has an unique format that need to customize
the data load logic.

Compatible Format
~~~~~~~~~~~~~~~~~

For dataset that is compatible with existed dataset, one can
simply reuse the corresponding dataset class with changing
the dataset root directory, since the abstract dataset has
supplied the extra ``root`` parameter to replace the default
MRT dataset location.

Codes like:

.. code-block::

  ds = dataset.DS_REG['imagenet'](
      (16, 3, 614, 614), # the dataset input shape
      root="your/dataset/path", # specify the new dataset path
      )

Unique Format
~~~~~~~~~~~~~

You need to implement the unique dataset class after importing
the MRT dataset package. And we suggest that you review the
section: :ref:`abstract_dataset` for the dataset interface.

Generally, one should derive the :class:`Dataset` class and
implement the four abstract functions: ``_load_data``,
``iter_func``, ``metrics``, ``validate``.


Here are some example codes:

.. code-block::

  from mxnet import ndarray as nd
  @register_dataset("my_dataset")
  class MyDataset(Dataset):
      def _load_data(self):
          B = self.ishape[0]
          def _data_loader():
              for i in range(1000):
                  yield nd.array([i + c for c in range(B)])
          self.data = _data_loader()
  
      # use the default `iter_func` defined in base class
  
      def metrics(self):
          return {"count": 0, "total": 0}
      def validate(self, metrics, predict, label):
          for idx in range(predict.shape[0]):
              res_label = predict[idx].asnumpy().argmax()
              data_label = label[idx].asnumpy()
              if res_label == data_label:
                  metrics["acc"] += 1
              metrics["total"] += 1
          acc = 1. * metrics["acc"] / metrics["total"]
          return "{:6.2%}".format(acc)
  
  # usage
  md_cls = DS_REG["my_dataset"]
  ds = md_cls([8]) # batch size is 8
  data_iter_func = ds.iter_func()
  data_iter_func() # get the batch data

  # output
  NDArray<[0, 1, 2, 3, 4, 5, 6, 7] @ctx(cpu)>










