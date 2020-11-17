""" Dataset Class Definition.

    Customized datasets definition and customized interface
    definition including ``metrics``, ``validate``,
    ``_load_data`` and ``iter_func``.

    Only **crucial parts** of the custommized interface
    implementation are elaborated.
"""

import mxnet as mx
from mxnet import gluon
from mxnet import nd
from gluoncv import data as gdata
from gluoncv.data.batchify import Tuple, Stack, Pad
from gluoncv.data.transforms.presets.yolo import YOLO3DefaultValTransform
from gluoncv.utils.metrics.voc_detection import VOC07MApMetric
from gluoncv.utils.metrics.coco_detection import COCODetectionMetric
from gluoncv.data.transforms.presets.ssd import SSDDefaultValTransform
import numpy as np
import requests
import tarfile

import os
from os import path
import math
import pickle
import logging

from . import conf

__all__ = ["DS_REG", "Dataset"]

# dataset_dir = path.expanduser("~/.mxnet/datasets")
src = "http://0.0.0.0:8827"

def extract_file(tar_path, target_path):
    tar = tarfile.open(tar_path, "r")
    if path.exists(path.join(target_path,
                tar.firstmember.name)):
        return
    tar.extractall(target_path)
    tar.close()

def download_files(category, files, base_url=src, root=conf.MRT_DATASET_ROOT):
    logger = logging.getLogger("dataset")
    root_dir = path.join(root, category)
    os.makedirs(root_dir, exist_ok=True)

    for df in files:
        url = path.join(base_url, 'datasets', category, df)
        fpath = path.join(root_dir, df)
        if path.exists(fpath):
            continue
        fdir = path.dirname(fpath)
        if not path.exists(fdir):
            os.makedirs(fdir)

        logger.info("Downloading dateset %s into %s from url[%s]",
                df, root_dir, url)
        r = requests.get(url)
        if r.status_code != 200:
            logger.error("Url response invalid status code: %s",
                    r.status_code)
            exit()
        r.raise_for_status()
        with open(fpath, "wb") as fout:
            fout.write(r.content)
    return root_dir

DS_REG = {
    # "voc": VOCDataset,
    # "imagenet": ImageNetDataset,
    # "cifar10": Cifar10Dataset,
    # "quickdraw": QuickDrawDataset,
    # "mnist": MnistDataset,
    # "trec": TrecDataset,
    # "coco": COCODataset,
}

def register_dataset(name):
    def _wrapper(dataset):
        dataset.name = name
        if name in DS_REG:
            raise NameError("Dataset " + name + " has been registered")
        DS_REG[name] = dataset;
        return dataset 
    return _wrapper

class Dataset:
    """ Base dataset class, with pre-defined interface.

        The dataset directory is located at the ``root`` directory containing
        the dataset `name` directory. And the custom dataset should pass
        the parameter location of root, or implement the derived class
        of your data iterator, metrics and validate function.

        Notice:
            Our default imagenet dataset is organized as an ``record``
            binary format, which can amplify the throughput for image read.
            Custom Image dataset of third party could be preprocessed by the
            `im2rec` procedure to transform the image into the record format.

            The transformation script is located at ``docs/mrt/im2rec.py``.
            And more details refer to the script helper documentation
            please(print usage with command ``-h``).


        Parameters
        ==========
        input_shape: Tuple or List
            The input shape requested from user, and some dataset would
            check the format validity. Generally, specific dataset will
            do some checks for input shape, such as the channel number for
            image.
            Example: imagenet's input shape is like to this, (N, C, H, W),
            where the C must be equal to 3, H equals to W and N indicates
            the batch size user want. Different H(W) requests the dataset
            loader to resize image.

        root: os.path.Path or path string
            The location where dataset is stored, defined with variable
            ``MRT_DATASET_ROOT`` in conf.py or custom directory.


        **Custom Dataset Implementation (derived this class):**

            1. Register dataset name into DS_REG that can be accessed
            at the ``dataset`` package API. And releated function is
            the ``register_dataset`` function.

            2. Override the abstract method defined in base dataset class:

                _load_data(self) [Required]:
                    Load data from disk that stored into the data variable.
                    And save the required `data_loader` to the member: `data`.

                iter_func(self) [Optional]:
                    Return the tuple (data, label) for each invocation according
                    to the member `data` loaded from the function `_load_data`.

                    Also, this function is optional since we have implemented
                    a naive version if the member `data` is python generator-
                    compatible type, supporting the `iter(data)` function. Or
                    you will override the function you need.

                metrics(self) [Required]:
                    Return the metrics object for the dataset, such as
                    some auxiliary variable.

                validate(self, metrics, predict, label) [Required]:
                    Calculates the accuracy for model inference of string.
                    Return formated string type

        Examples
        ========
        >>> from mxnet import ndarray as nd
        >>> @register_dataset("my_dataset")
        >>> class MyDataset(Dataset):
        ...     def _load_data(self):
        ...         B = self.ishape[0]
        ...         def _data_loader():
        ...             for i in range(1000):
        ...                 yield nd.array([i + c for c in range(B)])
        ...         self.data = _data_loader()
        ...
        ...     # use the default `iter_func` defined in base class
        ...
        ...     def metrics(self):
        ...         return {"count": 0, "total": 0}
        ...     def validate(self, metrics, predict, label):
        ...         for idx in range(predict.shape[0]):
        ...             res_label = predict[idx].asnumpy().argmax()
        ...             data_label = label[idx].asnumpy()
        ...             if res_label == data_label:
        ...                 metrics["acc"] += 1
        ...             metrics["total"] += 1
        ...         acc = 1. * metrics["acc"] / metrics["total"]
        ...         return "{:6.2%}".format(acc)
        >>>
        >>> # usage
        >>> md_cls = DS_REG["my_dataset"]
        >>> ds = md_cls([8]) # batch size is 8
        >>> data_iter_func = ds.iter_func()
        >>> data_iter_func() # get the batch data
        NDArray<[0, 1, 2, 3, 4, 5, 6, 7] @ctx(cpu)>
    """
    name = None
    """ Registered Dataset Name """

    def __init__(self, input_shape, root=conf.MRT_DATASET_ROOT):
        self.ishape = input_shape

        if self.name is None:
            raise RuntimeError("Dataset name not set")

        # Dataset not to download the file, it's user's responsibility
        self.root_dir = path.join(root, self.name)
        # self.root_dir = download_files(
        #     self.name, self.download_deps, base_url, root) \
        #     if dataset_dir is None else dataset_dir
        # for fname in self.download_deps:
        #     if fname.endswith(".tar") or fname.endswith(".tar.gz"):
        #         extract_file(
        #               path.join(self.root_dir, fname), self.root_dir)

        self.data = None
        self._load_data()

    def metrics(self):
        raise NotImplementedError(
                "Derived " + self.name + " dataset not override the" +
                " base `metric` function defined in Dataset")

    def validate(self, metrics, predict, label):
        raise NotImplementedError(
                "Derived " + self.name + " dataset not override the" +
                " base `validate` function defined in Dataset")

    def _load_data(self):
        """ Load data from disk.

            Save the data loader into member `data` like:

            .. code-block:: python

                self.data = data_loader

            And validate the input shape if necessary:

            .. code-block:: python

                N, C, H, W = self.ishape
                assert C == 3 and H == W
        """
        raise NotImplementedError(
                "Derived " + self.name + " dataset not override the" +
                " base `_load_data` function defined in Dataset")

    def iter_func(self):
        """ Returns (data, label) iterator function.

            Get the iterator of `self.data` and iterate each batch sample
            with `next` function manually. Call like this:

            .. code-block:: python

                data_iter_func = dataset.iter_func()
                data, label = data_iter_func()
        """
        data_iter = iter(self.data)
        def _wrapper():
            return next(data_iter)
        return _wrapper


@register_dataset("coco")
class COCODataset(Dataset):
    # download_deps = ['val2017.zip']

    def _load_data(self):
        """ Customized _load_data method introduction.

            COCO dataset only support layout of NCHW and the number of channels must be 3, i.e. (batch_size, 3, input_size, input_size).

            The validation dataset will be created by *MS COCO Detection Dataset* and use SSDDefaultValTransform as data preprocess function.
        """
        assert len(self.ishape) == 4
        N, C, H, W = self.ishape
        assert C == 3
        self.val_dataset = gdata.COCODetection(
            root=self.root_dir, splits='instances_val2017', skip_empty=False)
        val_batchify_fn = Tuple(Stack(), Pad(pad_val=-1))
        self.data = gluon.data.DataLoader(
            self.val_dataset.transform(SSDDefaultValTransform(W, H)),
            batch_size=N, shuffle=False, batchify_fn=val_batchify_fn,
            last_batch='rollover', num_workers=30)

    def metrics(self):
        """ Customized metrics method introduction.

            COCODetectionMetric is used which is the detection metric for COCO bbox task.
        """
        _, _, H, W = self.ishape
        metric = COCODetectionMetric(
            self.val_dataset, '_eval', cleanup=True, data_shape=(H, W))
        metric.reset()
        return metric

    def validate(self, metrics, predict, label):
        """ Customized validate method introduction.

            The image height must be equal to the image width.

            The model output is [id, score, bounding_box], 
            where bounding_box is of layout (x1, y1, x2, y2).

            The data label is implemented as follows:

            .. code-block:: python

                map_name, mean_ap = metrics.get()
                acc = {k: v for k,v in zip(map_name, mean_ap)}
                acc = float(acc['~~~~ MeanAP @ IoU=[0.50, 0.95] ~~~~\\n']) / 100
        """
        det_ids, det_scores, det_bboxes = [], [], []
        gt_ids, gt_bboxes, gt_difficults = [], [], []

        _, _, H, W = self.ishape
        assert H == W
        ids, scores, bboxes = predict
        det_ids.append(ids)
        det_scores.append(scores)
        # clip to image size
        det_bboxes.append(bboxes.clip(0, H))
        gt_ids.append(label.slice_axis(axis=-1, begin=4, end=5))
        gt_difficults.append(
            label.slice_axis(axis=-1, begin=5, end=6) \
            if label.shape[-1] > 5 else None)
        gt_bboxes.append(label.slice_axis(axis=-1, begin=0, end=4))

        metrics.update(det_bboxes, det_ids, det_scores,
                            gt_bboxes, gt_ids, gt_difficults)
        names, values = metrics.get()
        acc = {k:v for k,v in zip(names, values)}
        acc = float(acc['~~~~ MeanAP @ IoU=[0.50,0.95] ~~~~\n']) / 100
        return "{:6.2%}".format(acc)


@register_dataset("voc")
class VOCDataset(Dataset):
    # name = "voc"
    # download_deps = ["VOCtest_06-Nov-2007.tar"]

    def _load_data(self):
        """ Customized _load_data method introduction.

            VOC dataset only support layout of NCHW and the number of channels must be 3, i.e. (batch_size, 3, input_size, input_size).

            The validation dataset will be created by Pascal *VOC detection Dataset* and use YOLO3DefaultValTransform as data preprocess function.
        """
        assert len(self.ishape) == 4
        N, C, H, W = self.ishape
        assert C == 3
        val_dataset = gdata.VOCDetection(
            root=path.join(self.root_dir, 'VOCdevkit'),
            splits=[('2007', 'test')])
        val_batchify_fn = Tuple(Stack(), Pad(pad_val=-1))
        self.data = gluon.data.DataLoader(
            val_dataset.transform(YOLO3DefaultValTransform(W, H)),
            N, False, batchify_fn=val_batchify_fn,
            last_batch='discard', num_workers=30)

    def metrics(self):
        """ Customized metric method introduction.

            VOC07MApMetric is used which is the Mean average precision metric for PASCAL V0C 07 dataset.
        """
        metric = VOC07MApMetric(
            iou_thresh=0.5, class_names=gdata.VOCDetection.CLASSES)
        metric.reset()
        return metric

    def validate(self, metrics, predict, label):
        """ Customized validate method introduction.

            The image height must be equal to the image width.

            The model output is [id, score, bounding_box], 
            where bounding_box is of layout (x1, y1, x2, y2).

            The data label is implemented as follows:

            .. code-block:: python

                map_name, mean_ap = metrics.get()
                acc = {k: v for k,v in zip(map_name, mean_ap)}['mAP']
        """
        det_ids, det_scores, det_bboxes = [], [], []
        gt_ids, gt_bboxes, gt_difficults = [], [], []

        _, _, H, W = self.ishape
        assert H == W
        ids, scores, bboxes = predict
        det_ids.append(ids)
        det_scores.append(scores)
        # clip to image size
        det_bboxes.append(bboxes.clip(0, H))
        gt_ids.append(label.slice_axis(axis=-1, begin=4, end=5))
        gt_difficults.append(
            label.slice_axis(axis=-1, begin=5, end=6) \
            if label.shape[-1] > 5 else None)
        gt_bboxes.append(label.slice_axis(axis=-1, begin=0, end=4))

        metrics.update(det_bboxes, det_ids, det_scores,
                            gt_bboxes, gt_ids, gt_difficults)
        map_name, mean_ap = metrics.get()
        acc = {k:v for k,v in zip(map_name, mean_ap)}['mAP']
        return "{:6.2%}".format(acc)


class VisionDataset(Dataset):
    def metrics(self):
        """ Customized metric method introduction.

            Computes accuracy classification score and top k predictions accuracy.
        """
        return [mx.metric.Accuracy(),
                mx.metric.TopKAccuracy(5)]

    def validate(self, metrics, predict, label):
        """ Customized metric method introduction.

            The model output include score for 1000 classes.
        """
        metrics[0].update(label, predict)
        metrics[1].update(label, predict)
        _, top1 = metrics[0].get()
        _, top5 = metrics[1].get()
        return "top1={:6.2%} top5={:6.2%}".format(top1, top5)


@register_dataset("imagenet")
class ImageNetDataset(VisionDataset):
    # name = "imagenet"
    download_deps = ["rec/val.rec", "rec/val.idx"]

    def _load_data(self):
        """ Customized _load_data method introduction.

            ImageNet dataset only support layout of NCHW and the number of channels must be 3, i.e. (batch_size, 3, input_size, input_size). The image height must be equal to the image width.

            The data preprocess process includes:

            .. math::
                crop_ratio = 0.875

            .. math::
                resize = ceil(H / crop\_ratio)

            .. math::
                mean_rgb = [123.68, 116.779, 103.939]

            .. math::
                std_rgb = [58.393, 57.12, 57.375]

            Use ImageRecordIter to iterate on image record io files.
        """
        assert len(self.ishape) == 4
        N, C, H, W = self.ishape
        assert C == 3
        assert H == W

        crop_ratio = 0.875
        resize = int(math.ceil(H / crop_ratio))
        mean_rgb = [123.68, 116.779, 103.939]
        std_rgb = [58.393, 57.12, 57.375]
        rec_val = path.join(self.root_dir, self.download_deps[0])
        rec_val_idx = path.join(self.root_dir, self.download_deps[1])


        self.data = mx.io.ImageRecordIter(
            path_imgrec         = rec_val,
            path_imgidx         = rec_val_idx,
            preprocess_threads  = 24,
            shuffle             = False,
            batch_size          = N,

            resize              = resize,
            data_shape          = (3, H, W),
            mean_r              = mean_rgb[0],
            mean_g              = mean_rgb[1],
            mean_b              = mean_rgb[2],
            std_r               = std_rgb[0],
            std_g               = std_rgb[1],
            std_b               = std_rgb[2],
        )

    def iter_func(self):
        def _wrapper():
            data = self.data.next()
            return data.data[0], data.label[0]
        return _wrapper


@register_dataset("cifar10")
class Cifar10Dataset(VisionDataset):
    # name = "cifar10"
    #  download_deps = ["cifar-10-binary.tar.gz"]

    def _load_data(self):
        """ Customized _load_data method introduction.

            Cifar10Dataset only support layout of NCHW and the number of channels must be 3, i.e. (batch_size, 3, 32, 32). The image height and width must be equal to 32.

            The data preprocess process includes:

            .. math::
                mean = [0.4914, 0.4822, 0.4465]

            .. math::
                std = [0.2023, 0.1994, 0.2010]
        """
        N, C, H, W = self.ishape
        assert C == 3 and H == W and H == 32
        transform_test = gluon.data.vision.transforms.Compose([
            gluon.data.vision.transforms.ToTensor(),
            gluon.data.vision.transforms.Normalize([0.4914, 0.4822, 0.4465],
                                                   [0.2023, 0.1994, 0.2010])])
        self.data = gluon.data.DataLoader(
            gluon.data.vision.CIFAR10(root=self.root_dir,
                train=False).transform_first(transform_test),
            batch_size=N, shuffle=False, num_workers=4)


@register_dataset("quickdraw")
class QuickDrawDataset(VisionDataset):
    name = "quickdraw"

    def __init__(self, input_shape, is_train=False, **kwargs):
        self.download_deps = [
            "quickdraw_X.npy", "quickdraw_y.npy"] if is_train else \
            ["quickdraw_X_test.npy", "quickdraw_y_test.npy"]
        self.is_train = is_train
        super().__init__(input_shape, **kwargs)

    def _load_data(self):
        """ Customized _load_data method introduction.

            QuickDrawDataset only support layout of NCHW and the number of channels must be 3, the image height and width must be equal to 32, i.e. (batch_size, 3, 28, 28).
        """
        N, C, H, W = self.ishape
        assert C == 1 and H == 28 and W == 28
        X = nd.array(np.load(path.join(self.root_dir, self.download_deps[0])))
        Y = nd.array(np.load(path.join(self.root_dir, self.download_deps[1])))
        self.data = gluon.data.DataLoader(
                mx.gluon.data.dataset.ArrayDataset(X, Y),
                batch_size=N,
                last_batch='discard',
                shuffle=self.is_train,
                num_workers=4)


@register_dataset("mnist")
class MnistDataset(VisionDataset):
    # name = "mnist"
    # there is no need to download the data from cortexlabs,
    #   since mxnet has supplied the neccesary download logic.
    # download_deps = ["t10k-images-idx3-ubyte.gz",
    #                  "t10k-labels-idx1-ubyte.gz",
    #                  "train-images-idx3-ubyte.gz",
    #                  "train-labels-idx1-ubyte.gz"]

    def data_xform(self, data):
        """Move channel axis to the beginning,
            cast to float32, and normalize to [0, 1].
        """
        return nd.moveaxis(data, 2, 0).astype('float32') / 255

    def _load_data(self):
        """ Customized _load_data method introduction.

            The MxNet gluon package will auto-download the mnist dataset.

            MnistDataset only support layout of NCHW and the number of channels must be 1, the image height and width must be equal to 32, i.e. (batch_size, 1, 28, 28).
        """
        val_data = mx.gluon.data.vision.MNIST(
            root=self.root_dir, train=False).transform_first(
                self.data_xform)

        N, C, H, W = self.ishape
        assert C == 1 and H == 28 and W == 28
        self.data = mx.gluon.data.DataLoader(
            val_data, shuffle=False, batch_size=N)


@register_dataset("trec")
class TrecDataset(Dataset):
    # name = "trec"
    download_deps = ["TREC.train.pk", "TREC.test.pk"]

    def __init__(self, input_shape, is_train=False, **kwargs):
        self.is_train = is_train
        super().__init__(input_shape, **kwargs)

    def _load_data(self):
        """ Customized _load_data method introduction.

            The MxNet gluon package will auto-download the mnist dataset.

            TrecDataset only support layout of (I, N), the image height and width must be equal to 32, i.e. (batch_size, 1, 28, 28).
        """
        fname = path.join(
            self.root_dir, self.download_deps[0] \
            if self.is_train else self.download_deps[1])

        I, N = self.ishape
        assert I == 38

        # (38, batch), (batch,)
        with open(fname, "rb") as fin:
            reader = pickle.load(fin)

        def data_loader():
            data, label = [], []
            for x, y in reader:
                if len(data) < self.ishape[1]:
                    data.append(x)
                    label.append(y)
                else:
                    yield nd.transpose(nd.array(data)), nd.array(label)
                    data, label = [], []
            yield nd.transpose(nd.array(data)), nd.array(label)
            raise RuntimeError("Data loader have been the end")

        self.data = data_loader()

    def metrics(self):
        return {"acc": 0, "total": 0}

    def validate(self, metrics, predict, label):
        """ Customized validate method introduction.

            The score for 6 classes is the model output. The data label is implemented as follows:

            .. code-block:: python

                acc = 1. * metrcs["acc"] / metrics["total"]
        """
        for idx in range(predict.shape[0]):
            res_label = predict[idx].asnumpy().argmax()
            data_label = label[idx].asnumpy()
            if res_label == data_label:
                metrics["acc"] += 1
            metrics["total"] += 1

        acc = 1. * metrics["acc"] / metrics["total"]
        return "{:6.2%}".format(acc)
