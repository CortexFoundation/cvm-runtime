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

        The dataset directory is located at the `root` directory containing
            the dataset `name` directory. And the custom dataset should pass
            the parameter location of root, or implement the derived class
            of your data iterator, metrics and validate function.

        Notice:
        =======
            our default imagenet dataset is organized as an `record`
            binary format, which can amplify the throughput for image read.
            Custom dataset of third party should be preprocessed by the `im2rec`
            procedure to transform the image into the record format.
            The transformation script is located at `docs/mrt/im2rec.py`. And more
            details refer to the script helper documentation please(print usage
            with command `-h`).


        Parameters:
        ===========
            input_shape: the input shape requested from user, and some dataset would
                check the validity format.
            root: the location where dataset is stored, defined with variable `MRT_DATASET_ROOT`
                in conf.py or custom directory.


        Derived Class Implementation:
        =============================
            1. register dataset name into DS_REG that can be accessed
                at the `dataset` package API. releated function is
                `register_dataset`.

            2. override the abstract method defined in base dataset class,
                _load_data[Required]:
                    load data from disk that stored into the data variable.
                iter_func[Optional]:
                    return the tuple (data, label) for each invocation.
                metrics[Required]:
                    returns the metrics object for the dataset.
                validate[Required];
                    calculates the accuracy for model inference of string.

    """
    name = None

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

    def validate(self, metrics, predicts, labels):
        raise NotImplementedError(
                "Derived " + self.name + " dataset not override the" +
                " base `validate` function defined in Dataset")

    def _load_data(self):
        raise NotImplementedError(
                "Derived " + self.name + " dataset not override the" +
                " base `_load_data` function defined in Dataset")

    def __iter__(self):
        """ Returns (data, label) iterator """
        return iter(self.data)

    def iter_func(self):
        """ Returns (data, label) iterator function """
        data_iter = iter(self.data)
        def _wrapper():
            return next(data_iter)
        return _wrapper


@register_dataset("coco")
class COCODataset(Dataset):
    # download_deps = ['val2017.zip']

    def _load_data(self):
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
        _, _, H, W = self.ishape
        metric = COCODetectionMetric(
            self.val_dataset, '_eval', cleanup=True, data_shape=(H, W))
        metric.reset()
        return metric

    def validate(self, metrics, predict, label):
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
        metric = VOC07MApMetric(
            iou_thresh=0.5, class_names=gdata.VOCDetection.CLASSES)
        metric.reset()
        return metric

    def validate(self, metrics, predict, label):
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
        return [mx.metric.Accuracy(),
                mx.metric.TopKAccuracy(5)]

    def validate(self, metrics, predict, label):
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

    def _load_data(self):
        """
            The MxNet gluon package will auto-download the mnist dataset.
        """
        val_data = mx.gluon.data.vision.MNIST(
            root=self.root_dir, train=False).transform_first(data_xform)

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
        fname = path.join(
            self.root_dir, self.download_deps[0] \
            if self.is_train else self.download_deps[1])

        # (38, batch), (batch,)
        with open(fname, "rb") as fin:
            self.data = pickle.load(fin)

        I, N = self.ishape
        assert I == 38

    def iter_func(self):
        def _wrapper():
            data, label = [], []
            for x, y in self.data:
                if len(data) < self.ishape[1]:
                    data.append(x)
                    label.append(y)
                else:
                    return nd.transpose(nd.array(data)), nd.array(label)
            return nd.transpose(nd.array(data)), nd.array(label)
        return _wrapper

    def metrics(self):
        return {"acc": 0, "total": 0}

    def validate(self, metrics, predict, label):
        for idx in range(predict.shape[0]):
            res_label = predict[idx].asnumpy().argmax()
            data_label = label[idx].asnumpy()
            if res_label == data_label:
                metrics["acc"] += 1
            metrics["total"] += 1

        acc = 1. * metrics["acc"] / metrics["total"]
        return "{:6.2%}".format(acc)
