from os import path
import os

import mxnet as mx
from mxnet import ndarray as nd
import numpy as np
import cv2

from mrt import dataset as ds
from utils import (
    non_max_suppression, scale_coords, Annotator, concat_out, make_squre, Colors)


class Yolov5MetricV2:
    def __init__(self):
        self.conf_thres = 0.25
        self.iou_thres = 0.45
        self.names = ['person', 'bicycle', 'car', 'motorcycle', 'airplane',
            'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
            'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',
            'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe',
            'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
            'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat',
            'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
            'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
            'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
            'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
            'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop',
            'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven',
            'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
            'scissors', 'teddy bear', 'hair drier', 'toothbrush']
        self.results = []

    def reset(self):
        self.results.clear()

    def update(self, labels, predict, input_shape):
        batch_size, _, H, W = input_shape
        outs = []
        for i in range(batch_size):
            x, y, z = [o.slice_axis(axis=0, begin=i, end=i+1) for o in predict]
            out = concat_out(x, y, z).asnumpy()
            outs.append(out)
        for i in range(batch_size):
            out = non_max_suppression(
                outs[i], self.conf_thres, self.iou_thres, labels=[],
                multi_label=True, agnostic=False)
            f, img0s = labels[i]

            annotator = Annotator(img0s, line_width=1, example=str(self.names))

            pred = out[0]
            if pred.shape[0] > 0:
                pred[:, :4] = scale_coords(
                    (H,W), pred[:, :4], img0s.shape).round()

            for *xyxy, conf, cls in reversed(pred):
                c = int(cls)
                label = f'{self.names[c]} {conf:.2f}'
                annotator.box_label(xyxy, label, color=Colors()(c, True))

            img0s = annotator.result()
            self.results.append((f,img0s))

    def get(self):
        return self.results
        self.results.clear()


@ds.register_dataset("yolov5_dataset_v2")
class Yolov5DatasetV2(ds.Dataset):
    def __init__(self, input_shape, imgsz=640, **kwargs):
        super().__init__(input_shape, **kwargs)
        self.image_dir = path.join(self.root_dir, "images")
        self.label_dir = path.join(self.root_dir, "labels")
        self.imgsz = imgsz

    def _load_data(self):
        assert len(self.ishape) == 4, self.ishape
        batch_size = self.ishape[0]
        assert batch_size == 16, batch_size

        def data_loader():
            data, label = [], []
            for f in sorted(os.listdir(self.image_dir)):
                _, ext = os.path.splitext(f)
                if ext != ".jpg" and ext != ".JPG" \
                    and ext != ".png" and ext != ".PNG":
                    continue
                l = f.replace(f.split(".")[1], "txt")
                file_name = os.path.join(self.image_dir, f)
                _, _, _, _, img = make_squre(cv2.imread(file_name))
                img = cv2.resize(img, tuple(self.ishape[2:]))
                img0s = img.copy()
                img = img.astype("float32")/255.
                img = nd.array(img.transpose((2,0,1))[None])
                if len(data) == batch_size:
                    batch_data = nd.concatenate(data)
                    yield batch_data, label
                    data, label = [], []
                data.append(img)
                label.append((f,img0s))
            if len(data) == batch_size:
                batch_data = nd.concatenate(data)
                yield batch_data, label

        self.data = data_loader()

    def metrics(self):
        metric = Yolov5MetricV2()
        metric.reset()
        return metric

    def validate(self, metrics, out, labels):
        metrics.update(labels, out, self.ishape)
        img0s_batch = metrics.get()
        return img0s_batch
