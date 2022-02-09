from os import path
import os

import mxnet as mx
from mxnet import ndarray as nd
import numpy as np
import cv2

from mrt import dataset as ds
from utils import (
    non_max_suppression, scale_coords, xywh2xyxy, process_batch, ap_per_class)


class Yolov5Metric:
    def __init__(
        self, conf_thres=0.001, iou_thres=0.6, iouv=np.linspace(0.5,0.95,10),
        nc=80, anchors=()):

        # metric parameters
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.iouv = iouv
        self.niou = iouv.shape[0]
        self.names = {
            0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane',
            5: 'bus', 6: 'train', 7: 'truck', 8: 'boat', 9: 'traffic light',
            10: 'fire hydrant', 11: 'stop sign', 12: 'parking meter',
            13: 'bench', 14: 'bird', 15: 'cat', 16: 'dog', 17: 'horse',
            18: 'sheep', 19: 'cow', 20: 'elephant', 21: 'bear', 22: 'zebra',
            23: 'giraffe', 24: 'backpack', 25: 'umbrella', 26: 'handbag',
            27: 'tie', 28: 'suitcase', 29: 'frisbee', 30: 'skis',
            31: 'snowboard', 32: 'sports ball', 33: 'kite', 34: 'baseball bat',
            35: 'baseball glove', 36: 'skateboard', 37: 'surfboard',
            38: 'tennis racket', 39: 'bottle', 40: 'wine glass', 41: 'cup',
            42: 'fork', 43: 'knife', 44: 'spoon', 45: 'bowl', 46: 'banana',
            47: 'apple', 48: 'sandwich', 49: 'orange', 50: 'broccoli',
            51: 'carrot', 52: 'hot dog', 53: 'pizza', 54: 'donut', 55: 'cake',
            56: 'chair', 57: 'couch', 58: 'potted plant', 59: 'bed',
            60: 'dining table', 61: 'toilet', 62: 'tv', 63: 'laptop',
            64: 'mouse', 65: 'remote', 66: 'keyboard', 67: 'cell phone',
            68: 'microwave', 69: 'oven', 70: 'toaster', 71: 'sink',
            72: 'refrigerator', 73: 'book', 74: 'clock', 75: 'vase',
            76: 'scissors', 77: 'teddy bear', 78: 'hair drier',
            79: 'toothbrush',
        }

        # detect parameters
        self.no = nc + 5
        self.na = len(anchors[0]) // 2
        self.stride = nd.array([8., 16., 32.])
        self.anchors = nd.array(
            [
                [
                    [ 1.25000,  1.62500],
                    [ 2.00000,  3.75000],
                    [ 4.12500,  2.87500]
                ],
                [
                    [ 1.87500,  3.81250],
                    [ 3.87500,  2.81250],
                    [ 3.68750,  7.43750]
                ],
                [
                    [ 3.62500,  2.81250],
                    [ 4.87500,  6.18750],
                    [11.65625, 10.18750]
                ]
            ]
        )

        # status variable
        self.stats = []

    def reset(self):
        self.stats.clear()

    def _make_grid(self, nx=20, ny=20, i=0, ctx=mx.cpu(0)):
        yv = nd.array(range(ny))[:,None].repeat(nx,axis=1)
        xv = nd.array(range(nx))[None,:].repeat(ny,axis=0)
        grid = nd.concat(
            xv[...,None], yv[...,None], dim=2)[None,None,...].repeat(
            self.na, axis=1)
        grid = nd.Cast(grid, dtype="float32")

        anchor_grid = (self.anchors[i].copy()*self.stride[i])
        anchor_grid = anchor_grid[None,:, None, None,:]
        anchor_grid = anchor_grid.repeat(ny, axis=-3)
        anchor_grid = anchor_grid.repeat(nx, axis=-2)
        return grid.as_in_context(ctx), anchor_grid.as_in_context(ctx)

    def update(self, labels, predict, input_shape):
        batch_size, _, H, W = input_shape
        outs = []
        for i in range(batch_size):
            x, y, z = [o.slice_axis(axis=0, begin=i, end=i+1) for o in predict]
            out = []

            bs, _, ny, nx, _ = x.shape
            grid, anchor_grid = self._make_grid(nx, ny, 0, ctx=x.ctx)
            tmp = x.sigmoid()
            # xy
            xy = (tmp[..., 0:2]*2-0.5+grid) * \
                self.stride[0].as_in_context(x.ctx)
            # wh
            wh = (tmp[..., 2:4]*2)**2 * anchor_grid
            tmp = nd.concat(xy, wh, tmp[..., 4:], dim=-1)
            out.append(tmp.reshape(bs, -1, self.no))

            bs, _, ny, nx, _ = y.shape
            grid, anchor_grid = self._make_grid(nx, ny, 1, ctx=y.ctx)
            tmp = y.sigmoid()
            # xy
            xy = (tmp[..., 0:2]*2-0.5+grid) * \
                self.stride[1].as_in_context(y.ctx)
            # wh
            wh = (tmp[..., 2:4]*2)*2 * anchor_grid
            tmp = nd.concat(xy, wh, tmp[..., 4:], dim=-1)
            out.append(tmp.reshape(bs, -1, self.no))

            bs, _, ny, nx, _ = z.shape
            grid, anchor_grid = self._make_grid(nx, ny, 2, ctx=z.ctx)
            tmp = z.sigmoid()
            # xy
            xy = (tmp[..., 0:2]*2-0.5+grid) * \
                self.stride[2].as_in_context(z.ctx)
            # wh
            wh = (tmp[..., 2:4]*2)**2 * anchor_grid
            tmp = nd.concat(xy, wh, tmp[..., 4:], dim=-1)
            out.append(tmp.reshape(bs, -1, self.no))

            out = nd.concat(*out, dim=1)
            outs.append(out)
        for i in range(batch_size):
            label = labels[i]
            nl = label.shape[0]
            out = non_max_suppression(
                outs[i].asnumpy(), self.conf_thres, self.iou_thres, labels=[],
                multi_label=True, agnostic=False)
            pred = out[0]
            tcls = label[:,0] if nl else []
            if pred.shape[0] == 0:
                if nl:
                    self.stats.append(
                        (np.zeros((0,self.niou)), np.zeros((0)), np.zeros((0)), tcls))
                continue
            predn = pred.copy()
            # native-space pred
            scale_coords((H,W), predn[:,:4], [H,W], [[1.0,1.0],[0.0,0.0]])
            if nl:
                # target boxes
                tbox = xywh2xyxy(label[:,1:5])
                # native-space label
                scale_coords((H,W), tbox, [H,W], [[1.0,1.0],[0.0,0.0]])
                # native-space label
                labelsn = np.concatenate((label[:,0:1],tbox), axis=1)
                correct = process_batch(predn, labelsn, self.iouv)
            else:
                correct = np.zeros((pred.shape[0], self.niou), dtype=np.bool)
            # (correct, conf, pcls, tcls)
            self.stats.append((correct, pred[:, 4], pred[:, 5], tcls))

    def get(self):
        # compute metrics
        # to numpy
        cur_stats = [np.concatenate(x, 0) for x in zip(*self.stats)]
        if len(cur_stats) and cur_stats[0].any():
            tp, fp, p, r, f1, ap, ap_class = ap_per_class(
                *cur_stats, plot=False, save_dir=None, names=self.names)
            # AP@0.5, AP@0.5:0.95
            ap50, ap = ap[:, 0], ap.mean(1)
            mp, mr, map50, map_ = p.mean(), r.mean(), ap50.mean(), ap.mean()
            # number of targets per class
            nt = np.bincount(cur_stats[3].astype(np.int64), minlength=80)
        else:
            nt = np.zeros(1)
            mp = mr = map50 = map_ = 0.
        return nt, mp, mr, map50, map_


@ds.register_dataset("yolov5_dataset")
class Yolov5Dataset(ds.Dataset):
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
                label_name = os.path.join(self.label_dir, l)
                img = cv2.imread(file_name)
                # hack size
                img = cv2.resize(img, tuple(self.ishape[2:]))
                try:
                    labels = np.loadtxt(label_name)
                except:
                    labels = np.array([])
                labels = labels.reshape((-1, 5))
                height, width = img.shape[0:2]
                scale = min(self.imgsz/height, self.imgsz/width)
                h0, w0 = height*scale, width*scale
                img0 = cv2.resize(img, (round(w0/32.)*32, round(h0/32.)*32))
                img = img0.astype("float32")/255.
                img = nd.array(img.transpose((2,0,1))[None])
                labels[:,1:] = labels[:,1:] * np.array([img.shape[3], img.shape[2]]*2)
                # if img.shape[2] != self.ishape[2] or img.shape[3] != self.ishape[3]:
                    # continue
                if len(data) == batch_size:
                    batch_data = nd.concatenate(data)
                    yield batch_data, label
                    data, label = [], []
                data.append(img)
                label.append(labels)
            if len(data) == batch_size:
                batch_data = nd.concatenate(data)
                yield batch_data, label

        self.data = data_loader()

    def metrics(
        self, conf_thres=0.001, iou_thres=0.6, iouv=np.linspace(0.5,0.95,10)):
        anchors = [
            [10, 13, 16, 30, 33, 23],
            [30 ,61, 62 ,45, 59, 119],
            [116, 90, 156, 198, 373, 326]
        ]
        metric = Yolov5Metric(
            conf_thres=conf_thres, iou_thres=iou_thres, iouv=iouv,
            anchors=anchors, nc=80)
        metric.reset()
        return metric

    def validate(self, metrics, out, labels):
        metrics.update(labels, out, self.ishape)
        nt, mp, mr, map50, map_ = metrics.get()
        return "#objects={}, ".format(nt.sum()) + \
            "mp={:6.2%}, mr={:6.2%}, ".format(mp, mr) + \
            "map50={:6.2%}, map={:6.2%}".format(map50, map_)
