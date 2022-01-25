from os import path
import os
import sys

from mxnet import ndarray as nd
import numpy as np
import cv2

from mrt.V3.utils import get_cfg_defaults, merge_cfg, override_cfg_args
from mrt.V3.execute import run
from mrt import dataset as ds
from utils import (
    non_max_suppression, scale_coords, xywh2xyxy, process_batch, ap_per_class)


class Yolov5Metric:
    def __init__(
        self, conf_thres=0.001, iou_thres=0.6, names=None,
        iouv=np.linspace(0.5,0.95,10)):
        # attributes
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.names = names
        self.iouv = iouv
        self.niou = iouv.shape[0]
        # status variable
        self.stats = []
        self.seen = 0

    def reset(self):
        self.stats.clear()
        self.seen = 0

    def update(self, labels, out):
        nl = labels.shape[0]
        out = non_max_suppression(
            out.asnumpy(), self.conf_thres, self.iou_thres, labels=[]
            multi_label=True, agnostic=False)
        pred = out[0]
        tcls = labels[:,0] if nl else []
        self.seen += 1
        if pred.shape[0] == 0:
            if nl:
                self.stats.append(
                    (np.zeros((0)), np.zeros((0)), np.zeros((0)), tcls))
            continue
        predn = pred.copy()
        # native-space pred
        _, _, H, W = self.ishape
        scale_coords((H,W), predn[:,:4], [H,W], [[1.0,1.0],[0.0,0.0]])
        if nl:
            # target boxes
            tbox = xywh2xyxy(labels[:,1:5])
            # native-space labels
            scale_coords((H,W), tbox, [H,W], [[1.0,1.0],[0.0,0.0]])
            # native-space labels
            labelsn = np.concatenate((labels[:,0:1],tbox), axis=1)
            correct = process_batch(predn, labelsn, self.iouv)
        else:
            correct = np.zeros((pred.shape[0], self.niou), dtype=np.bool)
        # (correct, conf, pcls, tcls)
        self.stats.append((correct, pred[:, 4], pred[:, 5], tcls))
        # compute metrics
        # to numpy
        cur_stats = [np.concatenate(x, 0) for x in zip(*self.stats)]
        if len(cur_stats) and cur_stats[0].any():
            tp, fp, p, r, f1, ap, ap_class = ap_per_class(
                *cur_stats, plot=False, save_dir=None, names=names)
            # AP@0.5, AP@0.5:0.95
            ap50, ap = ap[:, 0], ap.mean(1)
            mp, mr, map50, map_ = p.mean(), r.mean(), ap50.mean(), ap.mean()
            # number of targets per class
            nt = np.bincount(cur_stats[3].astype(np.int64), minlength=80)
        else:
            nt = np.zeros(1)
        return self.seen, nt, mp, mr, map50, map_


@ds.register_dataset("yolov5s_dataset")
class Yolov5sDataset(ds.Dataset):
    def __init__(self, input_shape, imgsz=640, **kwargs):
        super().__init__(input_shape, **kwargs)
        self.image_dir = path.join(self.root_dir, "images")
        self.label_dir = path.join(self.root_dir, "labels")
        self.imgsz = imgsz

    def _load_data(self):
        assert len(self.ishape) == 4, self.ishape
        assert self.ishape[0] == 1, self.ishape

        def data_loader():
            for f in os.listdir(self.image_dir):
                _, ext = os.path.splitext(f)
                if ext != ".jpg" and ext != ".JPG" and ext != ".png" and ext != ".PNG":
                    continue
                l = f.replace(f.split(".")[1], "txt")
                file_name = os.path.join(self.root_dir, f)
                label_name = os.path.join(self.label_dir, l)
                img = cv2.imread(file_name)
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
                yield img, labels

        self.data = data_loader()

    def metrics(
        self, conf_thres=0.001, iou_thres=0.6, names=None,
        iouv=np.linspace(0.5,0.95,10)):
        if names is None:
            names = {
                0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle',
                4: 'airplane', 5: 'bus', 6: 'train', 7: 'truck', 8: 'boat',
                9: 'traffic light', 10: 'fire hydrant', 11: 'stop sign',
                12: 'parking meter', 13: 'bench', 14: 'bird', 15: 'cat',
                16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow', 20: 'elephant',
                21: 'bear', 22: 'zebra', 23: 'giraffe', 24: 'backpack',
                25: 'umbrella', 26: 'handbag', 27: 'tie', 28: 'suitcase',
                29: 'frisbee', 30: 'skis', 31: 'snowboard', 32: 'sports ball',
                33: 'kite', 34: 'baseball bat', 35: 'baseball glove',
                36: 'skateboard', 37: 'surfboard', 38: 'tennis racket',
                39: 'bottle', 40: 'wine glass', 41: 'cup', 42: 'fork',
                43: 'knife', 44: 'spoon', 45: 'bowl', 46: 'banana', 47: 'apple',
                48: 'sandwich', 49: 'orange', 50: 'broccoli', 51: 'carrot',
                52: 'hot dog', 53: 'pizza', 54: 'donut', 55: 'cake', 56: 'chair',
                57: 'couch', 58: 'potted plant', 59: 'bed', 60: 'dining table',
                61: 'toilet', 62: 'tv', 63: 'laptop', 64: 'mouse', 65: 'remote',
                66: 'keyboard', 67: 'cell phone', 68: 'microwave', 69: 'oven',
                70: 'toaster', 71: 'sink', 72: 'refrigerator', 73: 'book',
                74: 'clock', 75: 'vase', 76: 'scissors', 77: 'teddy bear',
                78: 'hair drier', 79: 'toothbrush',
            }
        return Yolov5Metric(
            conf_thres=conf_thres, iou_thres=iou_thres, names=names, iouv=iouv)

    def validate(self, metrics, out, labels):
        metrics.update(labels, out)
        seen, nt, mp, mr, map50, map_ = metrics.get()
        return "{}: #images={}, #objects={}, ".fomrat(
            self.root_dir, seen, nt.sum()) + \
            "mp={02.2f}%, mr={02.2f}%, ".format(mp*100, mr*100) + \
            "map50={02.2f}%, map={02.2f}%".format(map50*100, map_*100)

def main(opt):
    print(opt)
    conf_thres = 0.001
    iou_thres = 0.6

    args = parse_opt()
    ctx = mx.cpu() if args.cpu else mx.gpu(args.gpu)

    gw = {"n":1, "s":2, "m":3, "l":4, "x":5}
    gd = {"n":1, "s":1, "m":2, "l":3, "x":4}
    postfix = args.model[-1]
    model = yolov5(batch_size=args.batch_size, mode="val", ctx=ctx, act=args.silu, gd=gd[postfix], gw=gw[postfix])
    model.collect_params().initialize(init=mx.init.Xavier(), ctx=ctx)
    #model.hybridize()

    try:
        EPOCH = []
        start_epoch = 0
        for f in os.listdir(args.model_dir):
            if f.endswith("params") and args.model in f:
                name_epoch = f.strip().split(".")[0].split("-")
                if len(name_epoch) == 2 and name_epoch[0] == args.model:
                    EPOCH.append(name_epoch[1])
        tmp = [int(_) for _ in EPOCH]
        ind = tmp.index(max(tmp))
        params_file = os.path.join(args.model_dir, args.model+"-"+EPOCH[ind]+".params")
        model.collect_params().load(params_file,ignore_extra=False)
        print(f'load weight {params_file} successfully')
    except:
        print("failed to load weight")

    iouv = np.linspace(0.5, 0.95, 10)
    niou = iouv.shape[0]
    seen = 0
    jdict, stats, ap, ap_class = [], [], [], []

    for f in os.listdir(args.dataset):
        _, ext = os.path.splitext(f)
        if ext != ".jpg" and ext != ".JPG" and ext != ".png" and ext != ".PNG":
            continue
        print(f)
        l = f.replace(f.split(".")[1], "txt")
        file_name = os.path.join(args.dataset, f)
        label_name = os.path.join(args.dataset.replace("images","labels"), l)
        img = cv2.imread(file_name)
        try:
            labels = np.loadtxt(label_name)
        except:
            labels = np.array([])
        labels = labels.reshape((-1, 5))
        
        height, width = img.shape[0:2]
        scale = min(args.imgsz/height, args.imgsz/width)
        h0, w0 = height*scale, width*scale
        img0 = cv2.resize(img, (round(w0/32.)*32, round(h0/32.)*32))

        img = img0.astype("float32")/255.
        img = nd.array(img.transpose((2,0,1))[None], ctx = ctx)
        labels[:,1:] = labels[:,1:]*np.array([img.shape[3], img.shape[2]]*2)

        nl = labels.shape[0]
        out = model(img).asnumpy()
        out = non_max_suppression(out, conf_thres, iou_thres, labels=[], multi_label=True, agnostic=False)
        pred = out[0]

        tcls = labels[:,0] if nl else []
        seen += 1
        
        if pred.shape[0] == 0:
            if nl:
                stats.append((np.zeros((0)), np.zeros((0)), np.zeros((0)), tcls))
            continue
        
        predn = pred.copy()
        scale_coords(img[0].shape[1:], predn[:, :4], [img.shape[2], img.shape[3]], [[1.0,1.0],[0.0,0.0]])  # native-space pred

        if nl:
            tbox = xywh2xyxy(labels[:, 1:5])  # target boxes
            scale_coords(img[0].shape[1:], tbox, [img.shape[2], img.shape[3]], [[1.0,1.0],[0.0,0.0]])  # native-space labels
            labelsn = np.concatenate((labels[:, 0:1], tbox), axis=1)  # native-space labels
            correct = process_batch(predn, labelsn, iouv)
        else:
            correct = np.zeros((pred.shape[0], niou), dtype=np.bool)
        stats.append((correct, pred[:, 4], pred[:, 5], tcls))  # (correct, conf, pcls, tcls)


if __name__ == "__main__":
    assert len(sys.argv) >= 1 and len(sys.argv)%2 == 1, \
        "invalid length: {} of sys.argv: {}".format(
        len(sys.argv), sys.argv)
    yaml_file = path.join(
        path.dirname(path.realpath(__file__)),
        "model_zoo", "prediction_SCTF.yaml")
    cfg = get_cfg_defaults()
    cfg = merge_cfg(yaml_file)
    cfg = override_cfg_args(cfg, sys.argv[1:])
    run(cfg)
