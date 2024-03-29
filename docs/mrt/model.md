# Pre-quantized Model Introduction

## Model Performance

Some models have been compiled and the Accuracy and size of params is provided in the following chart for reference.

| Model Name                | Category       | (Top1) Accuracy<br />(original / quantized / diff) |
| ------------------------- | -------------- | ------------------------------------------- |
| ssd_512_mobilenet1.0_coco | detection      | 21.50% / 15.60% / 5.9%                      |
| ssd_512_resnet50_v1_voc   | detection      | 80.27% / 80.01% / 0.26%                     |
| yolo3_darknet53_voc       | detection      | 81.37% / 82.08% / -0.71%                    |
| shufflenet_v1             | classification | 63.48% / 60.45% / 3.03%                     |
| mobilenet1_0              | classification | 70.77% / 66.11% / 4.66%                     |
| mobilenetv2_1.0           | classification | 71.51% / 69.39% / 2.12%                     |

| Model Name                | Params Size | Path                                    |
| ------------------------- | ----------- | --------------------------------------- |
| ssd_512_mobilenet1.0_coco | 23.2M       | /data/mrt/ssd_512_mobilenet1.0_coco_tfm |
| ssd_512_resnet50_v1_voc   | 36.4M       | /data/mrt/ssd_512_resnet50_v1_voc_tfm   |
| yolo3_darknet53_voc       | 59.3M       | /data/mrt/yolo3_darknet53_voc_tfm       |
| shufflenet_v1             | 1.8M        | /data/mrt/shufflenet_v1_tfm             |
| mobilenet1_0              | 4.1M        | /data/mrt/mobilenet1_0_tfm              |
| mobilenetv2_1.0           | 3.4M        | /data/mrt/mobilenetv2_1.0_tfm           |


## Model Preprocess

The data preprocess functions and input shapes are collected with respect to the dataset label in the following chart. for reference.

| Dataset Label | Data Preprocess Function                                     | Input Shape Format                      |
| ------------- | ------------------------------------------------------------ | --------------------------------------- |
| voc           | YOLO3DefaultValTransform(input_size, input_size)             | (batch_size, 3, input_size, input_size) |
| imagenet      | crop_ratio = 0.875<br />resize = $\lceil H/crop\_ratio \rceil$<br />mean_rgb = [123.68, 116.779, 103.939]<br />std_rgb = [58.393, 57.12, 57.375] | (batch_size, 3, input_size, input_size) |
| cifar10       | mean = [0.4914, 0.4822, 0.4465]<br />std = [0.2023, 0.1994, 0.2010] | (batch_size, 3, 32, 32)                 |
| quickdraw     | -                                                            | (batch_size, 1, 28, 28)                 |
| mnist         | mean = 0<br />std = 1                                        | (batch_size, 1, 28, 28)                 |
| trec          | -                                                            | (38, batch_size)                        |
| coco          | SSDDefaultValTransform(input_size, input_size)               | (batch_size, 3, input_size, input_size) |



## Model Output

Some model output introduction is concluded in the following chart.

| Model Type                                                   | Model Output Introduction                                    |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| ssd, yolo                                                    | [id, score, bounding_box]<br />bounding_box = (x1, y1, x2, y2) |
| mobilenet, rennet, shufflenet,<br />densenet,alexnet, squeezenet, vgg | score for 1000 classes                                       |
| cifar, quickdraw, mnist                                      | score for 10 classes                                         |
| trec                                                         | score for 6 classes                                          |



Some dataset might need a particular output index to extract the actual value of result which is also enumerated in the following chart.

| Dataset Label | Output Index Converting                                      |
| ------------- | ------------------------------------------------------------ |
| voc           | `map_name, mean_ap = metrics.get()`<br />`acc = {k: v for k,v in zip(map_name, mean_ap)}['mAP']` |
| trec          | `acc = 1. * metrcs["acc"] / metrics["total"]`                |
| coco          | `map_name, mean_ap = metrics.get()`<br />`acc = {k: v for k,v in zip(map_name, mean_ap)}`<br />`acc = float(acc['~~~~ MeanAP @ IoU=[0.50, 0.95] ~~~~\n']) / 100` |



## Model Testing

Some models have been tested and the precision before and after quantization is provided in the following chart for reference.

| model name               | Iteration | evalfunc                     | quantize                     | total sample |
| ------------------------ | --------- | ---------------------------- | ---------------------------- | ------------ |
| resnet50_v1              | 312       | top1=77.39%<br />top5=93.59% | top1=76.47%<br />top5=93.28% | 50080        |
| resnet50_v2              | 312       | top1=77.15%<br />top5=93.44% | top1=70.76%<br />top5=89.56% | 50080        |
| resnet18_v1              | 312       | top1=70.96%<br />top5=89.93% | top1=70.11%<br />top5=89.60% | 50080        |
| resnet18v1_b_0.89        | 312       | top1=67.21%<br />top5=87.45% | top1=63.75%<br />top5=85.63% | 50080        |
| quickdraw_wlt            | 349       | top1=81.90%<br />top5=98.26% | top1=81.83%<br />top5=98.24% | 56000        |
| qd10_resnetv1_20         | 349       | top1=85.79%<br />top5=98.73% | top1=85.79%<br />top5=98.73% | 56000        |
| densenet161              | 312       | top1=77.62%<br />top5=93.82% | top1=77.32%<br />top5=93.63% | 50080        |
| alexnet                  | 312       | top1=55.91%<br />top5=78.75% | top1=51.69%<br />top5=77.99% | 50080        |
| cifar_resnet20_v1        | 62        | top1=92.88%<br />top5=99.78% | top1=92.82%<br />top5=99.75% | 10000        |
| mobilenet1_0             | 312       | top1=70.77%<br />top5=89.97% | top1=66.11%<br />top5=87.35% | 50080        |
| mobilenetv2_1.0          | 312       | top1=71.51%<br />top5=90.10% | top1=69.39%<br />top5=89.30% | 50080        |
| shufflenet_v1            | 312       | top1=63.48%<br />top5=85.12% | top1=60.45%<br />top5=82.95% | 50080        |
| squeezenet1.0            | 312       | top1=57.20%<br />top5=80.04% | top1=55.16%<br />top5=78.67% | 50080        |
| tf_inception_v3          | 312       | top1=55.58%<br />top5=77.56% | top1=55.54%<br />top5=83.03% | 50080        |
| vgg19                    | 312       | top1=74.14%<br />top5=91.78% | top1=73.75%<br />top5=91.67% | 50080        |
| trec                     | 28        | 97.84%                       | 97.63%                       | 1102         |
| yolo3_darknet53_voc      | 29        | 81.37%                       | 82.08%                       | 4800         |
| yolo3_mobilenet1.0_voc   | 29        | 75.98%                       | 71.53%                       | 4800         |
| ssd_512_resnet50_v1_voc  | 29        | 80.27%                       | 80.01%                       | 4800         |
| ssd_512_mobilenet1.0_voc | 29        | 75.57%                       | 71.32%                       | 4800         |
| mnist                    | 62        | top1=99.18%<br />top5=100%   | top1=99.17%<br />top5=100%   | 10000        |


| model name               | Iteration | evalfunc                     | quantize                     | time(ms)  | total sample | cvm  (cpu&gpu)    |
| ------------------------ | --------- | ---------------------------- | ---------------------------- | --------- | ------------ | ----------------- |
| cifar_resnet20_v1        | 62        | top1=92.88%<br />top5=99.78% | top1=92.82%<br />top5=99.75% | 135 211   | 10080        | 92.80% 12ms  3ms  |
| cifar_resnet20_v2        | 62        | top1=92.68%<br />top5=99.77% | top1=92.39%<br />top5=99.77% | 122 209   | 10080        | 92.39% 14ms  3ms  |
| cifar_resnet56_v1        | 62        | top1=94.20%<br />top5=99.83% | top1=94.16%<br />top5=99.82% | 122 379   | 10080        | 94.21% 26ms  5ms  |
| cifar_resnet56_v2        | 62        | top1=94.64%<br />top5=99.89% | top1=94.48%<br />top5=99.87% | 90 300    | 10080        | 94.48% 33ms  6ms  |
| cifar_resnet110_v1       | 62        | top1=95.20%<br />top5=99.83% | top1=95.12%<br />top5=99.83% | 117 410   | 10080        | 95.09% 45ms  9ms  |
| cifar_resnet110_v2       | 62        | top1=95.54%<br />top5=99.84% | top1=95.31%<br />top5=99.82% | 109 707   | 10080        | 95.26% 58ms  10ms |
| cifar_wideresnet16_10    | 62        | top1=96.71%<br />top5=99.91% | top1=96.59%<br />top5=99.91% | 851 887   | 10080        | 96.58% 112ms 7ms  |
| cifar_wideresnet28_10    | 62        | top1=97.12%<br />top5=99.92% | top1=96.99%<br />top5=99.89% | 876 1016  | 10080        | 96.99% 204ms 14ms |
| cifar_wideresnet40_8     | 62        | top1=97.27%<br />top5=99.92% | top1=97.03%<br />top5=99.93% | 636 904   | 10080        | 97.04% 218ms 16ms |
| alexnet                  | 312       | top1=55.89%<br />top5=78.75% | top1=51.48%<br />top5=77.39% | 25 114    | 50080        | 51.46% 69ms  7ms  |
| densenet121              | 999       | top1=74.94%<br />top5=92.17% | top1=72.67%<br />top5=91.04% | 20 848    | 16000        | 72.72% 473ms 29ms |
| densenet161              | 312       | top1=77.62%<br />top5=93.82% | top1=77.27%<br />top5=93.60% | 425 3691  | 50080        | 77.26% 898ms 58ms |
| densenet169              | 999       | top1=76.29%<br />top5=92.98% | top1=75.63%<br />top5=92.77% | 27 851    | 16000        | 75.63% 591ms 41ms |
| densenet201              | 999       | top1=77.39%<br />top5=93.37% | top1=73.96%<br />top5=91.88% | 40 1152   | 16000        | 73.96% 756ms 53ms |
| mobilenet0.5             | 312       | top1=65.22%<br />top5=86.35% | top1=59.70%<br />top5=82.46% | 34 218    | 50080        | 59.69% 50ms  8ms  |
| mobilenet0.75            | 312       | top1=70.27%<br />top5=89.49% | top1=66.25%<br />top5=87.06% | 48 261    | 50080        | 66.25% 77ms  8ms  |
| mobilenet1_0             | 312       | top1=73.29%<br />top5=91.30% | top1=64.64%<br />top5=88.34% | 58 330    | 50080        | 64.65% 99ms  8ms  |
| mobilenetv2_0.5          | 312       | top1=64.43%<br />top5=85.32% | top1=61.65%<br />top5=83.60% | 39 286    | 50080        | 61.65% 66ms  7ms  |
| mobilenetv2_0.75         | 312       | top1=69.38%<br />top5=88.50% | top1=65.50%<br />top5=86.05% | 53 331    | 50080        | 65.49% 92ms  7ms  |
| mobilenetv2_1.0          | 312       | top1=72.05%<br />top5=90.58% | top1=69.79%<br />top5=89.14% | 68 409    | 50080        | 69.78% 117ms 7ms  |
| resnet18_v1              | 312       | top1=70.96%<br />top5=89.93% | top1=70.11%<br />top5=89.61% | 54 276    | 50080        | 70.11% 102ms 8ms  |
| resnet18_v1b_0.89        | 312       | top1=67.20%<br />top5=87.45% | top1=63.78%<br />top5=85.62% | 33 193    | 50080        | 63.77% 61ms  7ms  |
| resnet18_v1b             | 312       | top1=70.95%<br />top5=89.85% | top1=69.39%<br />top5=89.52% | 56 271    | 50080        | 69.38% 104ms 8ms  |
| resnet34_v1              | 312       | top1=74.39%<br />top5=91.88% | top1=73.75%<br />top5=91.71% | 87 486    | 50080        | 73.75% 181ms 13ms |
| resnet34_v1b             | 312       | top1=74.67%<br />top5=92.08% | top1=74.17%<br />top5=91.85% | 88 389    | 50080        | 74.17% 181ms 13ms |
| resnet34_v2              | 312       | top1=74.43%<br />top5=92.11% | top1=73.46%<br />top5=91.55% | 92 559    | 50080        | 73.45% 244ms 15ms |
| resnet50_v1              | 312       | top1=77.39%<br />top5=93.59% | top1=76.45%<br />top5=93.27% | 168 813   | 50080        | 76.43% 250ms 12ms |
| resnet50_v1b             | 312       | top1=77.69%<br />top5=93.83% | top1=76.55%<br />top5=93.35% | 161 867   | 50080        | 76.54% 273ms 12ms |
| resnet50_v1c             | 312       | top1=78.05%<br />top5=94.11% | top1=77.48%<br />top5=93.80% | 173 911   | 50080        | 77.47% 292ms 13ms |
| resnet50_v1d_0.11        | 312       | top1=63.22%<br />top5=84.79% | top1=61.14%<br />top5=83.40% | 30 324    | 50080        | 61.21% 47ms  7ms  |
| resnet50_v1d_0.37        | 312       | top1=70.72%<br />top5=89.75% | top1=68.08%<br />top5=88.12% | 36 318    | 50080        | 67.95% 73ms  7ms  |
| resnet50_v1d_0.48        | 312       | top1=74.68%<br />top5=92.35% | top1=72.26%<br />top5=91.19% | 54 428    | 50080        | 72.31% 108ms 9ms  |
| resnet50_v1d_0.86        | 312       | top1=78.03%<br />top5=93.83% | top1=76.42%<br />top5=93.32% | 105 582   | 50080        | 76.37% 176ms 11ms |
| resnet50_v1d             | 312       | top1=79.19%<br />top5=94.59% | top1=78.26%<br />top5=94.22% | 170 986   | 50080        | 78.25% 285ms 13ms |
| resnet50_v1s             | 312       | top1=78.89%<br />top5=94.36% | top1=78.48%<br />top5=94.18% | 200 936   | 50080        | 78.47% 352ms 14ms |
| resnet50_v2              | 312       | top1=77.15%<br />top5=93.44% | top1=74.15%<br />top5=91.74% | 171 1339  | 50080        | 75.13% 415ms 18ms |
| resnet101_v1             | 312       | top1=78.36%<br />top5=94.01% | top1=77.62%<br />top5=93.64% | 277 1334  | 50080        | 77.62% 430ms 20ms |
| resnet101_v1b            | 312       | top1=79.23%<br />top5=94.62% | top1=77.68%<br />top5=94.00% | 263 1378  | 50080        | 77.68% 444ms 20ms |
| resnet101_v1c            | 312       | top1=79.62%<br />top5=94.77% | top1=78.14%<br />top5=94.25% | 273 1432  | 50080        | 78.14% 479ms 21ms |
| resnet101_v1d_0.73       | 312       | top1=78.92%<br />top5=94.49% | top1=74.32%<br />top5=92.97% | 138 841   | 50080        | 74.31% 295ms 18ms |
| resnet101_v1d_0.76       | 312       | top1=79.48%<br />top5=94.70% | top1=75.70%<br />top5=93.71% | 153 878   | 50080        | 75.70% 310ms 18ms |
| resnet101_v1d            | 312       | top1=80.55%<br />top5=95.13% | top1=77.49%<br />top5=94.44% | 272 1513  | 50080        | 77.48% 489ms 21ms |
| resnet101_v1s            | 312       | top1=80.34%<br />top5=95.25% | top1=79.75%<br />top5=95.06% | 302 1530  | 50080        | 79.74% 519ms 23ms |
| resnet152_v1             | 312       | top1=79.25%<br />top5=94.65% | top1=78.56%<br />top5=94.38% | 417 1964  | 50080        | 78.54% 641ms 27ms |
| resnet152_v1b            | 312       | top1=79.70%<br />top5=94.75% | top1=78.77%<br />top5=94.36% | 373 2007  | 50080        | 78.76% 650ms 28ms |
| resnet152_v1c            | 312       | top1=80.05%<br />top5=94.96% | top1=79.22%<br />top5=94.66% | 381 2057  | 50080        | 79.21% 714ms 29ms |
| resnet152_v1d            | 312       | top1=80.63%<br />top5=95.36% | top1=78.76%<br />top5=94.84% | 383 2058  | 50080        | 78.76% 675ms 29ms |
| resnet152_v1s            | 312       | top1=81.13%<br />top5=95.55% | top1=80.81%<br />top5=95.38% | 410 2117  | 50080        | 80.81% 716ms 32ms |
| resnet152_v2             | 312       | top1=79.26%<br />top5=94.66% | top1=78.56%<br />top5=94.38% | 397 2126  | 50080        | 78.45% 1012ms 41ms|
| squeezenet1.0            | 312       | top1=57.19%<br />top5=80.04% | top1=54.91%<br />top5=78.64% | 70 331    | 50080        | 55.38% 89ms  8ms  |
| squeezenet1.1            | 312       | top1=56.96%<br />top5=79.77% | top1=52.62%<br />top5=78.04% | 43 259    | 50080        | 52.61% 55ms  8ms  |
| vgg11_bn                 | 3124      | top1=69.66%<br />top5=89.43% | top1=67.99%<br />top5=88.46% | 18 60     | 50000        | 67.98% 299ms 11ms |
| vgg11                    | 3124      | top1=68.10%<br />top5=88.25% | top1=66.37%<br />top5=87.69% | 16 60     | 50000        | 66.36% 315ms 11ms |
| vgg13_bn                 | 3124      | top1=70.52%<br />top5=89.84% | top1=68.75%<br />top5=88.82% | 27 80     | 50000        | 68.74% 433ms 13ms |
| vgg13                    | 3124      | top1=68.94%<br />top5=88.88% | top1=68.47%<br />top5=88.66% | 24 82     | 50000        | 68.46% 452ms 13ms |
| vgg16_bn                 | 3124      | top1=73.12%<br />top5=91.35% | top1=72.23%<br />top5=90.95% | 32 91     | 50000        | 72.23% 536ms 15ms |
| vgg16                    | 3124      | top1=73.22%<br />top5=91.31% | top1=71.50%<br />top5=91.20% | 28 83     | 50000        | 71.50% 532ms 15ms |
| vgg19_bn                 | 781       | top1=74.35%<br />top5=91.86% | top1=73.68%<br />top5=91.61% | 143 290   | 50048        | 73.66% 640ms 17ms |
| vgg19                    | 781       | top1=74.13%<br />top5=91.77% | top1=73.28%<br />top5=91.52% | 130 326   | 50048        | 73.27% 628ms 17ms |
| ssd_512_mobilenet1.0_voc | 308       | 75.51%                       | 71.26%                       | 179 425   | 4944         |                   |
| ssd_512_resnet50_v1_voc  | 76        | 80.30%                       | 80.05%                       | 480 2028  | 4928         |                   |
| yolo3_darknet53_voc      | 102       | 81.51%                       | 81.51%                       | 333 2537  | 4944         |                   |
| yolo3_mobilenet1.0_voc   | 76        | 76.03%                       | 71.56%                       | 265 1135  | 4928         |                   |
