# cvm-runtime
CVM Runtime



## Performance

model|  Jetson Nano \- Cortex\-A57(s) | Intel E5\-2650(s) |  Jetson Nano \- GPU(128 CUDA Cores)(s) | 1080Ti(3584 CUDA Cores)(s)
-|-|-|-|-
yolo_tfm | | | 1.384 | 0.056
resnet50_mxg | 1.2076| 0.3807| 0.235 | 0.011
resnet50_v2 |1.4674| 0.5005 | 0.242 | 0.013
qd10_resnet20_v2|0.2944|0.1605 | 0.102 | 0.012
trec | 0.0075| 0.0028 | 0.002 | 0.001
dcnet_mnist_v1|0.0062|0.0057 | 0.006 | 0.001
mobilenetv1.0_imagenet|0.3508| 0.1483| 0.070  | 0.003
resnet50_v1_imagenet|1.2453| 0.3429 | 0.252 | 0.011
animal10 | 0.3055 | 0.1466 | 0.077 | 0.012
vgg16_gcv|4.3787| 0.6092 | 0.833 | 0.020
sentiment_trec|0.0047| 0.0022 |  0.002 | 0.001
vgg19_gcv|5.1753| 0.7513 | 0.820 | 0.023
squeezenet_gcv1.1|0.3889| 0.0895 |  0.047 | 0.004
squeezenet_gcv1.0|0.1987| 0.1319 | 0.098 | 0.005
shufflenet|1.4575| 0.7697 | 0.157 | 0.007
ssd| | |0.935 | 0.035
ssd_512_mobilenet1.0_coco_tfm| | | 0.414 | 0.026
