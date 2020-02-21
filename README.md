# cvm-runtime
CVM Runtime



## Performance
cpu: multi cpu > avx2 > 1 cpu

model|  Jetson Nano \- Cortex\-A57(s) | Intel E5\-2650(s) |  Jetson Nano \- GPU(128 CUDA Cores)(s) | 1080Ti(3584 CUDA Cores)(s)
-|-|-|-|-
resnet50_mxg | 1.2076| 0.3807| 1.2366 | 0.0931
resnet50_v2 |1.4674| 0.5005 | 0.7962 | 0.0844
qd10_resnet20_v2|0.2944|0.1605 | 0.1243 | 0.0247
trec | 0.0075| 0.0028 | 0.0137 | 0.0248
dcnet_mnist_v1|0.0062|0.0057 | 0.0089 | 0.0111
mobilenetv1.0_imagenet|0.3508| 0.1483 | 0.1223 | 0.0257
resnet50_v1_imagenet|1.2453|0.3429 | 0.6937 | 0.0830
animal10 | 0.3055 | 0.1466 | 0.1304 | 0.0269
vgg16_gcv|4.3787| 0.6092 | 2.1159 | 0.1252
sentiment_trec|0.0047| 0.0022 |   0.0150 | 0.0110
vgg19_gcv|5.1753| 0.7513 | 2.4624 | 0.1208
squeezenet_gcv1.1|0.3889| 0.0895 |  0.1080 | 0.0211
squeezenet_gcv1.1|0.1987| 0.1319 | 0.1850 | 0.0342
shufflenet|1.4575| 0.7697 | 0.2358 | 0.0204
