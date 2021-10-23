[TOC]

# Introduction

evoke passes by `mrt_entry.py` 

1. for debugging purpose: `mrt_prepare`, `mrt_calibrate`, `mrt_quantize`
2. for evaluation purpose: `mrt_evaluate`
3. for compilation purpose: `mrt_compile`
4. for the whole process, the configuration module will combine the afore mentioned processes.

Currently supported configuration format by MRT: `yaml`, `argparse`, `ini`(not integrated into mrt_entry.py yet)

# YAML Configuration Examples

## alexnet

Config  `alexnet.yaml` as follows.

```yaml
COMMON:
  MODEL_NAME: alexnet
  VERBOSITY: debug
  START_AFTER: calibrate
  RUN_EVALUATE: True
CALIBRATE:
  BATCH: 16
  NUM_CALIB: 1
  LAMBD: 16
  DATASET_NAME: imagenet
  DEVICE_TYPE: gpu
  DEVICE_IDS: [2]
QUANTIZE:
  INPUT_PRECISION: 8
  OUTPUT_PRECISION: 8
  DEVICE_TYPE: gpu
  DEVICE_IDS: [2]
EVALUATE:
  BATCH: 160
  DEVICE_TYPE: gpu
  DEVICE_IDS: [0]
  ITER_NUM: 10
```

run command

```python
python main2.py ~/mrt_yaml_root/alexnet.yaml
```

or run either of the following commands for each pass.

```bash
# preparation
python main2.py ~/mrt_yaml_root/alexnet.yaml prepare
# calibration
python main2.py ~/mrt_yaml_root/alexnet.yaml calibrate
# quantization
python main2.py ~/mrt_yaml_root/alexnet.yaml quantize
# evaluation
python main2.py ~/mrt_yaml_root/alexnet.yaml evaluate
# compilation
python main2.py ~/mrt_yaml_root/alexnet.yaml compile
```

## ssd_512_voc_resnet50_v1_voc

```yaml
COMMON:
  MODEL_NAME: ssd_512_resnet50_v1_voc
  VERBOSITY: info
  RUN_EVALUATE: True
PREPARE:
  INPUT_SHAPE: [-1, 3, 512, 512]
  SPLIT_KEYS: [
    "ssd0_multiperclassdecoder0_zeros_like0",
    "ssd0_multiperclassdecoder0_slice_axis0",
    "ssd0_normalizedboxcenterdecoder0_concat0"
  ]
CALIBRATE:
  NUM_CALIB: 1
  LAMBD: 16
  DATASET_NAME: voc
  DEVICE_TYPE: gpu
  DEVICE_IDS: [2]
QUANTIZE:
  OUTPUT_PRECISION: 30
  DEVICE_TYPE: gpu
  DEVICE_IDS: [2]
  THRESHOLDS: [
    ["data", 2.64],
    ["ssd0_multiperclassdecoder0_slice_axis0", 1],
  ]
  ATTRIBUTE_DEPS: [
    [
      "_greater_scalar",
      [
        ["scalar", "ssd0_multiperclassdecoder0_slice_axis0"],
      ]
    ],
    [
      "_contrib_box_nms",
      [
        ["valid_thresh", "ssd0_multiperclassdecoder0_slice_axis0"],
      ]
    ],
  ]
  OSCALE_MAPS: [
    ["ssd0_slice_axis41", "ssd0_multiperclassdecoder0_zeros_like0"],
    ["ssd0_slice_axis42", "ssd0_multiperclassdecoder0_slice_axis0"],
    ["ssd0_slice_axis43", "ssd0_normalizedboxcenterdecoder0_concat0"],
  ]
EVALUATE:
  BATCH: 15
  DEVICE_TYPE: gpu
  DEVICE_IDS: [0, 1, 2]
  ITER_NUM: 10
```

run command

```python
python main2.py ~/mrt_yaml_root/ssd.yaml
```

# CMD Configuration Examples

## alexnet

```python
python main.py cmd alexnet
```

### cmd_prepare

```bash
python main.py prepare alexnet
```

### cmd_calibrate

```bash
python main.py calibrate alexnet \
--batch-calibrate 16 \
--lambd 16 \
--calibrate-num 1 \
--dataset-name imagenet \
--device-type-calibrate gpu \
--device-ids-calibrate 2
```

### cmd_quantize

```bash
python main.py quantize alexnet \
--input-precision 8 \
--output-precision 8 \
--device-type-quantize gpu \
--device-ids-quantize 2
```

### cmd_evaluate

```bash
python main.py evaluate alexnet \
--batch-evaluate 160 \
--device-type-evaluate gpu \
--device-ids-evaluate 0 \
--iter-num 10
```

### cmd_compile

```bash
python main.py compile alexnet
```

### cmd_main

```bash
python main.py cmd alexnet \
--batch-calibrate 16 \
--lambd 16 \
--calibrate-num 1 \
--dataset-name imagenet \
--device-type-calibrate gpu \
--device-ids-calibrate 2 \
--input-precision 8 \
--output-precision 8 \
--device-type-quantize gpu \
--device-ids-quantize 2 \
--batch-evaluate 160 \
--device-type-evaluate gpu \
--device-ids-evaluate 0 \
--iter-num 10 \
--run-evaluate \
--run-compile
```

## ssd_512_voc_resnet50_v1_voc

### cmd_prepare

```bash
python main.py prepare ssd_512_resnet50_v1_voc \
--verbosity info \
--split-keys \
ssd0_multiperclassdecoder0_zeros_like0 \
ssd0_multiperclassdecoder0_slice_axis0 \
ssd0_normalizedboxcenterdecoder0_concat0 \
--input-shape -1 3 512 512
```

### cmd_calibrate

```bash
python main.py calibrate ssd_512_resnet50_v1_voc \
--verbosity info \
--dataset-name voc \
--device-type-calibrate gpu \
--device-ids-calibrate 2
```

### cmd_quantize

```bash
python main.py quantize ssd_512_resnet50_v1_voc \
--verbosity info \
--thresholds \
"{ \
	\"data\": 2.64, \
	\"ssd0_multiperclassdecoder0_slice_axis0\": 1 \
}" \
--output-precision 30 \
--attribute-deps \
"{ \
	\"_greater_scalar\": { \
		\"scalar\": \"ssd0_multiperclassdecoder0_slice_axis0\" \
	}, \
	\"_contrib_box_nms\": { \
		\"valid_thresh\": \"ssd0_multiperclassdecoder0_slice_axis0\" \
	} \
}" \
--oscale-maps \
"{ \
	\"ssd0_slice_axis41\": \"ssd0_multiperclassdecoder0_zeros_like0\", \
	\"ssd0_slice_axis42\": \"ssd0_multiperclassdecoder0_slice_axis0\", \
	\"ssd0_slice_axis43\": \"ssd0_normalizedboxcenterdecoder0_concat0\" \
}"
```

### cmd_evaluate

```bash
python main.py evaluate ssd_512_resnet50_v1_voc \
--verbosity info \
--batch-evaluate 14 \
--device-type-evaluate gpu \
--device-ids-evaluate 0 1 2 3 \
--iter-num 100
```

### cmd_compile

```bash
python main.py compile ssd_512_resnet50_v1_voc
```

