# Quantization Reference

[TOC]

## Introdution

MRT, short for **Model Representation Tool**, aims to convert floating model into a deterministic and non-data-overflow network. MRT links the off-chain developer community to the on-chain ecosystem, from Off-chain deep learning to MRT transformations, and then uploading to Cortex Blockchain for on-chain deterministic inference.

A full deterministic deep learning framework designed by Cortex is ran within CVM, the Cortex Virtual Machine ,and the integral part in our Cortex Blockchain source is called CVM runtime. All data flow in CVM is an integer with some precision ranged in 0 and 32. We proposed approaches to certify the data non-flow over INT32. The model that goes under MRT transformation can be accepted by CVM, which we called it on-chain model.

MRT is based on the MXNet symbol, doing operations on the whole operators with topological order in models. Besides, for scalability, we've researched the model transformation from TensorFlow into MXNet, models such as mobilenet, inception_v3 have been successfully converted and more operators will be supported in the future. Other deep learning frameworks like PyTorch and Caffe is in the roadmap of our plan.

MRT transformation usage is simple to model-training programmer since we have separated model quantization procedures from source code. One can invoke MRT via programming or configuring the settings file, more detail usage is introduced as below.

## Configuration File 

MRT has separated model quantization configurations from source code for simplifying the user-usage. So one can quantize their model quickly via configuring the .ini file. The running command script is as below.

``` bash
python cvm/quantization/main2.py config/file/path
```

Please refer to the example file: cvm/quantization/models/config.example.ini [link](https://github.com/CortexFoundation/tvm-cvm/blob/wlt/cvm/models/config.example.ini) for more configuration details. Copy the example file and configure the model's quantization settings locally. We have quantized and tested accuracy for some available models in MXNet gluon zoo with configurations file, whose settings are located in [link](https://github.com/CortexFoundation/tvm-cvm/blob/wlt/cvm/models/) cvm/quantization/models for reference. These accuracies are organized into a chart for analysis in section [Model Testing](#Model Testing).

The unify quantization procedure is defined in file: cvm/quantization/main2.py, refer to [main2](https://github.com/CortexFoundation/tvm-cvm/blob/ryt_tmp/cvm/quantization/main2.py) for more quantization details.

## Developer API

The Main public quantization API is located at cvm/quantization/transformer.py, see the detail interface in the following sections. And the main quantization procedure is: 

    Model Load >>> Preparation >>> [Optional] Model Split >>>
    
    Calibration >>> Quantization >>> [Optional] Model Merge >>> Compilation to CVM,

which maps the class methods: 

    Model.load >>> Model.prepare >>> [Optional] Model.split >>> 
    
    MRT.calibrate >>> MRT.quantize >>> [Optional] ModelMerger.merge >>> Model.to_cvm.

The Calibration and Quantization pass is achieved in class MRT.

### Split & Merge

MRT supports for most of MXNet operators while there still exists some unsupported. We advise splitting the model into two sub-graph if there are some unsupported operators and only quantizing the half model (named base_model, indicating the input nodes to split operators generally). In other words, it's the user's responsibility to select the split keys of splitting the original model, while the half model is ignored to quantization pass if necessary. 

#### Currently Supported Operators

Below operators are carefully selected by the MRT developers. The unsupported oprators are the ones that are unquantifiable. For the unsupported operators, you can either split the model with disable-quantization attributes or contact the MRT developers through GitHub for assitance.

##### Transformer

| Operator     | Supported          | Operator    | Supported          |
| ------------ | ------------------ | ----------- | ------------------ |
| SliceAxis    | :heavy_check_mark: | SwapAxis    | :heavy_check_mark: |
| Slice        | :heavy_check_mark: | Flatten     | :heavy_check_mark: |
| SliceLike    | :heavy_check_mark: | Concat      | :heavy_check_mark: |
| Transpose    | :heavy_check_mark: | where       | :heavy_check_mark: |
| repeat       | :heavy_check_mark: | expand_dims | :heavy_check_mark: |
| SliceChannel | :heavy_check_mark: | tile        | :heavy_check_mark: |
| squeeze      | :heavy_check_mark: | Reshape     | :heavy_check_mark: |
| clip         | :heavy_check_mark: | Embedding   | :heavy_check_mark: |

##### NN

| Operator       | Supported          | Operator | Supported          |
| -------------- | ------------------ | -------- | ------------------ |
| Convolution    | :heavy_check_mark: | Pad      | :heavy_check_mark: |
| FullyConnected | :heavy_check_mark: | relu     | :heavy_check_mark: |
| LeakyReLU      | :heavy_check_mark: | Pooling  | :heavy_check_mark: |
| UpSampling     | :x:(TODO)          | softmax  | :heavy_check_mark: |
| BatchNorm      | :heavy_check_mark: | Dropout  | :heavy_check_mark: |
| Activation     | :heavy_check_mark: |          |                    |

##### Broadcast

| Operator      | Supported          | Operator          | Supported          |
| ------------- | ------------------ | ----------------- | ------------------ |
| broadcast_div | :x:                | broadcast_add     | :heavy_check_mark: |
| broadcast_sub | :heavy_check_mark: | broadcast_mul     | :heavy_check_mark: |
| broadcast_to  | :x:                | broadcast_greater | :x:                |

##### Elemwise

| Operator        | Supported          | Operator     | Supported          |
| --------------- | ------------------ | ------------ | ------------------ |
| _mul_scalar     | :heavy_check_mark: | _div_scalar  | :heavy_check_mark: |
| elemwise_add    | :heavy_check_mark: | elemwise_sub | :heavy_check_mark: |
| ceil            | :x:                | round        | :x:                |
| fix             | :x:                | floor        | :x:                |
| abs             | :x:                | sigmoid      | :heavy_check_mark: |
| exp             | :heavy_check_mark: | negative     | ✔️                  |
| _minimum        | :x:                | _maximum     | :x:                |
| _plus_scalar    | :heavy_check_mark: | zeros_like   | :heavy_check_mark: |
| _greater_scalar | :heavy_check_mark: | ones_like    | ✔️                  |

##### Reduce

| Operator | Supported          | Operator | Supported |
| -------- | ------------------ | -------- | --------- |
| max      | :heavy_check_mark: | min      | :x:       |
| sum      | :heavy_check_mark: | argmin   | :x:       |
| argmax   | :x:                |          |           |

##### Vision

| Operator         | Supported | Operator | Supported |
| ---------------- | --------- | -------- | --------- |
| _contrib_box_nms | :x:       |          |           |

##### Others

| Operator | Supported          | Operator | Supported |
| -------- | ------------------ | -------- | --------- |
| _arange  | :heavy_check_mark: | Custom   | ❌         |
### Public Interface

#### Model

A wrapper class for MXNet symbol and params which indicates model. All the quantization passes return the class instance for unify representation. Besides, the class has wrapped some user-friendly functions API  listed as below.

| func name                                          | usage                                                        |
| -------------------------------------------------- | ------------------------------------------------------------ |
| input_names()                                      | List the model's input names.                                |
| output_names()/names()                             | List the model's output names.                               |
| to_graph([dtype, ctx])                             | A convenient method to create model runtime.<br />Returns mxnet.gluon.nn.SymbolBlock. |
| save(symbol_file, params_file)                     | Dump model to disk.                                          |
| load(symbol_file, params_file)                     | **[staticmethod]** Load model from disk.                     |
| split(keys)                                        | Split the model by `keys` of model internal names.<br />Returns two sub-graph Model instances. |
| merger(base, top[, base_name_maps])                | [**staticmethod**] Returns the ModelMerger with two Model instance. |
| prepare([input_shape])                             | Model preparation passes, do operator checks, operator fusing, operator rewrite, ...etc. |
| to_cvm(model_name[, datadir, input_shape, target]) | Compile current mxnet quantization model into CVM accepted JSON&BINARY format. |

#### MRT

A wrapper class for model transformation tool which simulates deep learning network integer computation within a float-point context. Model calibration and quantization are performed based on a specified model. This class has wrapped some user-friendly functions API introduced as below.

| func name                        | usage                                                        |
| -------------------------------- | ------------------------------------------------------------ |
| set_data(data)                   | Set the data before calibration.                             |
| calibrate([ctx, lambd, old_ths]) | Calibrate the current model after setting mrt data.<br />Context on which intermediate result would be stored, hyperparameter lambd and reference threshold dict could also be specified. <br />Return the threshold dict of node-level output. |
| set_threshold(name, threshold)   | Manually set the threshold of the node output, given node name. |
| set_th_dict(th_dict)             | Manually set the threshold dict.                             |
| set_input_prec(prec)             | Set the input precision before quantization.                 |
| set_out_prec(prec)               | Set the output precision before quantization.                |
| set_softmax_lambd(val)           | Set the hyperparameter softmax_lambd before quantization.    |
| set_shift_bits(val)              | Set the hyperparameter shift_bits before quantization.       |
| quantize()                       | Quantize the current model after calibration.<br />Return the quantized model. |
| get_output_scales()              | Get the output scale of the model after quantization.        |
| get_maps()                       | Get the current name to old name map of the outputs after calibration or quantization. |
| get_inputs_ext()                 | Get the input_ext of the input after quantization.           |
| save(model_name[, datadir])      | save the current mrt instance into disk.                     |
| load(model_name[, datadir])      | [**staticmethod**]Return the mrt instance.<br />The given path should contain corresponding '.json' and '.params' file storing model information and '.ext' file storing mrt information. |

#### ModelMerger

A wrapper class for model merge tool. This class has wrapped some user-friendly functions API introduced as below.

| func name                             | usage                                                        |
| ------------------------------------- | ------------------------------------------------------------ |
| merge([callback])                     | Return the merged model. <br />Callback function could also be specified for updating the top node attributes. |
| get_output_scales(base_oscales, maps) | Get the model output scales after merge.<br />Base model output scales and base name maps should be specified. |
