
# Train AI Model on MxNet

This section shows a full example illustrating how to make train dataset, construct AI model, train model, MRT quantization and finally make an AI contract based on such model.

### Train & Val Dataset Preparation
On MxNet framework, user can train AI model using open-source datasets, such as MNIST, CIFA ... and so on, he can also construct customized train & validation dataset.

In the following python script,

```bash
tests/mrt/train_custom.py
```
we have our own pictures align with respective labels, for simplicity, these pictures are same as those in MNIST. User can use his own dataset or open-source one, such as [imagenet](http://www.image-net.org/challenges/LSVRC/2012/nnoupb/ILSVRC2012_img_train.tar). Each sub-folder contains pictures belong to the class named as sub-folder, both training and validation dataset obey same directory structure.

Training/testing dataset directory structure

|           train              |          test               |
|------------------------------|-----------------------------|
|    train_dataset/0/xx.png    |    test_dataset/0/xx.png    |
|       ...                    |       ...                   |
|    train_dataset/0/yy.png    |    test_dataset/0/yy.png    |
|        .                     |       .                     |
|        .                     |       .                     |
|        .                     |       .                     |
|    train_dataset/9/xx.png    |    test_dataset/9/xx.png    |
|       ...                    |       ...                   |
|    train_dataset/9/yy.png    |    test_dataset/9/yy.png    |


### Data Loader
After datasets preparation, data loader is defined as following to load train & test pictures in training and validating stages. Parameter flag=0 in mx.gluon.data.vision.datasets.ImageFolderDataset() specifies that image data only has one channel as all pictures are gray.

Although CVM-Runtime is run in fixed-point, all the data format could be float format in training stage. The image & label data are formatted as np.float32.

```python
batch_size = 512

train_path = "path-to-train_data"
test_path  = "path-to-test_data"
train_dataset = mx.gluon.data.vision.datasets.ImageFolderDataset(
                               train_path, 
                               flag=0, 
                               transform = lambda data, label: (nd.transpose(data.astype('float32'), (2,0,1)), label)
                              )
test_dataset  = mx.gluon.data.vision.datasets.ImageFolderDataset(
                               test_path, 
                               flag=0, 
                               transform = lambda data, label: (nd.transpose(data.astype('float32'), (2,0,1)), label)
                              )

train_loader = mx.gluon.data.DataLoader(train_dataset, shuffle=True, batch_size=batch_size)
test_loader  = mx.gluon.data.DataLoader(test_dataset,  shuffle=True, batch_size=batch_size)


```

### AI Model definition
According to the task of training, user can define own AI model as normal, add some layers needed, just like following snippet. Notice that currently CVM does not support all the layers :ref:`Supported Operators`.

```python
net = nn.HybridSequential(prefix='Custom')
with net.name_scope():
    net.add(
        nn.Conv2D(channels=32, kernel_size=(3, 3), activation='relu'),
        nn.MaxPool2D(pool_size=(2, 2), strides=(1, 1)),
        ...
        nn.Block(...)
        ...
        nn.Conv2DTranspose(...)
        ...
        nn.Conv2D(channels=64, kernel_size=(3, 3), activation='relu'),
        ...
        nn.ELU(...)
        ...
        nn.Dropout(...)
        ...
        nn.MaxPool2D(pool_size=(2, 2), strides=(2, 2)),
        nn.Flatten(),
        ...
        nn.Dense(10, activation=None),
        # add whatever NN layer you want
        )
```

### Training

Based upon the dataloader and AI model, training starts to minimize the predefined cost function, one can check the convergency while training. 

```python
    for epoch in range(num_epochs):
        batch_id = 0
        for inputs, labels in train_loader:
            inputs = inputs.as_in_context(ctx)
            labels = labels.as_in_context(ctx)

            with autograd.record():
                outputs = net(inputs)
                loss = loss_function(outputs, labels)

            loss.backward()
            metric.update(labels, outputs)

            trainer.step(batch_size=inputs.shape[0])
            print("epoch = {}, batch_id = {}".format(epoch, batch_id))
            batch_id = batch_id + 1
        name, acc = metric.get()
        print('After epoch {}: {} = {:5.2%}'.format(epoch + 1, name, acc))
        metric.reset()
```

### Model Saving
The trained model are save in two files, one specifies model structure including layer information, and the other records the layer parameters. These two files are the input to MRT when quantization.

```python
    sym = net(mx.sym.var('data'))
    sym_file   = "path-to-model.json"
    param_file = "path-to-model.param"

    with open(sym_file, "w") as fout:
        fout.write(sym.tojson())
    net.collect_params().save(param_file)
```

# MRT Quantization
Fixed-point quantization is based upon the MRT methodology which includes 6 stages, i.e., Prepare, Split Model, Calibration, Quantization, Merge Model, Evaluation and Compilation. For each stage where needs some specification parameters, user can adjust these parameters to get higher quantization performance than that from default settings.

The following is executed for MRT quantizaiton.

```python
python python/mrt/main2.py python/mrt/model_zoo/mymodel.ini
```

### Datasets

User need to define some datasets when quantization in stages [CALIBRATION], [EVALUATION] and [COMPILATION]. When use custom datasets, the custom dataset should be registered firstly in file 

```python
python/mrt/dataset.py
```

```python
@register_dataset("Custom")
class CustomDataset(VisionDataset):

    def _load_data(self):
        self.dataset = mx.gluon.data.vision.datasets.ImageFolderDataset(
            root=self.root_dir, 
            flag=0, 
            transform = lambda data, label: (nd.transpose(data.astype('float32'), (2,0,1)), label)
        )
        
        batch_size = self.ishape[0]
        self.data = gluon.data.DataLoader(
            self.dataset,
            batch_size=batch_size, shuffle=False, 
            last_batch='rollover'
        )
        
```

This is always same in training data preparation in Model Training.

Below an example is illustrated for how to do MRT quantization.

### Prepare

The raw model need to be prepared in order to be compatible with the mrt pipeline:

``` python
    model = Model.load(sym_path, prm_path)
    model.prepare(set_batch(input_shape, 1))
```

where sys_path and prm_path are the path of model configuration & parameter just trained above, they are specified in configuration file mymodel.ini 

```python
[DEFAULT]
Model_dir=path-to-model
Model_name=Custom
Device_type=
Device_ids=
# default: None(0), DEBUG(10), INFO(20)
# WARNING(30), ERROR(40), CRITICAL(50)
Verbosity=20
Input_shape=(-1, 1, 28, 28)
Start=
```

and parsed as following.

```python
    default_dir = ""
    model_dir = _get_val(cfg, sec, 'Model_dir', dval=default_dir)
    assert path.exists(model_dir), \
        "Please create the folder `data` first"
    model_name = _get_val(cfg, sec, 'Model_name')
    model_prefix = path.join(model_dir, model_name)

```

In [DEFAULT] section of mymodel.ini, one can uses "cpu" or "Device_ids"-th "gpu", starts from "Start=" stage to do quantization.

### Split Model

In a model split operation, a model can be split into top and base by specifying keys. Default values would be fine in Section [SPLIT_MODEL] in mymodel.ini.

```python
    base, top = model.split(keys)
```

### Calibration

In some cases, it's better to calibrate the model before quantization so as to make the model fit the training or testing data. By doing so, one can achieve higher quantization accuracy in later stages.

One can specifies the calibration dataset, batch size, where data storing, and threshold of internal data flow.

```python
[CALIBRATION]
Batch=256
Calibrate_num=1
Lambda=
Dataset=Custom
Dataset_dir=/home/mint/.mxnet/datasets
Device_type=cpu
Device_ids=
Dump=
```

A mrt instance is created firstly for calibration and quantization stage.

```python
mrt = model.get_mrt() if keys == '' else base.get_mrt()
```

Then, calibration is executed by the given calibrated parameters.

```python
for i in range(calibrate_num):
    data, _ = data_iter_func()
    mrt.set_data(data)
    mrt.calibrate(lambd=lambd, ctx=ctx)
```

### Quantiation

A mrt instance can perform the quantization process, the user can set up some predefined parameters for mrt if needed, such as input precision, output precision, softmax lambda, shift bits as well as threshold for a particular node, etc.

```python
mrt.set_input_prec(input_precision)
mrt.set_output_prec(output_precision)
mrt.set_softmax_lambd(softmax_lambd)
mrt.set_shift_bits(shift_bits)
mrt.set_threshold(name, threshold)
```
The quantization process is performed as follows:

```python
mrt.quantize()
```

### Merge Model

By specifying the base and top models along with corresponding node key maps, the user can create a model merger instance.

```python
model_merger = Model.merger(qmodel, top, mrt.get_maps())
```

By specifying callback merging function, the user can merge the top and base models, and get the ouput scales by configure oscale_maps. Commonly user can just use default values in this stage.

```python
qmodel = model_merger.merge(callback=mergefunc)
oscale_maps = _get_val(cfg, sec, 'Oscale_maps', dtype=PAIR(str_t, str_t))
oscales = model_merger.get_output_scales(mrt_oscales, oscale_maps)
```

### Evaluation

Quantized model reduction and performance comparison are implemented in the evaluation stage:

```python
org_model = Model.load(sym_path, prm_path)
graph = org_model.to_graph(ctx=ctx)
dataset = ds.DS_REG[ds_name](set_batch(input_shape, batch))
data_iter_func = dataset.iter_func()
metric = dataset.metrics()

...

split_batch = batch//ngpus
rqmodel = reduce_graph(qmodel, {
    'data': set_batch(input_shape, split_batch)})
qgraph = rqmodel.to_graph(ctx=ctx)
qmetric = dataset.metrics()

...

utils.multi_validate(evalfunc, data_iter_func, quantize,
                     iter_num=iter_num,
                     logger=logging.getLogger('mrt.validate'),
                     batch_size=batch)
```

One can use custom dataset to evaluate the model performance after quantization.

```python
        ds_name = _get_val(cfg, sec, 'Dataset')
        dataset_dir = _get_val(cfg, sec, 'Dataset_dir', dval="")
```

where 'Dataset' is given in 

```python
[EVALUATION]
Dataset=Custom
Dataset_dir=path-to-evaluation-dataset
Batch=256
Device_type=cpu
Device_ids=0
Iter_num=32
```

### Compilation

Compilation stage include model conversion from mxnet to cvm, and model dump:

```python
qmodel.to_cvm(model_name_tfm, datadir=dump_dir,
    input_shape=set_batch(input_shape, batch),
    target=device_type, device_ids=device_ids)
```

as well as dump of sample data and ext files:

```python
dump_data = sim.load_real_data(
    dump_data.astype("float64"), 'data', mrt.get_inputs_ext())
model_root = path.join(dump_dir, model_name_tfm)
np.save(path.join(model_root, "data.npy"),
        dump_data.astype('int8').asnumpy())
infos = {
    "inputs_ext": inputs_ext,
    "oscales": oscales,
    "input_shapes": input_shape,
}
sim.save_ext(path.join(model_root, "ext"), infos)
```
All parameter related in Compilation should be defined in [COMPLIATION] section of mymodel.ini.

```python
[COMPILATION]
Dataset=Custom
Dataset_dir=path-to-compilation-dataset
Batch=256
Device_type=
Device_ids=
Dump_dir=
```

As results, "dump_data" and CVM model are made and stored in "model_root".

Currently MRT framework only supports MxNet model for quantization, ML model transformation trained with other AI framwworks, for example TensorFlow, PyTorch etc., will comming soon.


# Model&Data Uploading

There are 3 phases to upload ml models/data on the storage layer of the Cortex blockchain. There are more detailed, step-by-step description on how to upload your model and data [link](https://github.com/CortexFoundation/tech-doc/blob/master/model-writing-tutorial.md).

### Upload Phase
mI models and input data are treated as a special type of smart contract on the Cortex blockchain. Creators need to send a special transaction with a function call to advanced the upload progress. Each transaction will increase the file upload progress by 512K bytes, consuming the corresponding storage quota.

### Preparation Phase
After the completion of the upload phase, the file preparation phase is entered. This phase lasts for 100 blocks (about 25 minutes).

### Mature Phase
After 100 blocks, the prepared files enter the mature phase. The model/data is saved on the storage layer of the Cortex blockchain. After mature phase, the file will completed broadcasting to the network to reach the entire distributed file system; otherwise, the network consensus will reject relevant contract calls.


# Smart Contract

Cortex officially supports Solidity programming language to develop AI contracts and complie to CVM bytcode. After one uploaded ML model and model input data as in previous Section, one can make a smart contract via either [Solidity Docs](https://solidity.readthedocs.org/en/latest/) or [Cortex-Remix](https://cerebro.cortexlabs.ai/remix) programming language.

For more details on Cortex smart contract, please refer [AI Contracts](https://github.com/CortexFoundation/tech-doc/blob/master/ai-contracts.md).
