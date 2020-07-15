# Mnist Training & Quantization

This tutorial gives an example of compiling CVM and converting a pre-trained floating point model for mnist dataset to a fixed-point model which is executable on CVM.

### CVM-Runtime Project Compilation
See Section `2.2. Build the Shared Library` in [MRT Installation](README.md)

### Mnist Training
This section gives an example of training a model for mnist dataset and storing the trained model as `~/mrt_model/mnist_dapp.json` and `~/mrt_model/mnist_dapp.params`. CVM is not necessary during this procedure.

Execute the following command:

```bash
python3 tests/mrt/train_mnist.py
```
> Pay attention to python dependencies. This python script uses GPU defaultly so `mxnet` should be gpu version corresponding to CUDA version.
> You can use the `--cpu` argument to use CPU to train the model.

### Mnist Quantization
This section is an example of converting the model trained above to a model executable on CVM.

Execute the following command:

```bash
python3 python/mrt/main2.py python/mrt/model_zoo/mnist.ini
```
`main2.py` does the convert job and `mnist.ini` provides necessary config information, including full path to the model, input shape, dataset, etc.
The converted model is defaultly stored in the same directory as input model directory, as specified by `Model_dir` in `DEFAULT` section of the `ini` file.

 All the pre-quantized model configuration file is stored in `python/mrt/model_zoo`, and the file `config.example.ini` expositions all the key meanings and value. 
