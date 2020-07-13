# Mnist Training & Quantization

This tutorial gives an example of compiling CVM and converting a pre-trained floating point model for mnist dataset to a fixed-point model which is executable on CVM.

### CVM-Runtime Project Compilation
This section gives an example of compiling CVM.

1. Config the configuration for compilation

   1. Check file `config.cmake` exists in your project root, or execute the following command:

      ```bash
      cp cmake/config.cmake .
      ```

   2. set the `ENABLE_CUDA` variable `ON` in `config.cmake` line 6.

2. Compile with following command

```bash
make -j8 lib
```
> Note that executing `make -j8 lib` for the first time might encounter an error like

  ``` bash
  Error copying file (if different) from "/alongdirname/cuda_compile_1_generated_broadcast.cu.o.depend.tmp" to "/alongdirname/cuda_compile_1_generated_broadcast.cu.o.depend".
  CMake Error at /alongdirname/cuda_compile_1_generated_broadcast.cu.o.cmake:246 (message):
    Error generating
    /alongdirname/./cuda_compile_1_generated_broadcast.cu.o
  build.make:83: recipe for target '/alongdirname/cuda_compile_1_generated_broadcast.cu.o' failed
  ```

> Executing it again will fix the problem.


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
