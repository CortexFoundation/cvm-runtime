# Installation

The doc contains two parts mainly: C backend module build and the CVM & MRT python package installation.

## Pre-requisites

#### Ubuntu (CentOS, RedHat)

Dependent compilation tools: g++, make, cmake, python3

#### MacOS

we have already tested the project setup on MacOS, and the system need some extra dependency library installation besides ubuntu requests. 

Using the `brew` command to install `libomp` in Mac. And if neccessary, install the `brew` tools in Mac terminal first, please.

``` bash
brew install libomp
```

## CXX Build

This section gives a guide of compiling CVM.

### Setup Configurations for Compilation

Check file `config.cmake` exists in your project root,
or execute the following command to create:

``` bash
cp cmake/config.cmake .
```

The `config.cmake` make some settings to change the behavior of
compilation:

- `PROFILE`: print the inference time cost of each operators in model.
- `DEBUG`: add the `-g` flag in cxx compilation to gdb debug.
- `ENABLE_CPU`: enable CPU inference
- `ENABLE_FORMAL`: enable FORMALIZATION inference(slow but code easy read)
- `ENABLE_CUDA`: enable CUDA device
- `ENABLE_OPENCL`: enable OPENCL device(not well-supported)
- `USE_NANO`: option that enable the optimizer for NVIDIA jetson-nano device

### CVM Build

All the targets locate at the directory `build` by default. You can build the project with command:

``` bash
make all
```

*Notice*: execute `make -j8 lib` for the first time might encounter an error like:

  ```
  Error copying file (if different) from "/alongdirname/cuda_compile_1_generated_broadcast.cu.o.depend.tmp" to "/alongdirname/cuda_compile_1_generated_broadcast.cu.o.depend".
  CMake Error at /alongdirname/cuda_compile_1_generated_broadcast.cu.o.cmake:246 (message):
    Error generating
    /alongdirname/./cuda_compile_1_generated_broadcast.cu.o
  build.make:83: recipe for target '/alongdirname/cuda_compile_1_generated_broadcast.cu.o' failed
  ```

  Executing it again will fix the problem.

However the command above will compile all modules in the project, you can specify build target as below sections for a less complication time.

#### cvm library

CVM project is pure C&C++ backend AI inference framework. It's easy for developer to compile the target shared library via below commands:

``` bash
make lib
```

The command will generate two dynamic libraries: `libcvm.so` and `libcvm_runtime.so`, where `libcvm_runtime.so` is lighter than `libcvm.so`. Generally library `cvm_runtime` contains only the inference relative module whereas the `cvm` library compiles the extra symbol, graph, compile pass module etc.

#### unit test

This project have some unit test in subdirectory `tests`, you could compile the test binary with the command:

``` bash
make tests
```

It will generate many test binary in the `build/tests/` directory.


#### doc generator

CVM support `sphinx` documentation, and before compiling the documentation in directory `docs`, you need to install some required python packages. We have collected the neccessary packages in the `docs/requirements.txt`, so you can simply run the command to install the pre-requisites:

``` bash
pip install -r docs/requirements.txt
```

Besides, make sure that you have installed the doxygen for C++ API
generator. Refer to offical website for more details:
[Doxygen Install](https://www.doxygen.nl/manual/install.html).

And then generate the html format with this command:

``` bash
make html
```

All the target html resource files locate at directory `docs/html` as static assets, you can serve it with command:

``` bash
python -m http.server --directory docs/html [port]
```

And then read the doc.

## Python Installation

CVM & MRT python packages are respectively stored at directories:
`python/cvm` and `python/mrt`. and you may setup python for the
ability of importing cvm and mrt.

### Pre-requisites

- python version: 3.6 and higher
- python dependencies: numpy

#### CVM requirements

Use the command to install CVM python requirements:

``` bash
pip install -r install/requirements.txt
```

More dependency details refer to `install/requirements.txt` please.

#### MRT requirements

The mxnet and gluoncv are required. For installation, please respectively refer to:

- [mxnet installation](https://mxnet.apache.org/get_started)

- [gluoncv installation](https://gluon-cv.mxnet.io/install.html)

*Notice*: Mxnet in official website has different versions that 
whether enables cuda. Install with pip for the right mxnet
release version if you want to use GPU, such as the `mxnet_cu101` 
version etc.

### Setup

There are two way of installing python packages: using the python default setup tools or export `PYTHONPATH` environments if you don't want to pollute your python package environments.

#### Target Build

The `python` makefile target have been set for installing the cvm and mrt python package into `sitepackages`. Run the command as below:

``` bash
make python
```

*Notice*: the command above will install the cvm and mrt packages, so that it's neccessary to prepare the two requirements before.

#### Export Environments

Execute the commands to export `PYTHONPATH` in terminal or add the commands into your `.bashrc` if you are frustrated about typing export commands before every time opening terminal:

``` bash
export PYTHONPATH={the project path}/python:${PYTHONPATH}
export LD_LIBRARY_PATH={the project path}/build:${LD_LIBRARY_PATH}
```

### Install Test

After passing the above procedures, try to import the cvm & mrt packages for test:

``` bash
python -c 'import cvm'
python -c 'import mrt'
```

Here is a successfull output:

``` bash
Register cvm library into python module: cvm.symbol
```




