# Installation

The doc contains two parts mainly: C backend module build and the python package installation.

## Pre-requisites

### Ubuntu(CentOS, RedHat)

Dependent compilation tools: g++, make, cmake, python3

### MacOS

we have already tested the project setup on MacOS, and the system need some extra dependency library installation besides ubuntu requests. 

Using the `brew` command to install `libomp` in Mac. And if neccessary, install the `brew` tools in Mac terminal first, please.

``` bash
brew install libomp
```

## Build Targets

This section gives an example of compiling CVM.

1. Config the configuration for compilation

   1. Check file `config.cmake` exists in your project root, or execute the following command:

      ```bash
      cp cmake/config.cmake .
      ```

   2. set the `ENABLE_CUDA` variable `ON` in `config.cmake` line 6.

2. build CVM

All the targets locate at the directory `build` by default. You can build the project with command:

``` bash
make -j all
```

> Note that executing `make -j8 lib` for the first time might encounter an error like

```
Error copying file (if different) from "/alongdirname/cuda_compile_1_generated_broadcast.cu.o.depend.tmp" to "/alongdirname/cuda_compile_1_generated_broadcast.cu.o.depend".
CMake Error at /alongdirname/cuda_compile_1_generated_broadcast.cu.o.cmake:246 (message):
	Error generating
	/alongdirname/./cuda_compile_1_generated_broadcast.cu.o
build.make:83: recipe for target '/alongdirname/cuda_compile_1_generated_broadcast.cu.o' failed
```

> Executing it again will fix the problem.

However the command above will compile all modules in the project, you can specify build target as below sections for a less complication time.

### cvm library

CVM project is pure C&C++ backend AI inference framework. It's easy for developer to compile the target shared library via below commands:

``` bash
make -j lib
```

The command will generate two dynamic libraries: `libcvm.so` and `libcvm_runtime.so`, where `libcvm_runtime.so` is lighter than `libcvm.so`. Generally library `cvm_runtime` contains only the inference relative module whereas the `cvm` library compiles the extra symbol, graph, compile pass module etc.

### unit test

This project have some unit test in subdirectory `tests`, you could compile the test binary with the command:

``` bash
make tests
```

It will generate many test binary in the `build/tests/` directory.


### doc generator

CVM support `sphinx` documentation, and before compile the documentation in directory `docs`, you need to install some python required packages. We have collect the neccessary packages in the `docs/requirements.txt`, so you can simply run the command to install the pre-requisites:

``` bash
pip install -r docs/requirements.txt
```

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

CVM python packages are stored in `python/`, and you have to setup python for the ability of importing cvm.

### Pre-requisites

- python version: 3.6 and higher
- python dependencies: numpy

### Setup

There are two way of installing python packages: using the python default setup tools or export `PYTHONPATH` environments if you don't want to pollute your python package environments.

#### 1. Build Target

The `python` target have been set for installing the cvm and mrt python package into `sitepackages`. Run the command as below:

``` bash
make python
```

#### 2. Export Environments

Execute the commands to export `PYTHONPATH` in terminal or add the commands into your `.bashrc` if you are frustrated about typing export commands before every time opening terminal:

``` bash
export PYTHONPATH={the project path}/python:${PYTHONPATH}
export LD_LIBRARY_PATH={the project path}/build:${LD_LIBRARY_PATH}
```

### Installation Test

After passing the above procedures, try to import the `cvm` package for test:

``` bash
python -c 'import cvm'
```

Here is an successfull output:

``` bash
Register cvm library into python module: cvm.symbol
```




