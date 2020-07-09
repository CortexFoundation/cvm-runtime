# Installation

The doc contains two parts mainly: C backend module build and the python package installation.

## Build Targets

All the targets locate at the directory `build` by default. You can build the project with command:

``` bash
make -j all
```

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

CVM python packages are stored in `python/`, and you have to export the path to python for the `import cvm` primitive.

### Pre-requisites

- python version: 3.6 and higher
- python dependencies: numpy

### Export Environments

Execute the commands to export `PYTHONPATH` in terminal or add the commands into your `.bashrc` if you are frustrated about typing export commands before every time opening terminal:

``` bash
export PYTHONPATH={the project path}/python:${PYTHONPATH}
export LD_LIBRARY_PATH={the project path}/build:${LD_LIBRARY_PATH}
```

After that, try to import the `cvm` package for test:

``` bash
python -c 'import cvm'
```

Here is an successfull output:

``` bash
Register cvm library into python module: cvm.symbol
Load the operators: ['__add_symbol__', '__div_symbol__', '__greater_symbol__',
'__max_symbol__', '__mul_symbol__', '__sub_symbol__', 'abs', 'add', 
'broadcast_add', 'broadcast_div', 'broadcast_greater', 'broadcast_max',
'broadcast_mul', 'broadcast_sub', 'clip', 'concatenate', 'conv2d', 
'cvm_clip', 'cvm_left_shift', 'cvm_lut', 'cvm_op', 'cvm_precision', 
'cvm_right_shift', 'dense', 'elemwise_add', 'elemwise_sub', 'expand_dims',
'flatten', 'get_valid_counts', 'max', 'max_pool2d', 'multiply', 
'negative', 'nn.relu', 'non_max_suppression', 'relu', 'repeat', 
'reshape', 'slice_like', 'squeeze', 'strided_slice', 'subtract', 
'sum', 'take', 'tile', 'transpose', 'upsampling',
'vision.non_max_suppression', 'where']
```


