# Mnist Training & Quantization

### CVM-Runtime Project Compilation

1. Config the configuration for compilation

   1. Check file `config.cmake` exists in your project root, or execute the following command:

      ```bash
      cp cmake/config.cmake .
      ```

   2. set the `ENABLE_CUDA` variable `ON` in `config.cmake` line 6.

2. Compile with following command

```bash
make -j8 lib
make python
```

### Mnist Training

Execute the following command:

```bash
python3 tests/python/train_mnist.py
```

Training model is stored in `~/mrt_model`.

### Mnist Quantization

Execute the following command:

```bash
python3 python/mrt/main2.py python/mrt/model_zoo/mnist.ini
```

 All the pre-quantized model configuration file is stored in `python/mrt/model_zoo`, and the file `config.example.ini` expositions all the key meanings and value. 