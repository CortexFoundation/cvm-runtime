# cvm-runtime test_model
CVM-Runtime Test_Model

## prerequisite
1. prepare models files under data1
```
/data1
`-- lz_model_storage
    `-- animal10
        `── data
            |── params
            |── result_0.txt
            `── symbol
        `── result_0.txt
`-- std_out
    `-- yolo_tfm
        |-- data.npy
        |-- params
        |-- result_0.txt
        `-- symbol
  ...
```

2. prepare qemu envs
  - for qemu mips gcc cross compile build
  - using qemu-mips docker containers can be: `docker run -it -v $PATH_TO_PROJECT:/project asmimproved/qemu-mips`
  - `apt install gcc-mips-linux-gnu g++-mips-linux-gnu`
  - `apt install qemu-system qemu-system-mips qemu-system-mipsel`
  - `apt install qemu-user-static`
  - using mipsel for little-endian


## build process
- build libcvm_runtime.so / libcvm_runtime.a
- rename libcvm_runtime.so / libcvm_runtime.a with '_cpu' '_formal' suffix
- append the path to LD_LIBRARY_PATH
- build test_model.cc

## execute bin
- check binary/elf: `file ./test_model_formal`
- LD_LIBRARY_PATH="/path_to_libcvm_runtime_library":${LD_LIBRARY_PATH} qemu-mips ./test_model_arch0
- LD_LIBRARY_PATH="/path_to_libcvm_runtime_library":${LD_LIBRARY_PATH} qemu-mips ./test_model_arch1
After running 2 Architectures, the result file can be compared.

- static run: `LD_LIBRARY_PATH="/path_to_libcvm_runtime_library":${LD_LIBRARY_PATH} qemu-mips-static ./test_model_arch0`


# To execute under dynamic library (so: shared library)
- using mips32 big endian
- debian-mips-qemu, install img and start kernel. has ld.so.1 (single core mips cpu)
- scp -R cvm-runtime -P 2222 debian@127.0.0.1:/home/debian
- ssh -p 2222 debian@127.0.0.1 # with password debian
- build libso, build test_model, and execute binary.
