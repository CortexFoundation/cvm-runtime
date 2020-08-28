# Test the cvm-runtime under QEMU with RISC-V support

## Config The QEMU Environment
1. mkdir riscv64-linux
2. cd riscv64-linux

### riscv-gnu-toolchain
```
1. git clone --recursive https://github.com/riscv/riscv-gnu-toolchain
2. cd riscv-gnu-toolchain && ./configure --prefix=/opt/riscv
3. make linux
4. export RISCV=/opt/riscv
5. export PATH=$PATH:$RISCV/bin
```
### qemu
```
1. git clone https://github.com/qemu/qemu
2. cd qemu && git checkout v3.0.0
3. ./configure --target-list=riscv64-softmmu
4. make -j $(nproc)
5. sudo make install
```
### linux kernel
```
1. git clone https://github.com/torvalds/linux.git
2. cd linux && git checkout v4.19-rc3
3. make ARCH=riscv CROSS_COMPILE=riscv64-unknown-linux-gnu- defconfig
4. make ARCH=riscv CROSS_COMPILE=riscv64-unknown-linux-gnu- -j $(nproc)
```

### busybear-linux
```
1. git clone https://github.com/michaeljclark/busybear-linux.git
2. cd busybear-linux && git submodule init && git submodule update
3. cd src/riscv-pk && git checkout master
2. cd ../../ && CROSS_COMPILE=riscv{{bits}}-unknown-linux-gnu- make -j $(nproc)
```

## Deploy CVM-Runtime to QEMU

1. git clone git@github.com:CortexFoundation/cvm-runtime.git
2. cd cvm-runtime && git checkout riscv
3. cp cmake/config . 
4. make lib && make test_model_riscv
5. cd ../riscv64-linux/busybear-linux 
6. mkdir etc/tests
7. cp ../../cvm-runtime/build/libcvm.so ../../cvm-runtime/build/tests/test_model_riscv etc/tests/ 
8. cp model data: cp -r /data/std_out/resnet50_v2 etc/tests/
9. modify the scripts/start-qemu.sh:
```
 32 # construct command
 33 cmd="${QEMU_SYSTEM_BIN} -nographic -machine virt \
 34   -kernel build/riscv-pk/bbl \
 35   -m 2G \
 36   -append \"root=/dev/vda ro console=ttyS0\" \
 37   -drive file=busybear.bin,format=raw,id=hd0 \
 38   -device virtio-blk-device,drive=hd0 \
 39   -netdev ${QEMU_NETDEV},id=net0 \
 40   -device virtio-ne
```
9. make clean && make && ./scripts/start-qemu.sh
10. login as root, and password is busybear
11. cd /etc/tests
12. export LD_LIBRARY_PATH=/etc/tests
13. mkdir -p /data/std_out/
14. mv /etc/tests/resnet50_v2 /data/std_out/
15. cd /etc/tests && ./test_model_riscv
