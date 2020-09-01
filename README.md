# Test the cvm-runtime under QEMU with RISC-V support

## Config The QEMU Environment
1. mkdir riscv64-linux
2. cd riscv64-linux

### riscv-gnu-toolchain
```
1. git clone --recursive https://github.com/riscv/riscv-gnu-toolchain
2. sudo apt-get install autoconf automake autotools-dev curl python3 libmpc-dev libmpfr-dev libgmp-dev gawk build-essential bison flex texinfo gperf libtool patchutils bc zlib1g-dev libexpat-dev
3. cd riscv-gnu-toolchain && ./configure --prefix=/opt/riscv
4. make linux
5. export RISCV=/opt/riscv
6. export PATH=$PATH:$RISCV/bin
```
### qemu
```
1. git clone https://github.com/qemu/qemu
2. cd qemu && git checkout v3.0.0
3. ./configure --target-list=riscv64-softmmu
(sudo apt-get install libpixman-1-dev)
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
16. output:
```
0 DDDDDD
load /data/std_out/resnet50_v2/symbol
load /data/std_out/resnet50_v2/params
ntpd: bad address '1.pool.ntp.org'
model loaded
GetOps: memory cost=711M percentage=0.0542804 ops=12396M percentage=0.94572
ops 12500
4
1 3 224 224
ntpd: bad address '0.pool.ntp.org'
output size = 1000
9 11 -6 -4 -7 -5 -19 7 -5 -13 -15 6 -15 -17 -4 -21 -14 -17 -11 -18 -23 -26 -18 -6 -21 -8 -16 -10 -13 5 -16 -4 -1 1 1 1 5 8 -13 -12 -18 -5 -2 -10 -15 -12 -13 -13 -10 -1
2 -22 20 4 -4 -16 0 -3 -9 -10 3
7 4 1 -7 3 -4 0 -12 8 -9 9 9 -11 6 16 0 -4 -1 0 -11 -15 -5 0 -11 -13 -1 2 -11 7 -1 4 4 8 3 -10 8 -8 -6 12 4 4 0 12 0 2 0 -10 5 -3 -18 -11 -9 -10 -26 -8 -8 3 -10 5 9
compare result: success
```
