# Test the cvm-runtime under QEMU with RISC-V support

## Config The QEMU Environment
You will need to refer to this [documentation](https://risc-v-getting-started-guide.readthedocs.io/en/latest/linux-qemu.html) to download and install the following：
```
1. https://github.com/riscv/riscv-gnu-toolchain : master
2. https://github.com/qemu/qemu : v3.0.0
3. https://github.com/torvalds/linux : v4.19
4. https://github.com/riscv/riscv-pk : master
5. https://github.com/michaeljclark/busybear-linux : master
```
## Testing
1. git clone git@github.com:CortexFoundation/cvm-runtime.git
2. git checkout riscv
3. cp cmake/config . (in cvm-runtime directory)
4. make lib && make test_model_riscv
5. cd ${busybear-linux directory}
6. mkdir etc/tests
7. cp ~/cvm-runtime/build/libcvm.so ~/cvm-runtime/build/tests/test_model_riscv etc/tests/ 
8. modify scripts/start-qemu.sh:
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
13. ./test_model_riscv
