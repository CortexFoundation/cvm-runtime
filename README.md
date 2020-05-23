
### Install the FPGA Development Environment
1. xilinx runtime (xrt)
2. compiler(vitis v++)
3. the code was tested on U50: https://www.xilinx.com/products/boards-and-kits/alveo/u50.html#gettingStarted

### Compiling
1. Change the makefile line 47 PLATFORM to your own: https://github.com/CortexFoundation/cvm-runtime/blob/fpga/Makefile#L47
2. make fpga
3. make test_model_fpga && ./build/tests/test_model_fpga
 
