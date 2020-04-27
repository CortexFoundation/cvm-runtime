test_%:test_%.cpp util.hpp
	g++ -g $< -o $@ -lOpenCL -std=c++11

TARGET=sw_emu
PLATFORM=xilinx_u50_gen3x16_xdma_201920_3

%:
	v++ -t $(TARGET) --platform=$(PLATFORM) -c -k $@ -o '$@.xo' ../../src/runtime/opencl/ops/fpga/$@.cpp
	v++ -t $(TARGET) --platform=$(PLATFORM) -l -o 'ops.xclbin' $@.xo 


clean:
	rm -f *v++* *xo* *xilin* *xclbin*
