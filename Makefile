.PHONY: clean all dep test_cpu test_gpu test_formal python
# .PHONY: test_model_cpu test_model_gpu test_model_formal
# .PHONY: test_op_cpu test_op_gpu test_op_formal

BUILD := build
INCLUDE := include
TESTS := tests

all: lib python test_cpu test_gpu test_formal
	echo ${TEST_CPUS}

dep:
	@cp cmake/config.cmake . --update
	@mkdir -p ${BUILD}
	@mkdir -p ${BUILD}/${TESTS}
	@mkdir -p ${BUILD}/fpga

lib: dep
	@cd ${BUILD} && cmake ../ && $(MAKE)

TEST_SRCS := $(wildcard ${TESTS}/*.cc)
TEST_EXES := $(patsubst ${TESTS}/%.cc,%,${TEST_SRCS})

TEST_CPUS := $(patsubst %,%_cpu,${TEST_EXES})
TEST_GPUS := $(patsubst %,%_gpu,${TEST_EXES})
TEST_FORMALS := $(patsubst %,%_formal,${TEST_EXES})
TEST_OPENCL := $(patsubst %,%_opencl,${TEST_EXES})

test_cpu: ${TEST_CPUS}
test_gpu: ${TEST_GPUS}
test_formal: ${TEST_FORMALS}
test_opencl: ${TEST_OPENCL}

%_cpu: ${TESTS}/%.cc lib
	g++ -o ${BUILD}/${TESTS}/$@ $< -DDEVICE=0 -std=c++11 -I${INCLUDE} -L${BUILD} -lcvm -fopenmp -fsigned-char -pthread -Wl,-rpath=${BUILD}

%_gpu: ${TESTS}/%.cc lib
	g++ -o ${BUILD}/${TESTS}/$@ $< -DDEVICE=1 -std=c++11 -I${INCLUDE} -L${BUILD} -lcvm -fopenmp -fsigned-char -pthread -Wl,-rpath=${BUILD}

%_formal: ${TESTS}/%.cc lib
	g++ -o ${BUILD}/${TESTS}/$@ $< -DDEVICE=2 -std=c++11 -I${INCLUDE} -L${BUILD} -lcvm -fopenmp -fsigned-char -pthread -Wl,-rpath=${BUILD}

%_opencl: ${TESTS}/%.cc lib
	g++ -o ${BUILD}/${TESTS}/$@ $< -DDEVICE=3 -std=c++11 -I${INCLUDE} -L${BUILD} -lcvm_runtime -fopenmp -L/usr/local/cuda/lib64/ -lOpenCL -fsigned-char -pthread -Wl,-rpath=${BUILD}

TARGET=hw
PLATFORM=xilinx_u50_gen3x16_xdma_201920_3

FPGA_SRC=$(wildcard src/runtime/opencl/ops/fpga/*.cpp)
#FPGA_TEMP_OUT=$(patsubst ${BUILD}/fpga/%.xo, $(notdir $(FPGA_SRC)))
FPGA_OBJS=$(patsubst %.cpp,%.xo,$(FPGA_SRC))

FPGA_OUT=ops.xclbin
fpga:$(FPGA_OBJS)
	v++ -t $(TARGET) --platform=$(PLATFORM) -l -o $(FPGA_OUT) $(FPGA_OBJS)
%.xo:%.cpp
#v++ -t $(TARGET) --platform=$(PLATFORM) -c -k $(basename $(notdir $<)) -o '${BUILD}/fpga/$(basename $(notdir $<)).${TARGET}.xo' $<
	v++ -t $(TARGET) --platform=$(PLATFORM) -c -k $(basename $(notdir $<)) -o '$@' $<
	rm $@.*

cleanfpga:
	rm -f v++* *xclbin.* *.xo* src/runtime/opencl/ops/fpga/*.xo*
#fpga:src/runtime/opencl/ops/fpga/*.cpp
#	v++ -t ${TARGET} --platform=${PLATFORM} -c -k $(basename $(notdir $<)) -o '${BUILD}/fpga/$(basename $(notdir $<)).${TARGET}.xo' $< 

test_fpga: tests/test_fpga.cpp
	g++ tests/test_fpga.cpp -o tests/test_fpga -I/usr/local/include/opencv4 -lopencv_core -lopencv_imgcodecs -lopencv_highgui -lopencv_imgproc -I${INCLUDE} -L${BUILD} -lcvm -L/usr/local/cuda/lib64 -lOpenCL -fsigned-char -pthread -Wl,-rpath=${BUILD}

python: lib
	bash env.sh

clean:
	  rm -rf ./build/*
