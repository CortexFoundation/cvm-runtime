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

TARGET=sw_emu
PLATFORM=xilinx_u50_gen3x16_xdma_201920_3

fpga:src/runtime/opencl/ops/fpga/*.cpp
	v++ -t ${TARGET} --platform=${PLATFORM} -c -k $(basename $(notdir $<)) -o '${BUILD}/fpga/$(basename $(notdir $<)).${TARGET}.xo' $< 

python: lib
	bash env.sh

clean:
	  rm -rf ./build/*
