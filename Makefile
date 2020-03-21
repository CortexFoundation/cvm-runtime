.PHONY: clean all dep test_cpu test_gpu test_formal 

BUILD := build
INCLUDE := include
TESTS := tests

all: lib test_cpu test_gpu test_formal
	echo ${TEST_CPUS}

dep:
	@cp cmake/config.cmake . --update
	@mkdir -p ${BUILD}
	@mkdir -p ${BUILD}/${TESTS}

lib: dep
	@cd ${BUILD} && cmake ../ && $(MAKE)

TEST_SRCS := $(wildcard ${TESTS}/*.cc)
TEST_EXES := $(patsubst ${TESTS}/%.cc,%,${TEST_SRCS})

TEST_CPUS := $(patsubst %,%_cpu,${TEST_EXES})
TEST_GPUS := $(patsubst %,%_gpu,${TEST_EXES})
TEST_FORMALS := $(patsubst %,%_formal,${TEST_EXES})

test_cpu: ${TEST_CPUS}
test_gpu: ${TEST_GPUS}
test_formal: ${TEST_FORMALS}

%_cpu: ${TESTS}/%.cc lib
	g++ -o ${BUILD}/${TESTS}/$@ $< -DDEVICE=0 -std=c++11 -I${INCLUDE} -L${BUILD} -lcvm_runtime -fopenmp -fsigned-char -pthread -Wl,-rpath=${BUILD}

%_gpu: ${TESTS}/%.cc lib
	g++ -o ${BUILD}/${TESTS}/$@ $< -DDEVICE=1 -std=c++11 -I${INCLUDE} -L${BUILD} -lcvm_runtime -fopenmp -fsigned-char -pthread -Wl,-rpath=${BUILD}

%_formal: ${TESTS}/%.cc lib
	g++ -o ${BUILD}/${TESTS}/$@ $< -DDEVICE=2 -std=c++11 -I${INCLUDE} -L${BUILD} -lcvm_runtime -fopenmp -fsigned-char -pthread -Wl,-rpath=${BUILD}

clean:
	  rm -rf ./build/*
