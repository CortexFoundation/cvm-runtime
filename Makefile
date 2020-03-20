.PHONY: clean all test_model test_op dep

BUILD := build

all: lib test_model test_op

dep:
	@cp cmake/config.cmake . --update
	@mkdir -p ${BUILD}

lib: dep
	@cd ${BUILD} && cmake ../ && $(MAKE)

test_model: test_model_cpu test_model_gpu test_model_formal 
test_op: test_op_cpu test_op_gpu test_op_formal

test_model_cpu: tests/test_model.cc lib
	g++ -o ${BUILD}/$@ $< -DUSE_GPU=0 -std=c++11 -Iinclude -L${BUILD} -lcvm_runtime -fopenmp -fsigned-char -pthread -Wl,-rpath=${BUILD}
	@${BUILD}/$@

test_model_gpu: tests/test_model.cc lib
	g++ -o ${BUILD}/$@ $< -DUSE_GPU=1 -std=c++11 -Iinclude -L${BUILD} -lcvm_runtime -fopenmp -fsigned-char -pthread -Wl,-rpath=${BUILD}
	@${BUILD}/$@

test_model_formal: tests/test_model.cc lib
	g++ -o ${BUILD}/$@ $< -DUSE_GPU=2 -std=c++11 -Iinclude -L${BUILD} -lcvm_runtime -fopenmp -fsigned-char -pthread -Wl,-rpath=${BUILD}
	@${BUILD}/$@

test_op_cpu: tests/test_op.cc lib
	g++ -o ${BUILD}/$@ $< -DUSE_GPU=0 -std=c++11 -Iinclude -L${BUILD} -lcvm_runtime -fopenmp -fsigned-char -pthread -Wl,-rpath=${BUILD}
	@${BUILD}/$@

test_op_gpu: tests/test_op.cc lib
	g++ -o ${BUILD}/$@ $< -DUSE_GPU=1 -std=c++11 -Iinclude -L${BUILD} -lcvm_runtime -fopenmp -fsigned-char -pthread -Wl,-rpath=${BUILD}
	@${BUILD}/$@

test_op_formal: tests/test_op.cc lib
	g++ -o ${BUILD}/$@ $< -DUSE_GPU=2 -std=c++11 -Iinclude -L${BUILD} -lcvm_runtime -fopenmp -fsigned-char -pthread -Wl,-rpath=${BUILD}
	@${BUILD}/$@

clean:
	  rm -rf ./build/*
