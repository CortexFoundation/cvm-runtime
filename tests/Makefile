
all: test_model_cpu test_model_gpu test_model_formal test_op 

test_model_cpu: test_model.cc
	g++ -o test_model_cpu test_model.cc -I../include -L../build/cpu/  -lcvm_runtime_cpu --std=c++11 -pthread -fopenmp -I../ -ldl -g -DUSE_GPU=0 -DCVM_PROFILING -fsigned-char

test_model_formal: test_model.cc
	g++ -o test_model_formal test_model.cc -I../include -L../build/formal/ -lcvm_runtime_formal --std=c++11 -pthread -fopenmp -I../ -ldl -g -DUSE_GPU=0 -fsigned-char

test_model_gpu: test_model.cc
	g++ -o test_model_gpu test_model.cc -I../include -L../build/cpu/ -L../build/gpu/ -lcvm_runtime_cuda  -lcudart -lcuda --std=c++11 -pthread -fopenmp -I./ -ldl -g -DUSE_GPU=1 -DCVM_PROFILING -fsigned-char

test_op : test_op.cc
	g++ -o test_op test_op.cc -I../include -L../build/ -lcvm_runtime --std=c++11 -pthread -fopenmp -I../ -ldl -g -fsigned-char

clean:
	rm -f test_model_cpu test_model_formal test_model_gpu test_op
