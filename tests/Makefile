
all: test_model test_op 

test_model: test_model.cc
	g++ -o test_model test_model.cc -I../include -L../build/ -lcvm_runtime --std=c++11 -pthread -fopenmp -I../ -ldl -g -fsigned-char

test_op : test_op.cc
	g++ -o test_op test_op.cc -I../include -L../build/ -lcvm_runtime --std=c++11 -pthread -fopenmp -I../ -ldl -g -fsigned-char

clean:
	rm -f test_model_cpu test_model_formal test_model_gpu test_op
