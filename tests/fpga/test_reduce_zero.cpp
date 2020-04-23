#include <CL/opencl.h>
#include <iostream>
#include <string>
#include <vector>
#include <iostream>
#include <fstream>
#include "util.hpp"
using namespace std;

void reduce_cpu(const int *input, int *output, const int n, const int type){
  int ret = input[0];
  for(int i = 1; i < n; i++){
    if(type == 0){
      ret = ret < input[i] ? input[i] : ret;
    }else{
      ret += input[i];
    }
  }
  output[0] = ret;
}

void reduce_fpga(const int *input, int *output, const int n, const int type){
  cl_int code;

  cl_mem bufA = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(int)*n, NULL, &code);

  clEnqueueWriteBuffer(queue, bufA, CL_TRUE, 0, sizeof(int)*n, input, 0, nullptr, nullptr);

  cl_mem bufC = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(int), NULL, &code);
  cl_kernel kernel = clCreateKernel(program, "reduce_zero", &code);
  int index = 0;
  clSetKernelArg(kernel, index++, sizeof(cl_mem), (void*)&bufA);
  clSetKernelArg(kernel, index++, sizeof(cl_mem), (void*)&bufC);
  clSetKernelArg(kernel, index++, sizeof(int), (void*)&n);
  clSetKernelArg(kernel, index++, sizeof(int), (void*)&type);
  clEnqueueTask(queue, kernel, 0, NULL, NULL);

  clEnqueueReadBuffer(queue, bufC, CL_TRUE, 0, sizeof(int), output, 0, nullptr, nullptr); 
}

int main(){
  init_opencl("ops.xclbin");

  const int n = 257;
  int *input = new int[n];
  int output;// = new int[n];
  int output2;// = new int[n];

  for(int i = 0; i < n; i++){
    input[i] = i % 127;
  }

  int type = 0;
  reduce_cpu(input, &output, n, type);
  reduce_fpga(input, &output2, n, type);

  verify(&output, &output2, 1);
}
