#include <CL/opencl.h>
#include <iostream>
#include <string>
#include <stdint.h>
#include <vector>
#include <iostream>
#include <fstream>

#include <string.h>
#include "util.hpp"

using namespace std;
void get_valid_count(const int32_t *x_data, int32_t *y_data, int32_t *valid_count_data, const int32_t batchs, const int32_t n, const int32_t k, const int32_t score_threshold){
  memset(y_data, -1, sizeof(int)*n*k*batchs);
  for(int32_t i = 0; i < batchs; i++){
      int32_t y_index = 0;
      const int32_t *input = x_data + i * n * k;
      int32_t *output = y_data + i * n * k;
      for(int32_t j = 0; j < n; j++){
          const int32_t *row = input + j * k;
          if(row[1] > score_threshold){
              memcpy(&output[y_index * k], row, k * sizeof(int32_t));
              y_index += 1;
          }
      }
      valid_count_data[i] = y_index;
      if(y_index < n){
          memset(&output[y_index * k], -1, (n-y_index) * k * sizeof(int32_t));
      }
  }
}

void get_valid_count_fpga(const int32_t *x_data, int32_t *y_data, int32_t *valid_count_data, const int32_t batch, const int32_t n, const int32_t k, const int32_t score_threshold){
  cl_int code;
  cl_kernel kernel = clCreateKernel(program, "get_valid_count", &code);
  cl_mem bufI = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(int)*n*batch*k, NULL, &code);
  cl_mem bufV = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(int)*batch, NULL, &code);
  cl_mem bufO = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(int)*n*batch*k, NULL, &code);

  clEnqueueWriteBuffer(queue, bufI, CL_TRUE, 0, sizeof(int)*n*batch*k, x_data, 0, nullptr, nullptr);
  int zero = -1;
  clEnqueueFillBuffer(queue, bufO, &zero, sizeof(int), 0, sizeof(int)*n*batch*k, 0, NULL, NULL);
  int index = 0;
  clSetKernelArg(kernel, index++, sizeof(cl_mem), (void*)&bufI);
  clSetKernelArg(kernel, index++, sizeof(cl_mem), (void*)&bufV);
  clSetKernelArg(kernel, index++, sizeof(cl_mem), (void*)&bufO);
  clSetKernelArg(kernel, index++, sizeof(int), (void*)&batch);
  clSetKernelArg(kernel, index++, sizeof(int), (void*)&n);
  clSetKernelArg(kernel, index++, sizeof(int), (void*)&k);
  clSetKernelArg(kernel, index++, sizeof(int), (void*)&score_threshold);
  clEnqueueTask(queue, kernel, 0, NULL, NULL);

  clEnqueueReadBuffer(queue, bufO, CL_TRUE, 0, sizeof(int) * n*batch*k, y_data, 0, nullptr, nullptr); 
  clEnqueueReadBuffer(queue, bufV, CL_TRUE, 0, sizeof(int) * batch, valid_count_data, 0, nullptr, nullptr); 
}

int main(){
  init_opencl("ops.xclbin");
  const int n = 10;
  const int k = 6;
  const int batch = 1;
  const int score_threshold = 10;

  int *inputs = new int[batch * n * k];
  int *outputs = new int [batch * n * k];
  int *outputs2 = new int [batch * n * k];
  int *valid_count = new int [batch];
  int *valid_count2 = new int [batch];

  for(int i = 0; i < n*k*batch; i++){
    inputs[i] = i % 127;
  }

  get_valid_count(inputs, outputs, valid_count, batch, n, k, score_threshold);
  get_valid_count_fpga(inputs, outputs2, valid_count2, batch, n, k, score_threshold);
  
  verify(valid_count, valid_count2, batch);
  verify(outputs, outputs2, batch*n*k);
  return 0;
}
