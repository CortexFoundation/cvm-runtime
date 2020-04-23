#include <CL/opencl.h>
#include <iostream>
#include <string>
#include <vector>
#include <iostream>
#include <fstream>

#include <string.h>
#include "util.hpp"
using namespace std;

void concatenate_cpu(vector<DLTensor*>& args, const int axis){
  int len = args.size();
  DLTensor *input0 = args[0];
  DLTensor *out = args[--len];
  int32_t *out_data = static_cast<int32_t*>(out->data);
  int32_t ndim = static_cast<int32_t>(input0->ndim);

  int64_t y_size = 1;
  for (int i = 0; i < axis; ++i) y_size *= out->shape[i];
  int32_t axis_batch = 1;
  for (int i = axis+1; i < ndim; ++i) axis_batch *= out->shape[i];

  int64_t y_start_idx = 0;
  int64_t y_axis_batch = out->shape[axis] * axis_batch;
  for (int m = 0; m < len; ++m) {
    DLTensor* input = args[m];
    int32_t* Ix = static_cast<int32_t*>(input->data);
    auto x_axis_batch = input->shape[axis] * axis_batch;

    for (int64_t y_iter = 0; y_iter < y_size; ++y_iter) {
      memcpy(out_data+y_iter*y_axis_batch+y_start_idx,
          Ix+y_iter*x_axis_batch,
          x_axis_batch*sizeof(int32_t));
    }

    y_start_idx += x_axis_batch;
  }
}

const int MAX_INPUT_SIZE = 1024;
void concatenate_fpga(vector<DLTensor*>& args, const int axis){
  cl_int code;
  int len = args.size();
  DLTensor *input0 = args[0];
  DLTensor *out = args[--len];
  int32_t *out_data = static_cast<int32_t*>(out->data);
  int32_t ndim = static_cast<int32_t>(input0->ndim);

  cl_kernel kernel = clCreateKernel(program, "concatenate", &code);
  cl_mem bufA = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(int)*MAX_INPUT_SIZE, NULL, &code);

  cl_mem bufC = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(int) * getSize(out), NULL, &code);

  int y_size = 1;
  for (int i = 0; i < axis; ++i) y_size *= out->shape[i];
  int axis_batch = 1;
  for (int i = axis+1; i < ndim; ++i) 
    axis_batch *= out->shape[i];

  int y_start_idx = 0;
  int y_axis_batch = out->shape[axis] * axis_batch;
  int ninput = len;
  for (int m = 0; m < ninput; ++m) {
    //void* Ix = inputs[m];
    clEnqueueWriteBuffer(queue, bufA, CL_TRUE, 0, sizeof(int)*getSize(args[m]), args[m]->data, 0, nullptr, nullptr);
    int x_axis_batch = args[m]->shape[axis] * axis_batch;

    int n = x_axis_batch * y_size;
  
    int index = 0;
    clSetKernelArg(kernel, index++, sizeof(cl_mem), (void*)&bufA);
    clSetKernelArg(kernel, index++, sizeof(cl_mem), (void*)&bufC);
    clSetKernelArg(kernel, index++, sizeof(int), (void*)&y_axis_batch);
    clSetKernelArg(kernel, index++, sizeof(int), (void*)&x_axis_batch);
    clSetKernelArg(kernel, index++, sizeof(int), (void*)&n);
    clSetKernelArg(kernel, index++, sizeof(int), (void*)&y_start_idx);
    clEnqueueTask(queue, kernel, 0, NULL, NULL);
    y_start_idx += x_axis_batch;
  }

  clEnqueueReadBuffer(queue, bufC, CL_TRUE, 0, sizeof(int) * getSize(out), out_data, 0, nullptr, nullptr); 
}

int main(){
  init_opencl("ops.xclbin");

  vector<DLTensor*> args, args2;
  DLTensor in0(2), in1(2), out(2), out2(2);
  int axis = 1;
  in0.shape[0] = 3;
  in0.shape[1] = 4;
  in1.shape[0] = 3;
  in1.shape[1] = 5;
  out.shape[0] = 3;
  out.shape[1] = 9;
  out2.shape[0] = 3;
  out2.shape[1] = 9;

  in0.data = new int[getSize(&in0)];
  in1.data = new int[getSize(&in1)];
  out.data = new int[getSize(&out)];
  out2.data = new int[getSize(&out2)];
  for(int i = 0; i < getSize(&in0); i++){
    in0.data[i] = i % 127;
  }
  for(int i = 0; i < getSize(&in1); i++){
    in1.data[i] = i % 127;
  }

  args.push_back(&in0);
  args.push_back(&in1);
  args.push_back(&out);
  concatenate_cpu(args, axis);

  args2.push_back(&in0);
  args2.push_back(&in1);
  args2.push_back(&out2);
  concatenate_fpga(args2, axis);


  verify(out.data, out2.data, getSize(&out));

}
