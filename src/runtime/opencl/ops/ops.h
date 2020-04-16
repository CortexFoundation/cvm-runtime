#ifndef OPENCL_OPS_H
#define OPENCL_OPS_H

#include <string>

#include <cvm/dlpack.h>
#include <cvm/runtime/device_api.h>
#include "../devapi/opencl_device_api.h"

const std::string kernel_str = R"(
  __kernel void elemwise_add(__global const int* a, __global const int* b, __global int *c, int n){
    int gid = get_global_id(0);
    if(gid < n){
      c[gid] = a[gid] + b[gid];
    }
  } 
)";

const DLContext ctx = {kDLOpenCL, 0};

cvm::runtime::OpenCLDeviceAPI *openclDeviceAPI = NULL;

void init(){

  static bool is_init = false;
  if(!is_init){
    openclDeviceAPI = (cvm::runtime::OpenCLDeviceAPI*)cvm::runtime::DeviceAPI::Get(ctx);
    //openclDeviceAPI->CompileProgram(kernel_str);
  }
}

cl_kernel get_kernel(const char* kernel_name){
  cl_int ret;
  cl_kernel clkernel = clCreateKernel(openclDeviceAPI->program, kernel_name, &ret);
  OPENCL_CHECK_ERROR(ret);
  
  return clkernel; 
}

void exe_kernel(cl_kernel kernel, int32_t n){
  //clEnqueueTask(openclDeviceAPI->queue, kernel,0, NULL, NULL);
  size_t local_size = 256;
  size_t global_size = (n + 255) / 256 * 256;
  clEnqueueNDRangeKernel(openclDeviceAPI->queue, kernel,1, NULL, &global_size, &local_size, 0, NULL, NULL); 
}

void exe_kernel(cl_kernel kernel){
  clEnqueueTask(openclDeviceAPI->queue, kernel,0, NULL, NULL);
}

void opencl_elemwise_add(void *a, void *b, void *c, uint64_t n){
  init();
  cl_kernel kernel = get_kernel("vadd");
  clSetKernelArg(kernel, 0, sizeof(cl_mem), (void*)&a);
  clSetKernelArg(kernel, 1, sizeof(cl_mem), (void*)&b);
  clSetKernelArg(kernel, 2, sizeof(cl_mem), (void*)&c);
  clSetKernelArg(kernel, 3, sizeof(int), (void*)&n);

  //exe_kernel(kernel, n);
  exe_kernel(kernel);
}

void opencl_conv2d(void* input, void *weight, void *bias, void *output,
    const int batch, const int c, const int h, const int w,
    const int oc, const int kh, const int kw,
    const int oh, const int ow,
    const int pad_h, const int pad_w,
    const int stride_h, const int stride_w,
    const int dilation_h, const int dilation_w){
  init();
  cl_kernel kernel = get_kernel("conv");

  clSetKernelArg(kernel, 0, sizeof(cl_mem), input);
  clSetKernelArg(kernel, 1, sizeof(cl_mem), weight);
  clSetKernelArg(kernel, 2, sizeof(cl_mem), output);
  clSetKernelArg(kernel, 3, sizeof(int), (void*)&batch);
  clSetKernelArg(kernel, 4, sizeof(int), (void*)&c);
  clSetKernelArg(kernel, 5, sizeof(int), (void*)&h);
  clSetKernelArg(kernel, 6, sizeof(int), (void*)&w);
  clSetKernelArg(kernel, 7, sizeof(int), (void*)&oc);
  clSetKernelArg(kernel, 8, sizeof(int), (void*)&oh);
  clSetKernelArg(kernel, 9, sizeof(int), (void*)&ow);

  exe_kernel(kernel);
}

void opencl_max_pool2d(const int* input, int* output,
	const int batch, const int c, const int h, const int w,
	const int kh, const int kw,   
	const int oh, const int ow,
	const int pad_h, const int pad_w,             
	const int stride_h, const int stride_w){      
  init();
  cl_kernel kernel = get_kernel("pool");

  clSetKernelArg(kernel, 0, sizeof(cl_mem), (void*)&bufi);
  clSetKernelArg(kernel, 1, sizeof(cl_mem), (void*)&bufo);
  clSetKernelArg(kernel, 2, sizeof(int), (void*)&n);
  clSetKernelArg(kernel, 3, sizeof(int), (void*)&c);
  clSetKernelArg(kernel, 4, sizeof(int), (void*)&h);
  clSetKernelArg(kernel, 5, sizeof(int), (void*)&w);
  clSetKernelArg(kernel, 6, sizeof(int), (void*)&kh);
  clSetKernelArg(kernel, 7, sizeof(int), (void*)&kw);
  clSetKernelArg(kernel, 8, sizeof(int), (void*)&oh);
  clSetKernelArg(kernel, 9, sizeof(int), (void*)&ow);
  clSetKernelArg(kernel, 10, sizeof(int), (void*)&pad_h);
  clSetKernelArg(kernel, 11, sizeof(int), (void*)&pad_w);
  clSetKernelArg(kernel, 12, sizeof(int), (void*)&stride_h);
  clSetKernelArg(kernel, 13, sizeof(int), (void*)&stride_w);

  exe_kernel(kernel);
}

void opencl_cvm_clip(const int *input, int*output, const int n, const int precision){
  const int32_t min = -(((int64_t)1 << (precision-1))-1);
  const int32_t max = -min;

  init();
  cl_kernel kernel = get_kernel("cvm_clip");

  clSetKernelArg(kernel, 0, sizeof(cl_mem), input);
  clSetKernelArg(kernel, 1, sizeof(cl_mem), output);
  clSetKernelArg(kernel, 2, sizeof(int), (void*)&n);
  clSetKernelArg(kernel, 3, sizeof(int), (void*)&min);
  clSetKernelArg(kernel, 4, sizeof(int), (void*)&max);

  exe_kernel(kernel);
}

void opencl_cvm_right_shift(const int *input, int *output, const int shift_b, const int n, const int precision){
  const int32_t minV = -(((int64_t)1 << (precision - 1)) - 1);
  const int32_t maxV = -minV;

  init();
  cl_kernel kernel = get_kernel("cvm_right_shift");

  clSetKernelArg(kernel, 0, sizeof(cl_mem), input);
  clSetKernelArg(kernel, 1, sizeof(cl_mem), output);
  clSetKernelArg(kernel, 2, sizeof(int), (void*)&shift_b);
  clSetKernelArg(kernel, 3, sizeof(int), (void*)&n);
  clSetKernelArg(kernel, 4, sizeof(int), (void*)&min);
  clSetKernelArg(kernel, 5, sizeof(int), (void*)&max);

  exe_kernel();
}

void opencl_relu(const int* input, int*output, const int n){
  init();
  cl_kernel kernel = get_kernel("relu");

  clSetKernelArg(kernel, 0, sizeof(cl_mem), input);
  clSetKernelArg(kernel, 1, sizeof(cl_mem), output);
  clSetKernelArg(kernel, 2, sizeof(int), (void*)&n);
  exe_kernel();
}

void opencl_flatten(const int* input, int*output, const int n){
  init();

  if(input == output) return;
  clEnqueueCopyBuffer(openclDeviceAPI->queue, input, output, 0, 0, n * sizeof(int), 0, NULL, NULL);
}

#endif
  
#endif
  
