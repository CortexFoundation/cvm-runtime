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
    is_init = true;
    //openclDeviceAPI->CompileProgram(kernel_str);
  }
}

//#define CVM_OPENCL_PRINT_RESULT
void print_to_file(const void *buffer, const int n, const char*filename){
#ifdef CVM_OPENCL_PRINT_RESULT
  printf("save to file\n");
  int *data = new int[n];
  clEnqueueReadBuffer(openclDeviceAPI->queue, (cl_mem)buffer, CL_TRUE, 0, n*sizeof(int), data, 0, NULL, NULL);
  FILE *fp = fopen(filename, "a+");
  for(int i = 0; i < n && i < 1000; i++){
    fprintf(fp, "%d ", data[i]);
  }
  fprintf(fp, "\n");
  fclose(fp);
  delete data;
#endif
}

cl_kernel get_kernel(const std::string& kernel_name){
  init();
  cl_int ret;
  static std::map<std::string, cl_kernel> kernel_map;
  if(kernel_map.count(kernel_name) > 0){
    return kernel_map[kernel_name];
  } 

  cl_kernel clkernel = clCreateKernel(openclDeviceAPI->program, kernel_name.c_str(), &ret);
  OPENCL_CHECK_ERROR(ret);
  
  kernel_map[kernel_name] = clkernel;
  return clkernel; 
}

void exe_kernel(cl_kernel kernel, int32_t n){
  //clEnqueueTask(openclDeviceAPI->queue, kernel,0, NULL, NULL);
  size_t local_size = 256;
  size_t global_size = (n + 255) / 256 * 256;
  clEnqueueNDRangeKernel(openclDeviceAPI->queue, kernel,1, NULL, &global_size, &local_size, 0, NULL, NULL); 
#ifdef PROFILE
  clFlush(openclDeviceAPI->queue);
#endif 
}

void exe_kernel(cl_kernel kernel){
  clEnqueueTask(openclDeviceAPI->queue, kernel,0, NULL, NULL);
#ifdef PROFILE
  clFlush(openclDeviceAPI->queue);
#endif 
}

void opencl_elemwise_add(void *a, void *b, void *c, uint64_t n){
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
    const int dilation_h, const int dilation_w, bool use_bias,
    void *ext_space){
  cl_kernel kernel = use_bias == false ? get_kernel("conv") : get_kernel("conv_bias");

  int index = 0;
  clSetKernelArg(kernel, index++, sizeof(cl_mem), (void*)&input);
  clSetKernelArg(kernel, index++, sizeof(cl_mem), (void*)&weight);
  if(use_bias){
    clSetKernelArg(kernel, index++, sizeof(cl_mem), (void*)&bias);
  }
  clSetKernelArg(kernel, index++, sizeof(cl_mem), (void*)&output);
  clSetKernelArg(kernel, index++, sizeof(int), (void*)&batch);
  clSetKernelArg(kernel, index++, sizeof(int), (void*)&c);
  clSetKernelArg(kernel, index++, sizeof(int), (void*)&h);
  clSetKernelArg(kernel, index++, sizeof(int), (void*)&w);
  clSetKernelArg(kernel, index++, sizeof(int), (void*)&oc);
  clSetKernelArg(kernel, index++, sizeof(int), (void*)&kh);
  clSetKernelArg(kernel, index++, sizeof(int), (void*)&kw);
  clSetKernelArg(kernel, index++, sizeof(int), (void*)&oh);
  clSetKernelArg(kernel, index++, sizeof(int), (void*)&ow);
  //printf("exe conv2d: (%d %d %d %d),(%d %d %d), (%d %d), (%d %d), (%d %d), %d %d\n", batch, c, h, w, oc, kh, kw, oh, ow, pad_h, pad_w, stride_h, stride_w, dilation_h, dilation_w);
  exe_kernel(kernel);
  print_to_file(input, batch*h*w*c, "/media/nvme/data/mnist/conv_x.txt");
  print_to_file(output, batch*oh*ow*oc, "/media/nvme/data/mnist/conv.txt");
//
  
  //const int M = oc;
  //const int K = c * kh * kw;
  //const int N = oh * ow;

  //cl_kernel int32_to_int8 = get_kernel("int32_to_int8"); 
  //cl_kernel im2col = get_kernel("im2col");
  //cl_kernel gemm = get_kernel("gemm");

  //int index = 0;
  //int n = M*K;
  //clSetKernelArg(int32_to_int8, index++, sizeof(cl_mem), (void*)&weight);
  //clSetKernelArg(int32_to_int8, index++, sizeof(cl_mem), (void*)&ext_space);
  //clSetKernelArg(int32_to_int8, index++, sizeof(int), (void*)&n);
  //exe_kernel(int32_to_int8);

  //index = 0;
  //int offset = M*K;
  //n = c *oh *ow;
  //clSetKernelArg(im2col, index++, sizeof(cl_mem), (void*)&input);
  //clSetKernelArg(im2col, index++, sizeof(cl_mem), (void*)&ext_space);
  //clSetKernelArg(im2col, index++, sizeof(int), (void*)&n);
  //clSetKernelArg(im2col, index++, sizeof(int), (void*)&h);
  //clSetKernelArg(im2col, index++, sizeof(int), (void*)&w);
  //clSetKernelArg(im2col, index++, sizeof(int), (void*)&kh);
  //clSetKernelArg(im2col, index++, sizeof(int), (void*)&kw);
  //clSetKernelArg(im2col, index++, sizeof(int), (void*)&pad_h);
  //clSetKernelArg(im2col, index++, sizeof(int), (void*)&pad_w);
  //clSetKernelArg(im2col, index++, sizeof(int), (void*)&stride_h);
  //clSetKernelArg(im2col, index++, sizeof(int), (void*)&stride_w);
  //clSetKernelArg(im2col, index++, sizeof(int), (void*)&dilation_h);
  //clSetKernelArg(im2col, index++, sizeof(int), (void*)&dilation_w);
  //clSetKernelArg(im2col, index++, sizeof(int), (void*)&oh);
  //clSetKernelArg(im2col, index++, sizeof(int), (void*)&ow);
  //clSetKernelArg(im2col, index++, sizeof(int), (void*)&offset);
  //exe_kernel(im2col);

  //index = 0;
  //clSetKernelArg(gemm, index++, sizeof(cl_mem), (void*)&ext_space);
  //clSetKernelArg(gemm, index++, sizeof(cl_mem), (void*)&output);
  //clSetKernelArg(gemm, index++, sizeof(int), (void*)&M);
  //clSetKernelArg(gemm, index++, sizeof(int), (void*)&K);
  //clSetKernelArg(gemm, index++, sizeof(int), (void*)&N);
  //exe_kernel(gemm);

}

void opencl_max_pool2d(const void* input, void* output,
	const int batch, const int c, const int h, const int w,
	const int kh, const int kw,   
	const int oh, const int ow,
	const int pad_h, const int pad_w,             
	const int stride_h, const int stride_w){      
  cl_kernel kernel = get_kernel("pool");

  clSetKernelArg(kernel, 0, sizeof(cl_mem), (void*)&input);
  clSetKernelArg(kernel, 1, sizeof(cl_mem), (void*)&output);
  clSetKernelArg(kernel, 2, sizeof(int), (void*)&batch);
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

  //printf("exe max pool2d\n");
  exe_kernel(kernel);
  print_to_file(output, batch*c*oh*ow, "/media/nvme/data/mnist/pool.txt");
}

void opencl_cvm_clip(const void *input, void*output, const int n, const int precision){
  const int32_t min = -(((int64_t)1 << (precision-1))-1);
  const int32_t max = -min;

  cl_kernel kernel = get_kernel("cvm_clip");

  clSetKernelArg(kernel, 0, sizeof(cl_mem), (void*)&input);
  clSetKernelArg(kernel, 1, sizeof(cl_mem), (void*)&output);
  clSetKernelArg(kernel, 2, sizeof(int), (void*)&n);
  clSetKernelArg(kernel, 3, sizeof(int), (void*)&min);
  clSetKernelArg(kernel, 4, sizeof(int), (void*)&max);

  //printf("exe cvm clip %d %d %d\n", n, min, max);
  exe_kernel(kernel);
  print_to_file(output, n, "/media/nvme/data/mnist/cvm_clip.txt");
}

void opencl_cvm_right_shift(const void *input, void *output, const int shift_b, const int n, const int precision){
  const int32_t min = -(((int64_t)1 << (precision - 1)) - 1);
  const int32_t max = -min;

  cl_kernel kernel = get_kernel("cvm_right_shift");

  clSetKernelArg(kernel, 0, sizeof(cl_mem), (void*)&input);
  clSetKernelArg(kernel, 1, sizeof(cl_mem), (void*)&output);
  clSetKernelArg(kernel, 2, sizeof(int), (void*)&shift_b);
  clSetKernelArg(kernel, 3, sizeof(int), (void*)&n);
  clSetKernelArg(kernel, 4, sizeof(int), (void*)&min);
  clSetKernelArg(kernel, 5, sizeof(int), (void*)&max);

  //printf("exe cvm_right_shift %d %d %d\n", n, min, max);
  exe_kernel(kernel);
  print_to_file(input, n, "/media/nvme/data/mnist/cvm_right_shift_x.txt");
  print_to_file(output, n, "/media/nvme/data/mnist/cvm_right_shift.txt");
}

void opencl_relu(const void* input, void*output, const int n){
  cl_kernel kernel = get_kernel("relu");

  clSetKernelArg(kernel, 0, sizeof(cl_mem), (void*)&input);
  clSetKernelArg(kernel, 1, sizeof(cl_mem), (void*)&output);
  clSetKernelArg(kernel, 2, sizeof(int), (void*)&n);
  //printf("exe relu\n");
  exe_kernel(kernel);
  print_to_file(input, n, "/media/nvme/data/mnist/relu_x.txt");
  print_to_file(output, n, "/media/nvme/data/mnist/relu.txt");
}

void opencl_flatten(const void* input, void*output, const int n){
  init();

  if(input == output) return;
  clEnqueueCopyBuffer(openclDeviceAPI->queue, (cl_mem)input, (cl_mem)output, 0, 0, n * sizeof(int), 0, NULL, NULL);
}

void opencl_broadcast_mul(const void *a, const void* b, void *c, const int n){
  cl_kernel kernel = get_kernel("broadcast_mul");

  clSetKernelArg(kernel, 0, sizeof(cl_mem), (void*)&a);
  clSetKernelArg(kernel, 1, sizeof(cl_mem), (void*)&b);
  clSetKernelArg(kernel, 2, sizeof(cl_mem), (void*)&c);
  clSetKernelArg(kernel, 3, sizeof(int), (void*)&n);

  //printf("exe broadcast_mul\n");
  exe_kernel(kernel);
  print_to_file(a, n, "/media/nvme/data/mnist/broadcast_mul_a.txt");
  print_to_file(b, 1, "/media/nvme/data/mnist/broadcast_mul_b.txt");
  print_to_file(c, n, "/media/nvme/data/mnist/broadcast_mul.txt");
}
  
void opencl_dense(const void *a, const void *b, const void *bias, void *c, const int M, const int N, const int K, bool use_bias){
  cl_kernel kernel = use_bias == false ? get_kernel("dense") : get_kernel("dense_bias");

  int index = 0;
  clSetKernelArg(kernel, index++, sizeof(cl_mem), (void*)&a);
  clSetKernelArg(kernel, index++, sizeof(cl_mem), (void*)&b);
  if(use_bias)
    clSetKernelArg(kernel, index++, sizeof(cl_mem), (void*)&bias);
  clSetKernelArg(kernel, index++, sizeof(cl_mem), (void*)&c);
  clSetKernelArg(kernel, index++, sizeof(int), (void*)&M);
  clSetKernelArg(kernel, index++, sizeof(int), (void*)&N);
  clSetKernelArg(kernel, index++, sizeof(int), (void*)&K);

  //printf("exe dense\n");
  exe_kernel(kernel);
}
#endif
  
