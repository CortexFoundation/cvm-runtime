#ifndef OPENCL_OPS_H
#define OPENCL_OPS_H

#include <string>

#include <cvm/dlpack.h>
#include <cvm/runtime/device_api.h>
#include "../devapi/opencl_device_api.h"
#include <omp.h>
#include <iostream>
#include <memory>
#include <cvm/top/nn.h>

const std::string kernel_str = R"(
  __kernel void elemwise_add(__global const int* a, __global const int* b, __global int *c, int n){
    int gid = get_global_id(0);
    if(gid < n){
      c[gid] = a[gid] + b[gid];
    }
  } 
)";

#define MAX_DIM 6
template<typename T1, typename T2>
inline void get_opencl_shape(const T1 *ishape, const int dim, T2 *oshape){
  int shift = MAX_DIM - dim;
  for(int i = 0; i < MAX_DIM; i++){
    oshape[i] = 1;
    if(i >= shift){
      oshape[i] = ishape[i - shift];
    }
  }
}

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
const std::string root = "/media/nvme/data/ssd/";
void print_to_file(const void *buffer, const int n, const std::string filename){
#ifdef CVM_OPENCL_PRINT_RESULT
  clFinish(openclDeviceAPI->queue);
  std::string path = root + filename;
  int *data = new int[n];
  OPENCL_CALL(clEnqueueReadBuffer(openclDeviceAPI->queue, (cl_mem)buffer, CL_TRUE, 0, n*sizeof(int), data, 0, NULL, NULL));
  FILE *fp = fopen(path.c_str(), "a+");
  if(fp == NULL){
    printf("open file %s failed\n", path.c_str()); 
    assert(false);
  }
  fprintf(fp, "%d\n", n);
  int min = data[0];
  int max = data[0];
  for(int i = 1; i < n; i++){
    if(min > data[i]) min = data[i];
    if(max < data[i]) max = data[i];
  }
  fprintf(fp, "%d %d\n", min, max);
  printf("%d %d\n", min, max);
  for(int i = 0, j = 0; i < n && j < 1000; i++){
    if(data[i] != 0){
      fprintf(fp, "%d ", data[i]);
      j++;
      if(j < 10) printf("%d ", data[i]);
    }
  }
  printf("\n");
  fprintf(fp, "\n");
  fclose(fp);
  delete data;
#endif
}
void save_nms(const void *buffer, const int n, const std::string filename){
#ifdef CVM_OPENCL_PRINT_RESULT
  std::string path = root + filename;
  int *data = new int[n*6];
  clEnqueueReadBuffer(openclDeviceAPI->queue, (cl_mem)buffer, CL_TRUE, 0, n*6*sizeof(int), data, 0, NULL, NULL);
  FILE *fp = fopen(path.c_str(), "a+");
  if(fp == NULL){
    printf("open file %s failed\n", path.c_str()); 
    assert(false);
  }
  for(int i = 0; i < n; i++){
    for(int k = 0; k < 6; k++)
      fprintf(fp, "%d ", (int)data[i*6+k]);
    fprintf(fp, "\n");
  }
  fprintf(fp, "\n");
  fclose(fp);
  delete data;
#endif
}

cl_kernel get_kernel(const std::string& kernel_name){
  init();
  //printf("get %s\n", kernel_name.c_str());
  cl_int ret;
  static std::map<std::string, cl_kernel> kernel_map;
  if(kernel_map.count(kernel_name) > 0){
    return kernel_map[kernel_name];
  } 

  cl_kernel clkernel = clCreateKernel(openclDeviceAPI->program, kernel_name.c_str(), &ret);
  OPENCL_CHECK_ERROR(ret);
  
  kernel_map[kernel_name] = clkernel;
  //TODO(zkh) release kernel
  return clkernel; 
}

void set_kernel_args(const cl_kernel& kernel, std::vector<cl_mem>& mems, std::vector<int>& iargs){
  int index = 0;
  for(size_t i = 0; i < mems.size(); i++){
    clSetKernelArg(kernel, index++, sizeof(cl_mem), (void*)&mems[i]); 
  }
  for(size_t i = 0; i < iargs.size(); i++){
    clSetKernelArg(kernel, index++, sizeof(int), (void*)&iargs[i]);
  }
}

void exe_kernel(cl_kernel kernel, int n){
  //clEnqueueTask(openclDeviceAPI->queue, kernel,0, NULL, NULL);
  size_t local_size = 256;
  size_t global_size = (n + 255) / 256 * 256;
  clEnqueueNDRangeKernel(openclDeviceAPI->queue, kernel,1, NULL, &global_size, &local_size, 0, NULL, NULL); 
#ifdef PROFILE
  clFlush(openclDeviceAPI->queue);
#endif 
}

void exe_kernel(cl_kernel kernel){
  OPENCL_CALL(clEnqueueTask(openclDeviceAPI->queue, kernel,0, NULL, NULL));
  //clFinish(openclDeviceAPI->queue);
#ifdef PROFILE
  clFlush(openclDeviceAPI->queue);
#endif 
}

void opencl_elemwise(void *a, void *b, void *c, const int n, const int type){
  cl_kernel kernel = get_kernel("elemwise");
  clSetKernelArg(kernel, 0, sizeof(cl_mem), (void*)&a);
  clSetKernelArg(kernel, 1, sizeof(cl_mem), (void*)&b);
  clSetKernelArg(kernel, 2, sizeof(cl_mem), (void*)&c);
  clSetKernelArg(kernel, 3, sizeof(int), (void*)&n);
  clSetKernelArg(kernel, 4, sizeof(int), (void*)&type);

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
    void *ext_space, const int ext_space_size){
  //cl_kernel kernel = use_bias == false ? get_kernel("conv") : get_kernel("conv_bias");

  //int index = 0;
  //clSetKernelArg(kernel, index++, sizeof(cl_mem), (void*)&input);
  //clSetKernelArg(kernel, index++, sizeof(cl_mem), (void*)&weight);
  //if(use_bias){
  //  clSetKernelArg(kernel, index++, sizeof(cl_mem), (void*)&bias);
  //}
  //clSetKernelArg(kernel, index++, sizeof(cl_mem), (void*)&output);
  //clSetKernelArg(kernel, index++, sizeof(int), (void*)&batch);
  //clSetKernelArg(kernel, index++, sizeof(int), (void*)&c);
  //clSetKernelArg(kernel, index++, sizeof(int), (void*)&h);
  //clSetKernelArg(kernel, index++, sizeof(int), (void*)&w);
  //clSetKernelArg(kernel, index++, sizeof(int), (void*)&oc);
  //clSetKernelArg(kernel, index++, sizeof(int), (void*)&kh);
  //clSetKernelArg(kernel, index++, sizeof(int), (void*)&kw);
  //clSetKernelArg(kernel, index++, sizeof(int), (void*)&oh);
  //clSetKernelArg(kernel, index++, sizeof(int), (void*)&ow);
  ////printf("exe conv2d: (%d %d %d %d),(%d %d %d), (%d %d), (%d %d), (%d %d), %d %d\n", batch, c, h, w, oc, kh, kw, oh, ow, pad_h, pad_w, stride_h, stride_w, dilation_h, dilation_w);
  //exe_kernel(kernel);
//
  
  const int M = oc;
  const int K = c * kh * kw;
  const int N = oh * ow;

  //cl_kernel int32_to_int8 = get_kernel("int32_to_int8"); 
  cl_kernel im2col = get_kernel("im2col");
  cl_kernel gemm = use_bias ? get_kernel("gemm_bias") : get_kernel("gemm");

  int zero = 0;
  clEnqueueFillBuffer(openclDeviceAPI->queue, (cl_mem)ext_space, &zero, sizeof(int), 0, sizeof(int)*ext_space_size, 0, NULL, NULL);

  int index = 0;
  //const int TM = (M+63)/64*64;
  //const int TK = (K+63)/64*64;
  //const int TN = (N+63)/64*64;
 // int offset = TK*TN;
  int n = M*K;
 // clSetKernelArg(int32_to_int8, index++, sizeof(cl_mem), (void*)&weight);
 // clSetKernelArg(int32_to_int8, index++, sizeof(cl_mem), (void*)&ext_space);
 // clSetKernelArg(int32_to_int8, index++, sizeof(int), (void*)&M);
 // clSetKernelArg(int32_to_int8, index++, sizeof(int), (void*)&K);
 // clSetKernelArg(int32_to_int8, index++, sizeof(int), (void*)&offset);
 // exe_kernel(int32_to_int8);
 // clFlush(openclDeviceAPI->queue);

  for(int b = 0; b < batch; b++){
    int i_offset = b * c * h * w;
    int o_offset = b * oc * oh * ow;

    //static double im2col_time = 0;
    //static double gemm_time = 0;
    //double start = omp_get_wtime();

    index = 0;
    //int offset = TM*TK;
    n = c *oh *ow;
    //printf("%d %d %d %d, %d %d %d, %d %d, %d %d, %d %d\n", batch, c, h, w, oc, kh, kw, pad_h, pad_w, stride_h, stride_w, dilation_h, dilation_w);
    clSetKernelArg(im2col, index++, sizeof(cl_mem), (void*)&input);
    clSetKernelArg(im2col, index++, sizeof(cl_mem), (void*)&ext_space);
    clSetKernelArg(im2col, index++, sizeof(int), (void*)&n);
    clSetKernelArg(im2col, index++, sizeof(int), (void*)&h);
    clSetKernelArg(im2col, index++, sizeof(int), (void*)&w);
    clSetKernelArg(im2col, index++, sizeof(int), (void*)&kh);
    clSetKernelArg(im2col, index++, sizeof(int), (void*)&kw);
    clSetKernelArg(im2col, index++, sizeof(int), (void*)&pad_h);
    clSetKernelArg(im2col, index++, sizeof(int), (void*)&pad_w);
    clSetKernelArg(im2col, index++, sizeof(int), (void*)&stride_h);
    clSetKernelArg(im2col, index++, sizeof(int), (void*)&stride_w);
    clSetKernelArg(im2col, index++, sizeof(int), (void*)&dilation_h);
    clSetKernelArg(im2col, index++, sizeof(int), (void*)&dilation_w);
    clSetKernelArg(im2col, index++, sizeof(int), (void*)&oh);
    clSetKernelArg(im2col, index++, sizeof(int), (void*)&ow);
    clSetKernelArg(im2col, index++, sizeof(int), (void*)&i_offset);
    exe_kernel(im2col);
    print_to_file(ext_space, K*N, "im2col.txt");
    //clFlush(openclDeviceAPI->queue);
    //clFinish(openclDeviceAPI->queue);

    //double im2col_end = omp_get_wtime();

    //printf("%d %d %d\n", M, K, N);
    index = 0;
    clSetKernelArg(gemm, index++, sizeof(cl_mem), (void*)&weight);
    clSetKernelArg(gemm, index++, sizeof(cl_mem), (void*)&ext_space);
    if(use_bias)
      clSetKernelArg(gemm, index++, sizeof(cl_mem), (void*)&bias);
    clSetKernelArg(gemm, index++, sizeof(cl_mem), (void*)&output);
    clSetKernelArg(gemm, index++, sizeof(int), (void*)&M);
    clSetKernelArg(gemm, index++, sizeof(int), (void*)&K);
    clSetKernelArg(gemm, index++, sizeof(int), (void*)&N);
    clSetKernelArg(gemm, index++, sizeof(int), (void*)&o_offset);
    exe_kernel(gemm);
    //clFinish(openclDeviceAPI->queue);
    //double end = omp_get_wtime();
    //im2col_time += (double)(im2col_end - start);
    //gemm_time += (double)(end - im2col_end);
    //printf("im2col : %.4f, gemm: %.4f\n", im2col_time, gemm_time);
  }

  print_to_file(input, batch*h*w*c, "conv_x.txt");
  print_to_file(output, batch*oh*ow*oc, "conv.txt");
}
void opencl_groupwise_conv2d(
   void *x_data, int32_t n_batch, int32_t in_channels, int32_t x_h, int32_t x_w,
   void *w_data, int32_t filter_c, int32_t filter_h, int32_t filter_w,
   void *y_data, int32_t out_channels, int32_t o_h, int32_t o_w,
   void *b_data,
   int32_t pad_h, int pad_w, int32_t stride_h, int32_t stride_w, int32_t dilation_h, int32_t dilation_w,
   int32_t groups, 
   bool use_bias){
  cl_kernel kernel = use_bias ? get_kernel("groupwise_conv2d_bias") : get_kernel("groupwise_conv2d");
  
  //cl_kernel kernel = clCreateKernel(program, "groupwise_conv2d", &code);
  int index = 0;
  clSetKernelArg(kernel, index++, sizeof(cl_mem), (void*)&x_data);
  clSetKernelArg(kernel, index++, sizeof(cl_mem), (void*)&w_data);
  if(use_bias)
    clSetKernelArg(kernel, index++, sizeof(cl_mem), (void*)&b_data);
  clSetKernelArg(kernel, index++, sizeof(cl_mem), (void*)&y_data);
  clSetKernelArg(kernel, index++, sizeof(int), (void*)&n_batch);
  clSetKernelArg(kernel, index++, sizeof(int), (void*)&in_channels);
  clSetKernelArg(kernel, index++, sizeof(int), (void*)&x_h);
  clSetKernelArg(kernel, index++, sizeof(int), (void*)&x_w);
  clSetKernelArg(kernel, index++, sizeof(int), (void*)&filter_c);
  clSetKernelArg(kernel, index++, sizeof(int), (void*)&filter_h);
  clSetKernelArg(kernel, index++, sizeof(int), (void*)&filter_w);
  clSetKernelArg(kernel, index++, sizeof(int), (void*)&out_channels);
  clSetKernelArg(kernel, index++, sizeof(int), (void*)&o_h);
  clSetKernelArg(kernel, index++, sizeof(int), (void*)&o_w);
  clSetKernelArg(kernel, index++, sizeof(int), (void*)&pad_h);
  clSetKernelArg(kernel, index++, sizeof(int), (void*)&pad_w);
  clSetKernelArg(kernel, index++, sizeof(int), (void*)&stride_h);
  clSetKernelArg(kernel, index++, sizeof(int), (void*)&stride_w);
  clSetKernelArg(kernel, index++, sizeof(int), (void*)&dilation_h);
  clSetKernelArg(kernel, index++, sizeof(int), (void*)&dilation_w);
  clSetKernelArg(kernel, index++, sizeof(int), (void*)&groups);
  
  exe_kernel(kernel);
  print_to_file(y_data, n_batch * out_channels * o_h * o_w, "groupwise_conv.txt");
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
  print_to_file(output, batch*c*oh*ow, "pool.txt");
}

void opencl_cvm_clip(const void *input, void*output, const int n, const int precision){
  const int min = -(((int)1 << (precision-1))-1);
  const int max = -min;

  cl_kernel kernel = get_kernel("cvm_clip");

  clSetKernelArg(kernel, 0, sizeof(cl_mem), (void*)&input);
  clSetKernelArg(kernel, 1, sizeof(cl_mem), (void*)&output);
  clSetKernelArg(kernel, 2, sizeof(int), (void*)&n);
  clSetKernelArg(kernel, 3, sizeof(int), (void*)&min);
  clSetKernelArg(kernel, 4, sizeof(int), (void*)&max);

  //printf("exe cvm clip %d %d %d\n", n, min, max);
  exe_kernel(kernel);
  print_to_file(output, n, "cvm_clip.txt");
}

void opencl_clip(const void *input, void*output, const int n, const int max, const int min){
  cl_kernel kernel = get_kernel("cvm_clip");

  int index = 0;
  clSetKernelArg(kernel, index++, sizeof(cl_mem), (void*)&input);
  clSetKernelArg(kernel, index++, sizeof(cl_mem), (void*)&output);
  clSetKernelArg(kernel, index++, sizeof(int), (void*)&n);
  clSetKernelArg(kernel, index++, sizeof(int), (void*)&max);
  clSetKernelArg(kernel, index++, sizeof(int), (void*)&min);

  //printf("exe cvm clip %d %d %d\n", n, min, max);
  exe_kernel(kernel);
  print_to_file(output, n, "cvm_clip.txt");
}

void opencl_cvm_right_shift(const void *input, void *output, const int shift_b, const int n, const int precision){
  const int min = -(((int)1 << (precision - 1)) - 1);
  const int max = -min;

  cl_kernel kernel = get_kernel("cvm_shift");

  clSetKernelArg(kernel, 0, sizeof(cl_mem), (void*)&input);
  clSetKernelArg(kernel, 1, sizeof(cl_mem), (void*)&output);
  clSetKernelArg(kernel, 2, sizeof(int), (void*)&shift_b);
  clSetKernelArg(kernel, 3, sizeof(int), (void*)&n);
  clSetKernelArg(kernel, 4, sizeof(int), (void*)&min);
  clSetKernelArg(kernel, 5, sizeof(int), (void*)&max);
  int type = 0;
  clSetKernelArg(kernel, 6, sizeof(int), (void*)&type);

  //printf("exe cvm_right_shift %d %d %d\n", n, min, max);
  exe_kernel(kernel);
  print_to_file(input, n, "cvm_right_shift_x.txt");
  print_to_file(output, n, "cvm_right_shift.txt");
}
void opencl_cvm_left_shift(const void *input, void *output, const int shift_b, const int n, const int precision){
  cl_kernel kernel = get_kernel("cvm_shift");

  const int min = -(((int)1 << (precision - 1)) - 1);
  const int max = -min;
  int index = 0;
  clSetKernelArg(kernel, index++, sizeof(cl_mem), (void*)&input);
  clSetKernelArg(kernel, index++, sizeof(cl_mem), (void*)&output);
  clSetKernelArg(kernel, index++, sizeof(int), (void*)&shift_b);
  clSetKernelArg(kernel, index++, sizeof(int), (void*)&n);
  clSetKernelArg(kernel, index++, sizeof(int), (void*)&min);
  clSetKernelArg(kernel, index++, sizeof(int), (void*)&max);
  int type = 1;
  clSetKernelArg(kernel, index++, sizeof(int), (void*)&type);

  exe_kernel(kernel);
  print_to_file(input, n, "cvm_left_shift_x.txt");
  print_to_file(output, n, "cvm_left_shift.txt");
}

void opencl_relu(const void* input, void*output, const int n){
  cl_kernel kernel = get_kernel("relu");

  int index = 0;
  clSetKernelArg(kernel, index++, sizeof(cl_mem), (void*)&input);
  clSetKernelArg(kernel, index++, sizeof(cl_mem), (void*)&output);
  clSetKernelArg(kernel, index++, sizeof(int), (void*)&n);
  exe_kernel(kernel);

  print_to_file(output, n, "relu.txt");
  print_to_file(input, n, "relu_x.txt");
}

void opencl_flatten(const void* input, void*output, const int n){
  init();

  if(input == output) return;
  clEnqueueCopyBuffer(openclDeviceAPI->queue, (cl_mem)input, (cl_mem)output, 0, 0, n * sizeof(int), 0, NULL, NULL);
}

void opencl_broadcast(const void *a, const void* b, void *c, 
    const int64_t* a_shape, const int64_t *b_shape, const int64_t *c_shape,
    const int andim, const int bndim, const int cndim, const int asize, const int bsize, const int csize, const int type){
  cl_kernel kernel = get_kernel("broadcast");

  int ashape[MAX_DIM], bshape[MAX_DIM], cshape[MAX_DIM];
  get_opencl_shape(a_shape, andim, ashape);
  get_opencl_shape(b_shape, bndim, bshape);
  get_opencl_shape(c_shape, cndim, cshape);
  int index = 0;
  clSetKernelArg(kernel, index++, sizeof(cl_mem), (void*)&a);
  clSetKernelArg(kernel, index++, sizeof(cl_mem), (void*)&b);
  clSetKernelArg(kernel, index++, sizeof(cl_mem), (void*)&c);
  clSetKernelArg(kernel, index++, sizeof(int), (void*)&asize);
  clSetKernelArg(kernel, index++, sizeof(int), (void*)&bsize);
  clSetKernelArg(kernel, index++, sizeof(int), (void*)&csize);
  clSetKernelArg(kernel, index++, sizeof(int), (void*)&andim);
  clSetKernelArg(kernel, index++, sizeof(int), (void*)&bndim);
  clSetKernelArg(kernel, index++, sizeof(int), (void*)&cndim);
  clSetKernelArg(kernel, index++, sizeof(int), (void*)&ashape[0]);
  clSetKernelArg(kernel, index++, sizeof(int), (void*)&ashape[1]);
  clSetKernelArg(kernel, index++, sizeof(int), (void*)&ashape[2]);
  clSetKernelArg(kernel, index++, sizeof(int), (void*)&ashape[3]);
  clSetKernelArg(kernel, index++, sizeof(int), (void*)&ashape[4]);
  clSetKernelArg(kernel, index++, sizeof(int), (void*)&ashape[5]);

  clSetKernelArg(kernel, index++, sizeof(int), (void*)&bshape[0]);
  clSetKernelArg(kernel, index++, sizeof(int), (void*)&bshape[1]);
  clSetKernelArg(kernel, index++, sizeof(int), (void*)&bshape[2]);
  clSetKernelArg(kernel, index++, sizeof(int), (void*)&bshape[3]);
  clSetKernelArg(kernel, index++, sizeof(int), (void*)&bshape[4]);
  clSetKernelArg(kernel, index++, sizeof(int), (void*)&bshape[5]);

  clSetKernelArg(kernel, index++, sizeof(int), (void*)&cshape[0]);
  clSetKernelArg(kernel, index++, sizeof(int), (void*)&cshape[1]);
  clSetKernelArg(kernel, index++, sizeof(int), (void*)&cshape[2]);
  clSetKernelArg(kernel, index++, sizeof(int), (void*)&cshape[3]);
  clSetKernelArg(kernel, index++, sizeof(int), (void*)&cshape[4]);
  clSetKernelArg(kernel, index++, sizeof(int), (void*)&cshape[5]);

  clSetKernelArg(kernel, index++, sizeof(int), (void*)&type);
  exe_kernel(kernel);
  print_to_file(c, csize, "broadcast.txt");
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

void opencl_concatenate(void **inputs, int *ishapes, const int ninput, const int ndim, void *output,const int64_t* oshape, const int axis, int* axisSize){

  cl_kernel kernel = get_kernel("concatenate");

  int y_size = 1;
  for (int i = 0; i < axis; ++i) y_size *= oshape[i];
  int axis_batch = 1;
  for (int i = axis+1; i < ndim; ++i) axis_batch *= oshape[i];

  int y_start_idx = 0;
  int y_axis_batch = oshape[axis] * axis_batch;
  for (int m = 0; m < ninput; ++m) {
    void* Ix = inputs[m];
    int x_axis_batch = ishapes[m*ndim+axis] * axis_batch;
    int n = x_axis_batch * y_size;
    int index = 0;
    clSetKernelArg(kernel, index++, sizeof(cl_mem), (void*)&Ix);
    clSetKernelArg(kernel, index++, sizeof(cl_mem), (void*)&output);
    clSetKernelArg(kernel, index++, sizeof(int), (void*)&y_axis_batch);
    clSetKernelArg(kernel, index++, sizeof(int), (void*)&x_axis_batch);
    clSetKernelArg(kernel, index++, sizeof(int), (void*)&n);
    clSetKernelArg(kernel, index++, sizeof(int), (void*)&y_start_idx);
    exe_kernel(kernel);
    y_start_idx += x_axis_batch;
  }

  print_to_file(output, y_start_idx, "concat.txt");
}

#define REDUCE_MAX 0
#define REDUCE_SUM 1
void opencl_reduce(const void *x, void *y, const int xsize, const int ysize, const int64_t *xshape, const int64_t *yshape, const int* realAxis, const int* flag, const int *every_xdim_size, const int axis_size,const int xndim, const int yndim, const int axis_ndim, const int type){
  int dev_xshape[MAX_DIM], dev_yshape[MAX_DIM];
  int dev_every_xdim_size[MAX_DIM];
  int dev_flag[MAX_DIM], dev_axis[MAX_DIM];
  if(axis_ndim == 0){
    cl_kernel kernel = get_kernel("reduce_zero");
    int index = 0;
    clSetKernelArg(kernel, index++, sizeof(cl_mem), (void*)&x);
    clSetKernelArg(kernel, index++, sizeof(cl_mem), (void*)&y);
    clSetKernelArg(kernel, index++, sizeof(int), (void*)&xsize);
    clSetKernelArg(kernel, index++, sizeof(int), (void*)&type);
    exe_kernel(kernel);
  }else{
    get_opencl_shape(xshape, xndim, dev_xshape);
    get_opencl_shape(yshape, yndim, dev_yshape);
    get_opencl_shape(realAxis, axis_ndim, dev_axis);
    get_opencl_shape(every_xdim_size, xndim, dev_every_xdim_size);
    get_opencl_shape(flag, xndim, dev_flag);

    cl_kernel kernel = get_kernel("reduce");

    int index = 0;
    clSetKernelArg(kernel, index++, sizeof(cl_mem), (void*)&x);
    clSetKernelArg(kernel, index++, sizeof(cl_mem), (void*)&y);
    clSetKernelArg(kernel, index++, sizeof(int), (void*)&axis_ndim);
    clSetKernelArg(kernel, index++, sizeof(int), (void*)&xndim);
    clSetKernelArg(kernel, index++, sizeof(int), (void*)&yndim);
    clSetKernelArg(kernel, index++, sizeof(int), (void*)&ysize);
    clSetKernelArg(kernel, index++, sizeof(int), (void*)&axis_size);

    clSetKernelArg(kernel, index++, sizeof(int), (void*)&type);
    clSetKernelArg(kernel, index++, sizeof(int), (void*)&dev_xshape[0]);
    clSetKernelArg(kernel, index++, sizeof(int), (void*)&dev_xshape[1]);
    clSetKernelArg(kernel, index++, sizeof(int), (void*)&dev_xshape[2]);
    clSetKernelArg(kernel, index++, sizeof(int), (void*)&dev_xshape[3]);
    clSetKernelArg(kernel, index++, sizeof(int), (void*)&dev_xshape[4]);
    clSetKernelArg(kernel, index++, sizeof(int), (void*)&dev_xshape[5]);

    clSetKernelArg(kernel, index++, sizeof(int), (void*)&dev_yshape[0]);
    clSetKernelArg(kernel, index++, sizeof(int), (void*)&dev_yshape[1]);
    clSetKernelArg(kernel, index++, sizeof(int), (void*)&dev_yshape[2]);
    clSetKernelArg(kernel, index++, sizeof(int), (void*)&dev_yshape[3]);
    clSetKernelArg(kernel, index++, sizeof(int), (void*)&dev_yshape[4]);
    clSetKernelArg(kernel, index++, sizeof(int), (void*)&dev_yshape[5]);

    clSetKernelArg(kernel, index++, sizeof(int), (void*)&dev_axis[0]);
    clSetKernelArg(kernel, index++, sizeof(int), (void*)&dev_axis[1]);
    clSetKernelArg(kernel, index++, sizeof(int), (void*)&dev_axis[2]);
    clSetKernelArg(kernel, index++, sizeof(int), (void*)&dev_axis[3]);
    clSetKernelArg(kernel, index++, sizeof(int), (void*)&dev_axis[4]);
    clSetKernelArg(kernel, index++, sizeof(int), (void*)&dev_axis[5]);

    clSetKernelArg(kernel, index++, sizeof(int), (void*)&dev_every_xdim_size[0]);
    clSetKernelArg(kernel, index++, sizeof(int), (void*)&dev_every_xdim_size[1]);
    clSetKernelArg(kernel, index++, sizeof(int), (void*)&dev_every_xdim_size[2]);
    clSetKernelArg(kernel, index++, sizeof(int), (void*)&dev_every_xdim_size[3]);
    clSetKernelArg(kernel, index++, sizeof(int), (void*)&dev_every_xdim_size[4]);
    clSetKernelArg(kernel, index++, sizeof(int), (void*)&dev_every_xdim_size[5]);

    clSetKernelArg(kernel, index++, sizeof(int), (void*)&dev_flag[0]);
    clSetKernelArg(kernel, index++, sizeof(int), (void*)&dev_flag[1]);
    clSetKernelArg(kernel, index++, sizeof(int), (void*)&dev_flag[2]);
    clSetKernelArg(kernel, index++, sizeof(int), (void*)&dev_flag[3]);
    clSetKernelArg(kernel, index++, sizeof(int), (void*)&dev_flag[4]);
    clSetKernelArg(kernel, index++, sizeof(int), (void*)&dev_flag[5]);
    exe_kernel(kernel);
  }
  print_to_file(y, ysize, "reduce.txt");
}

void opencl_get_valid_count(const void *x_data, void *y_data, void *valid_count_data, const int32_t batch, const int32_t n, const int32_t k, const int32_t score_threshold){
  cl_kernel kernel = get_kernel("get_valid_count");
  int init_value = -1;
  clEnqueueFillBuffer(openclDeviceAPI->queue, (cl_mem)y_data, &init_value, sizeof(int), 0, sizeof(int)*batch*n*k, 0, NULL, NULL);
  int index = 0;
  clSetKernelArg(kernel, index++, sizeof(cl_mem), (void*)&x_data);
  clSetKernelArg(kernel, index++, sizeof(cl_mem), (void*)&valid_count_data);
  clSetKernelArg(kernel, index++, sizeof(cl_mem), (void*)&y_data);
  clSetKernelArg(kernel, index++, sizeof(int), (void*)&batch);
  clSetKernelArg(kernel, index++, sizeof(int), (void*)&n);
  clSetKernelArg(kernel, index++, sizeof(int), (void*)&k);
  clSetKernelArg(kernel, index++, sizeof(int), (void*)&score_threshold);
  exe_kernel(kernel);
  print_to_file(valid_count_data, batch, "valid_count.txt");
  print_to_file(y_data, batch*n*k, "get_valid_count.txt");
  print_to_file(x_data, batch*n*k, "get_valid_count_x.txt");
}

inline int64_t iou(const int32_t *rect1, const int32_t *rect2, const int32_t format){
    int32_t x1_min = format == FORMAT_CORNER ? rect1[0] : rect1[0] - rect1[2]/2;
    int32_t y1_min = format == FORMAT_CORNER ? rect1[1] : rect1[1] - rect1[3]/2;
    int32_t x1_max = format == FORMAT_CORNER ? rect1[2] : x1_min + rect1[2];
    int32_t y1_max = format == FORMAT_CORNER ? rect1[3] : y1_min + rect1[3];

    int32_t x2_min = format == FORMAT_CORNER ? rect2[0] : rect2[0] - rect2[2]/2;
    int32_t y2_min = format == FORMAT_CORNER ? rect2[1] : rect2[1] - rect2[3]/2;
    int32_t x2_max = format == FORMAT_CORNER ? rect2[2] : x2_min + rect2[2];
    int32_t y2_max = format == FORMAT_CORNER ? rect2[3] : y2_min + rect2[3];

    //x1,x2,y1,y2 precision <= 30
    //sum_arrea precision<=63
    int64_t sum_area = static_cast<int64_t>(x1_max-x1_min) * (y1_max-y1_min) + 
        static_cast<int64_t>(x2_max-x2_min) * (y2_max-y2_min);
    if (sum_area <= 0) return 0;

    //w,h precision <= 31
    int32_t w = std::max(0, std::min(x1_max, x2_max) - std::max(x1_min, x2_min));
    int32_t h = std::max(0, std::min(y1_max, y2_max) - std::max(y1_min, y2_min));
    //overlap_area precision <= 62
    int64_t overlap_area = static_cast<int64_t>(h)*w;
    //tmp precision <= 63
    int64_t tmp = (sum_area - overlap_area);
    if (tmp <= 0) return 0;

    int64_t max64 = ((uint64_t)1 << 63) - 1;
    if (max64 / 100 < overlap_area) { tmp /= 100; } 
    else { overlap_area *= 100; }

    return overlap_area / tmp;
}

void opencl_non_max_suppression(const void *inputs, const void *valid_count, void *outputs, 
      const int batch, const int N, const int K, 
      const bool force_suppress, const int iou_threshold, 
      const int max_output_size, const int top_k){
  int *hvalid = new int[batch];
  int *hinput = new int[batch*N*K];
  int *houtput = new int[batch*N*K];

  clEnqueueReadBuffer(openclDeviceAPI->queue, (cl_mem)valid_count, CL_TRUE, 0, sizeof(int) * batch, hvalid, 0, nullptr, nullptr); 
  clEnqueueReadBuffer(openclDeviceAPI->queue, (cl_mem)inputs, CL_TRUE, 0, sizeof(int) * batch * N*K, hinput, 0, nullptr, nullptr); 
  clEnqueueReadBuffer(openclDeviceAPI->queue, (cl_mem)outputs, CL_TRUE, 0, sizeof(int) * batch * N*K, houtput , 0, nullptr, nullptr); 
  for (int32_t b = 0; b < batch; ++b) {
    int32_t T = std::max(std::min(N, hvalid[b]), 0);
    std::vector<int32_t*> R(T); // sorted X in score descending order
    for (int i = 0; i < T; ++i) R[i] = hinput + b * N * K + i * K;

    std::stable_sort(R.begin(), R.end(), 
        [](const int32_t* a, const int32_t* b) -> bool {
        return a[1] > b[1];
        });

    int32_t n_max = T; // n_max = min{T, MOS}
  if (max_output_size >= 0)
    n_max = std::min(n_max, max_output_size);
  int32_t p_max = T; // p_max = min{TK, T}
  if (top_k >= 0) 
    p_max = std::min(p_max, top_k);

  int32_t n = 0; // dynamic calculate union U, as Y index.
  int32_t *y_batch = houtput + b * N * K; // temporary variable
  // dynamic calculate U, and n \in [0, min{n_max, card{U})
  for (int32_t p = 0; n < n_max && p < p_max; ++p) { // p \in [0, p_max)
    if (R[p][0] < 0) continue; // R[b, p, 0] >= 0

    bool ignored = false; // iou(p, q) <= iou_threshold, \forall q in U.
    for (int32_t i = 0; i < n; ++i) {
      if (force_suppress || y_batch[i*K+0] == R[p][0]) {
        int64_t iou_ret = iou(y_batch+i*K+2, R[p]+2, FORMAT_CORNER);
        if (iou_ret >= iou_threshold) {
          ignored = true;
          break;
        }
      }
    }

    if (!ignored) { // append U: copy corresponding element to Y.
      memcpy(y_batch+n*K, R[p], K*sizeof(int32_t));
      ++n;
    }
  }

  memset(y_batch+n*K, -1, (N-n)*K*sizeof(int32_t)); // others set -1.
  } 

  clEnqueueWriteBuffer(openclDeviceAPI->queue, (cl_mem)outputs, CL_TRUE, 0, sizeof(int) * batch * N*K, houtput , 0, nullptr, nullptr); 
  //cl_kernel kernel = get_kernel("non_max_suppression");
  //cl_kernel kernel_sort = get_kernel("recursion_sort");

  //std::shared_ptr<int> valid_count_data(new int[batch]);
  //clEnqueueReadBuffer(openclDeviceAPI->queue, (cl_mem)valid_count, CL_TRUE, 0, sizeof(int) * batch, valid_count_data.get(), 0, nullptr, nullptr); 
  //int32_t B = batch;

  //for (int32_t b = 0; b < B; ++b) {
  //  int32_t T = std::max(std::min(N, valid_count_data.get()[b]), 0);
  //  const int i_offset = b * N * K;
  //  const int o_offset = b * N * K;
  //  int index = 0;

  //  print_to_file(inputs, T*K, "non_max_suppression_x.txt");
  //  save_nms(inputs, T, "sort_befor.txt");

  //  clSetKernelArg(kernel_sort, index++, sizeof(cl_mem), (void*)&inputs);
  //  clSetKernelArg(kernel_sort, index++, sizeof(cl_mem), (void*)&outputs);
  //  clSetKernelArg(kernel_sort, index++, sizeof(int), (void*)&T);
  //  clSetKernelArg(kernel_sort, index++, sizeof(int), (void*)&K);
  //  int score_index = 1;
  //  clSetKernelArg(kernel_sort, index++, sizeof(int), (void*)&score_index);
  //  clSetKernelArg(kernel_sort, index++, sizeof(int), (void*)&i_offset);
  //  clSetKernelArg(kernel_sort, index++, sizeof(int), (void*)&o_offset);
  //  exe_kernel(kernel_sort);

  //  save_nms(inputs, T, "sort.txt");

  //  int init_value = -1;
  //  clEnqueueFillBuffer(openclDeviceAPI->queue, (cl_mem)outputs, &init_value, sizeof(int), b*N*K*sizeof(int), sizeof(int)*N*K, 0, NULL, NULL);

  //  int32_t n_max = T; // n_max = min{T, MOS}
  //  if (max_output_size >= 0)
  //    n_max = std::min(n_max, max_output_size);
  //  int32_t p_max = T; // p_max = min{TK, T}
  //  if (top_k >= 0) 
  //    p_max = std::min(p_max, top_k);

  //  index = 0;
  //  clSetKernelArg(kernel, index++, sizeof(cl_mem), (void*)&inputs);
  //  clSetKernelArg(kernel, index++, sizeof(cl_mem), (void*)&outputs);
  //  clSetKernelArg(kernel, index++, sizeof(int), (void*)&n_max);
  //  clSetKernelArg(kernel, index++, sizeof(int), (void*)&p_max);
  //  clSetKernelArg(kernel, index++, sizeof(bool), (void*)&force_suppress);
  //  clSetKernelArg(kernel, index++, sizeof(int), (void*)&iou_threshold);
  //  clSetKernelArg(kernel, index++, sizeof(int), (void*)&N);
  //  clSetKernelArg(kernel, index++, sizeof(int), (void*)&K);
  //  clSetKernelArg(kernel, index++, sizeof(int), (void*)&i_offset);
  //  clSetKernelArg(kernel, index++, sizeof(int), (void*)&o_offset);

  //  exe_kernel(kernel);
  //  print_to_file(outputs, N*K, "non_max_suppression.txt");
  //}
}

void opencl_repeat(const void *x_data, void *y_data, const int64_t *xshape,
    const int64_t *yshape, const uint64_t ysize, const int32_t xndim, const int32_t yndim, 
    const int32_t axis, const int32_t repeat){
  int64_t dev_xshape[MAX_DIM], dev_yshape[MAX_DIM];
  get_opencl_shape(xshape, xndim, dev_xshape);
  get_opencl_shape(yshape, yndim, dev_yshape);

  cl_kernel kernel = get_kernel("repeat");
  int index = 0;
  clSetKernelArg(kernel, index++, sizeof(cl_mem), (void*)&x_data);
  clSetKernelArg(kernel, index++, sizeof(cl_mem), (void*)&y_data);
  clSetKernelArg(kernel, index++, sizeof(int), (void*)&ysize);
  clSetKernelArg(kernel, index++, sizeof(int), (void*)&yndim);
  clSetKernelArg(kernel, index++, sizeof(int), (void*)&axis);
  clSetKernelArg(kernel, index++, sizeof(int), (void*)&repeat);
  clSetKernelArg(kernel, index++, sizeof(int), (void*)&dev_xshape[0]);
  clSetKernelArg(kernel, index++, sizeof(int), (void*)&dev_xshape[1]);
  clSetKernelArg(kernel, index++, sizeof(int), (void*)&dev_xshape[2]);
  clSetKernelArg(kernel, index++, sizeof(int), (void*)&dev_xshape[3]);
  clSetKernelArg(kernel, index++, sizeof(int), (void*)&dev_xshape[4]);
  clSetKernelArg(kernel, index++, sizeof(int), (void*)&dev_xshape[5]);

  clSetKernelArg(kernel, index++, sizeof(int), (void*)&dev_yshape[0]);
  clSetKernelArg(kernel, index++, sizeof(int), (void*)&dev_yshape[1]);
  clSetKernelArg(kernel, index++, sizeof(int), (void*)&dev_yshape[2]);
  clSetKernelArg(kernel, index++, sizeof(int), (void*)&dev_yshape[3]);
  clSetKernelArg(kernel, index++, sizeof(int), (void*)&dev_yshape[4]);
  clSetKernelArg(kernel, index++, sizeof(int), (void*)&dev_yshape[5]);

  exe_kernel(kernel);
}

void opencl_tile(const void *x_data, void *y_data, const uint64_t ysize, const int32_t yndim, const int32_t xndim,
    const int64_t *xshape, const int64_t *yshape){
  uint64_t tmp_y_size = 1;
  for(int i = 0; i < xndim; i++){
    tmp_y_size *= yshape[i + yndim - xndim];
  }
  int64_t dev_xshape[MAX_DIM], dev_yshape[MAX_DIM];
  get_opencl_shape(xshape, xndim, dev_xshape);
  get_opencl_shape(yshape, yndim, dev_yshape);

  uint64_t othery = 1;

  cl_kernel kernel = get_kernel("tile");
  int index = 0;
  clSetKernelArg(kernel, index++, sizeof(cl_mem), (void*)&x_data);
  clSetKernelArg(kernel, index++, sizeof(cl_mem), (void*)&y_data);
  clSetKernelArg(kernel, index++, sizeof(int), (void*)&tmp_y_size);
  clSetKernelArg(kernel, index++, sizeof(int), (void*)&yndim);
  clSetKernelArg(kernel, index++, sizeof(int), (void*)&xndim);

  clSetKernelArg(kernel, index++, sizeof(int), (void*)&dev_xshape[0]);
  clSetKernelArg(kernel, index++, sizeof(int), (void*)&dev_xshape[1]);
  clSetKernelArg(kernel, index++, sizeof(int), (void*)&dev_xshape[2]);
  clSetKernelArg(kernel, index++, sizeof(int), (void*)&dev_xshape[3]);
  clSetKernelArg(kernel, index++, sizeof(int), (void*)&dev_xshape[4]);
  clSetKernelArg(kernel, index++, sizeof(int), (void*)&dev_xshape[5]);

  clSetKernelArg(kernel, index++, sizeof(int), (void*)&dev_yshape[0]);
  clSetKernelArg(kernel, index++, sizeof(int), (void*)&dev_yshape[1]);
  clSetKernelArg(kernel, index++, sizeof(int), (void*)&dev_yshape[2]);
  clSetKernelArg(kernel, index++, sizeof(int), (void*)&dev_yshape[3]);
  clSetKernelArg(kernel, index++, sizeof(int), (void*)&dev_yshape[4]);
  clSetKernelArg(kernel, index++, sizeof(int), (void*)&dev_yshape[5]);
  exe_kernel(kernel);
  
  for(int i = 0; i < yndim-xndim; i++){
    othery *= yshape[i];
  }
  for(size_t i = 1; i < othery; i++){
 //   status = openclMemcpy(y_data + i*tmp_y_size, y_data, tmp_y_size * sizeof(int32_t), openclMemcpyDeviceToDevice);
    clEnqueueCopyBuffer(openclDeviceAPI->queue, (cl_mem)y_data, (cl_mem)y_data, 0, i*tmp_y_size*sizeof(int), tmp_y_size*sizeof(int), 0, NULL, NULL);
  }
}

void opencl_transpose(const void *x_data, const int64_t *axes_data, void *y_data, 
    const int64_t *xshape, const int64_t *yshape, const int32_t ndim, const uint64_t ysize,
    const int32_t axes_ndim){
  int64_t dev_xshape[MAX_DIM], dev_yshape[MAX_DIM], dev_axes[MAX_DIM];
  get_opencl_shape(xshape, ndim, dev_xshape);
  get_opencl_shape(yshape, ndim, dev_yshape);
  if(axes_ndim > 0){
    get_opencl_shape(axes_data, axes_ndim, dev_axes);
  }

  cl_kernel kernel = get_kernel("transpose");
  int index = 0;
  clSetKernelArg(kernel, index++, sizeof(cl_mem), (void*)&x_data);
  clSetKernelArg(kernel, index++, sizeof(cl_mem), (void*)&y_data);
  clSetKernelArg(kernel, index++, sizeof(int), (void*)&ndim);
  clSetKernelArg(kernel, index++, sizeof(int), (void*)&ysize);
  clSetKernelArg(kernel, index++, sizeof(int), (void*)&axes_ndim);

  clSetKernelArg(kernel, index++, sizeof(int), (void*)&dev_xshape[0]);
  clSetKernelArg(kernel, index++, sizeof(int), (void*)&dev_xshape[1]);
  clSetKernelArg(kernel, index++, sizeof(int), (void*)&dev_xshape[2]);
  clSetKernelArg(kernel, index++, sizeof(int), (void*)&dev_xshape[3]);
  clSetKernelArg(kernel, index++, sizeof(int), (void*)&dev_xshape[4]);
  clSetKernelArg(kernel, index++, sizeof(int), (void*)&dev_xshape[5]);

  clSetKernelArg(kernel, index++, sizeof(int), (void*)&dev_yshape[0]);
  clSetKernelArg(kernel, index++, sizeof(int), (void*)&dev_yshape[1]);
  clSetKernelArg(kernel, index++, sizeof(int), (void*)&dev_yshape[2]);
  clSetKernelArg(kernel, index++, sizeof(int), (void*)&dev_yshape[3]);
  clSetKernelArg(kernel, index++, sizeof(int), (void*)&dev_yshape[4]);
  clSetKernelArg(kernel, index++, sizeof(int), (void*)&dev_yshape[5]);
  
  clSetKernelArg(kernel, index++, sizeof(int), (void*)&dev_axes[0]);
  clSetKernelArg(kernel, index++, sizeof(int), (void*)&dev_axes[1]);
  clSetKernelArg(kernel, index++, sizeof(int), (void*)&dev_axes[2]);
  clSetKernelArg(kernel, index++, sizeof(int), (void*)&dev_axes[3]);
  clSetKernelArg(kernel, index++, sizeof(int), (void*)&dev_axes[4]);
  clSetKernelArg(kernel, index++, sizeof(int), (void*)&dev_axes[5]);
  exe_kernel(kernel);
}
void opencl_stride_slice(const void *x_data, void *y_data, const int64_t *begin_data,
    const int32_t begin_ndim, const int64_t *step_data, const int64_t *xshape, const int64_t *yshape, 
    const int32_t step_ndim, const int32_t y_ndim, const uint64_t ysize, const int32_t x_ndim){
  int64_t dev_xshape[MAX_DIM], dev_yshape[MAX_DIM], dev_begin[MAX_DIM], dev_step[MAX_DIM];
  get_opencl_shape(xshape, x_ndim, dev_xshape);
  get_opencl_shape(yshape, y_ndim, dev_yshape);
  get_opencl_shape(begin_data, y_ndim, dev_begin);
  get_opencl_shape(step_data, y_ndim, dev_step);

  cl_kernel kernel = get_kernel("stride_slice");
  int index = 0;
  clSetKernelArg(kernel, index++, sizeof(cl_mem), (void*)&x_data);
  clSetKernelArg(kernel, index++, sizeof(cl_mem), (void*)&y_data);
  clSetKernelArg(kernel, index++, sizeof(int), (void*)&x_ndim);
  clSetKernelArg(kernel, index++, sizeof(int), (void*)&ysize);

  clSetKernelArg(kernel, index++, sizeof(int), (void*)&dev_xshape[0]);
  clSetKernelArg(kernel, index++, sizeof(int), (void*)&dev_xshape[1]);
  clSetKernelArg(kernel, index++, sizeof(int), (void*)&dev_xshape[2]);
  clSetKernelArg(kernel, index++, sizeof(int), (void*)&dev_xshape[3]);
  clSetKernelArg(kernel, index++, sizeof(int), (void*)&dev_xshape[4]);
  clSetKernelArg(kernel, index++, sizeof(int), (void*)&dev_xshape[5]);

  clSetKernelArg(kernel, index++, sizeof(int), (void*)&dev_yshape[0]);
  clSetKernelArg(kernel, index++, sizeof(int), (void*)&dev_yshape[1]);
  clSetKernelArg(kernel, index++, sizeof(int), (void*)&dev_yshape[2]);
  clSetKernelArg(kernel, index++, sizeof(int), (void*)&dev_yshape[3]);
  clSetKernelArg(kernel, index++, sizeof(int), (void*)&dev_yshape[4]);
  clSetKernelArg(kernel, index++, sizeof(int), (void*)&dev_yshape[5]);
  
  clSetKernelArg(kernel, index++, sizeof(int), (void*)&dev_begin[0]);
  clSetKernelArg(kernel, index++, sizeof(int), (void*)&dev_begin[1]);
  clSetKernelArg(kernel, index++, sizeof(int), (void*)&dev_begin[2]);
  clSetKernelArg(kernel, index++, sizeof(int), (void*)&dev_begin[3]);
  clSetKernelArg(kernel, index++, sizeof(int), (void*)&dev_begin[4]);
  clSetKernelArg(kernel, index++, sizeof(int), (void*)&dev_begin[5]);

  clSetKernelArg(kernel, index++, sizeof(int), (void*)&dev_step[0]);
  clSetKernelArg(kernel, index++, sizeof(int), (void*)&dev_step[1]);
  clSetKernelArg(kernel, index++, sizeof(int), (void*)&dev_step[2]);
  clSetKernelArg(kernel, index++, sizeof(int), (void*)&dev_step[3]);
  clSetKernelArg(kernel, index++, sizeof(int), (void*)&dev_step[4]);
  clSetKernelArg(kernel, index++, sizeof(int), (void*)&dev_step[5]);
  exe_kernel(kernel);

  print_to_file(y_data, ysize, "stride_slice.txt");
}

void opencl_slice_like(const void *x_data, void *y_data, const int64_t *xshape, const int64_t *yshape,
    const int ysize, const int32_t ndim){
  int dev_xshape[MAX_DIM], dev_yshape[MAX_DIM];
  get_opencl_shape(xshape, ndim, dev_xshape);
  get_opencl_shape(yshape, ndim, dev_yshape);

  cl_kernel kernel = get_kernel("slice_like");
  int index = 0;
  clSetKernelArg(kernel, index++, sizeof(cl_mem), (void*)&x_data);
  clSetKernelArg(kernel, index++, sizeof(cl_mem), (void*)&y_data);
  clSetKernelArg(kernel, index++, sizeof(int), (void*)&ysize);
  clSetKernelArg(kernel, index++, sizeof(int), (void*)&ndim);

  clSetKernelArg(kernel, index++, sizeof(int), (void*)&dev_xshape[0]);
  clSetKernelArg(kernel, index++, sizeof(int), (void*)&dev_xshape[1]);
  clSetKernelArg(kernel, index++, sizeof(int), (void*)&dev_xshape[2]);
  clSetKernelArg(kernel, index++, sizeof(int), (void*)&dev_xshape[3]);
  clSetKernelArg(kernel, index++, sizeof(int), (void*)&dev_xshape[4]);
  clSetKernelArg(kernel, index++, sizeof(int), (void*)&dev_xshape[5]);

  clSetKernelArg(kernel, index++, sizeof(int), (void*)&dev_yshape[0]);
  clSetKernelArg(kernel, index++, sizeof(int), (void*)&dev_yshape[1]);
  clSetKernelArg(kernel, index++, sizeof(int), (void*)&dev_yshape[2]);
  clSetKernelArg(kernel, index++, sizeof(int), (void*)&dev_yshape[3]);
  clSetKernelArg(kernel, index++, sizeof(int), (void*)&dev_yshape[4]);
  clSetKernelArg(kernel, index++, sizeof(int), (void*)&dev_yshape[5]);
  exe_kernel(kernel);
}

void opencl_upsampling_nearest(const void *x_data, void *y_data, const uint32_t scale, const int32_t ih, const int32_t iw, 
    const uint32_t oh, const uint32_t ow, const uint32_t batch, const uint32_t channel){
  cl_kernel kernel = get_kernel("upsampling");
  for(uint32_t i = 0; i < batch; i++){

    int index = 0;
    clSetKernelArg(kernel, index++, sizeof(cl_mem), (void*)&x_data);
    clSetKernelArg(kernel, index++, sizeof(cl_mem), (void*)&y_data);
    clSetKernelArg(kernel, index++, sizeof(int), (void*)&scale);
    clSetKernelArg(kernel, index++, sizeof(int), (void*)&ih);
    clSetKernelArg(kernel, index++, sizeof(int), (void*)&iw);
    clSetKernelArg(kernel, index++, sizeof(int), (void*)&oh);
    clSetKernelArg(kernel, index++, sizeof(int), (void*)&ow);
    clSetKernelArg(kernel, index++, sizeof(int), (void*)&batch);
    clSetKernelArg(kernel, index++, sizeof(int), (void*)&channel);
    exe_kernel(kernel);
  }
}

void opencl_take(const void *x_data, const void *indices_data, void *y_data, 
    const int64_t *xshape, const int64_t *yshape, const int64_t *indices_shape, const int32_t yndim,
    const int32_t xndim, const int32_t indices_ndim, const uint64_t ysize, const int32_t axis){
  int64_t dev_xshape[MAX_DIM], dev_yshape[MAX_DIM], dev_indices_shape[MAX_DIM];
  get_opencl_shape(xshape, xndim, dev_xshape);
  get_opencl_shape(yshape, yndim, dev_yshape);
  get_opencl_shape(indices_shape, indices_ndim, dev_indices_shape);

  cl_kernel kernel = get_kernel("take");
  int index = 0;
  clSetKernelArg(kernel, index++, sizeof(cl_mem), (void*)&x_data);
  clSetKernelArg(kernel, index++, sizeof(cl_mem), (void*)&indices_data);
  clSetKernelArg(kernel, index++, sizeof(cl_mem), (void*)&y_data);
  clSetKernelArg(kernel, index++, sizeof(int), (void*)&yndim);
  clSetKernelArg(kernel, index++, sizeof(int), (void*)&xndim);
  clSetKernelArg(kernel, index++, sizeof(int), (void*)&indices_ndim);
  clSetKernelArg(kernel, index++, sizeof(int), (void*)&ysize);
  clSetKernelArg(kernel, index++, sizeof(int), (void*)&axis);

  clSetKernelArg(kernel, index++, sizeof(int), (void*)&dev_xshape[0]);
  clSetKernelArg(kernel, index++, sizeof(int), (void*)&dev_xshape[1]);
  clSetKernelArg(kernel, index++, sizeof(int), (void*)&dev_xshape[2]);
  clSetKernelArg(kernel, index++, sizeof(int), (void*)&dev_xshape[3]);
  clSetKernelArg(kernel, index++, sizeof(int), (void*)&dev_xshape[4]);
  clSetKernelArg(kernel, index++, sizeof(int), (void*)&dev_xshape[5]);

  clSetKernelArg(kernel, index++, sizeof(int), (void*)&dev_yshape[0]);
  clSetKernelArg(kernel, index++, sizeof(int), (void*)&dev_yshape[1]);
  clSetKernelArg(kernel, index++, sizeof(int), (void*)&dev_yshape[2]);
  clSetKernelArg(kernel, index++, sizeof(int), (void*)&dev_yshape[3]);
  clSetKernelArg(kernel, index++, sizeof(int), (void*)&dev_yshape[4]);
  clSetKernelArg(kernel, index++, sizeof(int), (void*)&dev_yshape[5]);

  clSetKernelArg(kernel, index++, sizeof(int), (void*)&dev_indices_shape[0]);
  clSetKernelArg(kernel, index++, sizeof(int), (void*)&dev_indices_shape[1]);
  clSetKernelArg(kernel, index++, sizeof(int), (void*)&dev_indices_shape[2]);
  clSetKernelArg(kernel, index++, sizeof(int), (void*)&dev_indices_shape[3]);
  clSetKernelArg(kernel, index++, sizeof(int), (void*)&dev_indices_shape[4]);
  clSetKernelArg(kernel, index++, sizeof(int), (void*)&dev_indices_shape[5]);
  exe_kernel(kernel);
  
}

void opencl_take(const void *x_data, const void *indices_data, void *y_data, const uint64_t ysize, const uint64_t xsize){

  cl_kernel kernel = get_kernel("take_no_axis");
  int index = 0;
  clSetKernelArg(kernel, index++, sizeof(cl_mem), (void*)&x_data);
  clSetKernelArg(kernel, index++, sizeof(cl_mem), (void*)&indices_data);
  clSetKernelArg(kernel, index++, sizeof(cl_mem), (void*)&y_data);
  clSetKernelArg(kernel, index++, sizeof(int), (void*)&ysize);
  clSetKernelArg(kernel, index++, sizeof(int), (void*)&xsize);
  exe_kernel(kernel);
}

void opencl_where(const void *x_data, const void *y_data, const void *condition_data, void *result_data, bool same_shape, const int n, const int shape0){
  if(same_shape){
    //kernel_where_same_shape<<<blockSize, threadSize>>>(x_data, y_data, condition_data, result_data, n);
    clEnqueueCopyBuffer(openclDeviceAPI->queue, (cl_mem)y_data, (cl_mem)result_data, 0, 0, n*sizeof(int), 0, NULL, NULL);
    cl_kernel kernel = get_kernel("where_same_shape");
    int index = 0;
    clSetKernelArg(kernel, index++, sizeof(cl_mem), (void*)&x_data);
    //clSetKernelArg(kernel, index++, sizeof(cl_mem), (void*)&y_data);
    clSetKernelArg(kernel, index++, sizeof(cl_mem), (void*)&condition_data);
    clSetKernelArg(kernel, index++, sizeof(cl_mem), (void*)&result_data);
    clSetKernelArg(kernel, index++, sizeof(int), (void*)&n);
    exe_kernel(kernel);
  }else{
    //kernel_where_shape0<<<blockSize, threadSize>>>(x_data, y_data, condition_data, result_data, shape0, n);
    cl_kernel kernel = get_kernel("wherw_shape0");
    int index = 0;
    clSetKernelArg(kernel, index++, sizeof(cl_mem), (void*)&x_data);
    clSetKernelArg(kernel, index++, sizeof(cl_mem), (void*)&y_data);
    clSetKernelArg(kernel, index++, sizeof(cl_mem), (void*)&condition_data);
    clSetKernelArg(kernel, index++, sizeof(cl_mem), (void*)&result_data);
    clSetKernelArg(kernel, index++, sizeof(int), (void*)&shape0);
    clSetKernelArg(kernel, index++, sizeof(int), (void*)&n);
    exe_kernel(kernel);
  }
}

void opencl_negative(const void *x_data, void *y_data, uint64_t n){
  //kernel_negative<<<blockSize, threadSize>>>(x_data, y_data, n);
  cl_kernel kernel = get_kernel("cvm_math");
  int index = 0;
  clSetKernelArg(kernel, index++, sizeof(cl_mem), (void*)&x_data);
  clSetKernelArg(kernel, index++, sizeof(cl_mem), (void*)&y_data);
  clSetKernelArg(kernel, index++, sizeof(int), (void*)&n);
  int type = 2;
  clSetKernelArg(kernel, index++, sizeof(int), (void*)&type);
  exe_kernel(kernel);
}

void opencl_log(const void *x, void *y, const uint64_t n){
  cl_kernel kernel = get_kernel("cvm_math");
  int index = 0;
  clSetKernelArg(kernel, index++, sizeof(cl_mem), (void*)&x);
  clSetKernelArg(kernel, index++, sizeof(cl_mem), (void*)&y);
  clSetKernelArg(kernel, index++, sizeof(int), (void*)&n);
  int type = 1;
  clSetKernelArg(kernel, index++, sizeof(int), (void*)&type);
  exe_kernel(kernel);
}

void opencl_abs(const void *x, void *y, const uint64_t n){
  cl_kernel kernel = get_kernel("cvm_math");
  int index = 0;
  clSetKernelArg(kernel, index++, sizeof(cl_mem), (void*)&x);
  clSetKernelArg(kernel, index++, sizeof(cl_mem), (void*)&y);
  clSetKernelArg(kernel, index++, sizeof(int), (void*)&n);
  int type = 0;
  clSetKernelArg(kernel, index++, sizeof(int), (void*)&type);
  exe_kernel(kernel);
}

#endif
  
