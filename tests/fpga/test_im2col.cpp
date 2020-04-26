#include <CL/opencl.h>
#include <iostream>
#include <string>
#include <vector>
#include <iostream>
#include <fstream>
#include "util.hpp"

using namespace std;

inline bool is_a_ge_zero_and_a_lt_b(int a, int b) {
  return static_cast<unsigned>(a) < static_cast<unsigned>(b);
}
void im2col_cpu(const int32_t* data_im, const int channels,
    const int height, const int width, const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w,
    const int stride_h, const int stride_w,
    const int dilation_h, const int dilation_w,
    char* data_col, bool &has_negetive)
{
  // auto data_col_init = data_col;
  const int output_h = (height + 2 * pad_h -
      (dilation_h * (kernel_h - 1) + 1)) / stride_h + 1;
  const int output_w = (width + 2 * pad_w -
      (dilation_w * (kernel_w - 1) + 1)) / stride_w + 1;
  const int channel_size = height * width;
  for (int channel = channels; channel--; data_im += channel_size) {
    for (int kernel_row = 0; kernel_row < kernel_h; kernel_row++) {
      for (int kernel_col = 0; kernel_col < kernel_w; kernel_col++) {
        int input_row = -pad_h + kernel_row * dilation_h;
        for (int output_rows = output_h; output_rows; output_rows--) {
          if (!is_a_ge_zero_and_a_lt_b(input_row, height)) {
            for (int output_cols = output_w; output_cols; output_cols--) {
              *(data_col++) = 0;
            }
          } else {
            int input_col = -pad_w + kernel_col * dilation_w;
            for (int output_col = output_w; output_col; output_col--) {
              if (is_a_ge_zero_and_a_lt_b(input_col, width)) {
                int32_t tv = data_im[input_row * width + input_col];
                if(tv < 0) {
                  has_negetive = true;
                }
                *(data_col++) = tv;
              } else {
                *(data_col++) = 0;
              }
              input_col += stride_w;
            }
          }
          input_row += stride_h;
        }
      }
    }
  }
}

void im2col_fpga(const int *input, char *output, 
    const int n, const int c, const int h, const int w,
    const int kh, const int kw, 
    const int pad_h, const int pad_w,
    const int stride_h, const int stride_w,
    const int dilation_h, const int dilation_w,
    const int oh, const int ow){
  cl_int code;
  const int K = c * kh * kw;
  const int N = oh * ow;
  const int TK = (K+63)/64*64;
  const int TN = (N+63)/64*64;
  cl_mem bufi = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(int)*n*c*h*w, NULL, &code);
  clEnqueueWriteBuffer(queue, bufi, CL_TRUE, 0, sizeof(int)*n*c*h*w, input, 0, nullptr, nullptr);
  cl_mem bufo = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(char)*TK*TN, NULL, &code);

  cl_kernel im2col = clCreateKernel(program, "im2col", &code);
  int index = 0;
  int num_kernels = c *oh *ow;
  clSetKernelArg(im2col, index++, sizeof(cl_mem), (void*)&bufi);
  clSetKernelArg(im2col, index++, sizeof(cl_mem), (void*)&bufo);
  clSetKernelArg(im2col, index++, sizeof(int), (void*)&num_kernels);
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
  int offset = 0;
  //clSetKernelArg(im2col, index++, sizeof(int), (void*)&offset);
  clEnqueueTask(queue, im2col, 0, NULL, NULL);
  clEnqueueReadBuffer(queue, bufo, CL_TRUE, 0, sizeof(char)*TK*TN, output, 0, nullptr, nullptr); 
}
int main(){

  init_opencl("ops.xclbin");
  const int n = 1;
  const int c = 1;
  const int h = 8;
  const int w = 8;
  const int oc = 16;
  const int kh = 3;
  const int kw = 3;
  const int oh = h - kh + 1;
  const int ow = w - kw + 1;
  const int pad_h = 0;
  const int pad_w = 0;
  const int stride_h = 1;
  const int stride_w = 1;
  const int dilation_h = 1;
  const int dilation_w = 1;
  const int K = c * kh * kw;
  const int N = oh * ow;
  const int TK = (K+63)/64*64;
  const int TN = (N+63)/64*64;
  int num_kernels = c *oh *ow;
  int input[n*c*h*w];
  char output[TK*TN], output2[TK*TN];
  for(int i = 0; i < n*c*h*w; i++){
    input[i] = i % 127;
  }
  bool has_negetive;
  im2col_cpu(input, c, h, w, kh, kw, 0, 0, 1, 1, 1, 1, output, has_negetive);
  im2col_fpga(input, output2, n, c, h, w, kh, kw, pad_h, pad_w, stride_h, stride_w, dilation_h, dilation_w, oh, ow);

//  verify(output, output2, TK*TN);
  for(int i = 0; i < K; i++){
    for(int j = 0; j < N; j++){
      if(output[i*N+j] != output2[i*TN+j]){
        cout << "failed: " << i*N+j << ": " << (int)output[i*N+j] << "," << (int)output2[i*TN+j]<< endl;
        return 0;
      }
    }
  }
  cout << "success\n";

  return 0;
}
