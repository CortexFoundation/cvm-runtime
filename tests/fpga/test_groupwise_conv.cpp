#include <CL/opencl.h>
#include <iostream>
#include <string>
#include <vector>
#include <iostream>
#include <fstream>

#include <string.h>
#include "util.hpp"
using namespace std;

void groupwise_conv2d(
   int32_t *x_data, int32_t n_batch, int32_t in_channels, int32_t x_h, int32_t x_w,
   int32_t *w_data, int32_t filter_c, int32_t filter_h, int32_t filter_w,
   int32_t *y_data, int32_t out_channels, int32_t o_h, int32_t o_w,
   int32_t *b_data,
   int32_t pad_h, int32_t pad_w, int32_t stride_h, int32_t stride_w, int32_t dilation_h, int32_t dilation_w,
   int32_t groups){
  int32_t ochannels_per_group = out_channels / groups;
  int32_t ichannels_per_group = in_channels / groups;
  for(int32_t n = 0; n < n_batch; ++n){
    for(int32_t oc = 0; oc < out_channels; ++oc){
      for(int32_t oh = 0; oh < o_h; ++oh){
        for(int32_t ow = 0; ow < o_w; ++ow){
          int32_t oi = n * out_channels * o_h * o_w + oc * o_h * o_w + oh * o_w + ow;
          int32_t sum = 0;
          int32_t ic = oc / ochannels_per_group * ichannels_per_group;
          for(int32_t tic = 0; tic < ichannels_per_group; ++tic){
            for(int32_t fh = 0; fh < filter_h; ++fh){
              for(int32_t fw = 0; fw < filter_w; ++fw){
                int32_t th = oh * stride_h + fh*dilation_h - pad_h;
                int32_t tw = ow * stride_w + fw*dilation_w - pad_w;
                if(th < 0 || tw < 0 || th >= x_h || tw >= x_w)
                  continue;
                sum += x_data[n * in_channels * x_h * x_w + (ic+tic) * x_h * x_w + th * x_w + tw]
                  * w_data[oc * filter_c * filter_h * filter_w + tic * filter_h * filter_w + fh * filter_w + fw];
              }
            }
          }
          y_data[oi] = sum + (b_data == nullptr ? 0 : b_data[oc]);
        }
      }
    }
  }
}

void groupwise_conv2d_fpga(
   int32_t *x_data, int32_t n_batch, int32_t in_channels, int32_t x_h, int32_t x_w,
   int32_t *w_data, int32_t filter_c, int32_t filter_h, int32_t filter_w,
   int32_t *y_data, int32_t out_channels, int32_t o_h, int32_t o_w,
   int32_t *b_data,
   int32_t padding[2], int32_t stride_h, int32_t stride_w, int32_t dilation_h, int32_t dilation_w,
   int32_t groups){
  
}

int main(){
//  init_opencl("ops.xclbin");
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
  const int groups = 1;

  const int nx = n*c*h*w;
  const int nw = oc*kh*kw*c;
  const int ny = n*oc*oh*ow;
  int *x_data = new int[nx];
  int *w_data = new int[nw];
  int *bias = new int[oc];
  int *y_data = new int[ny];
  for(int i = 0;i < nx; i++){
    x_data[i] = i % 127;
  }
  for(int i = 0; i < nw; i++){
    w_data[i] = i % 127;
  }
  for(int i = 0; i < oc; i++){
    bias[i] = i % 127;
  }

  groupwise_conv2d(x_data, n, c, h, w, w_data, c, kh, kw, y_data, oc, oh, ow, bias, pad_h, pad_w, stride_h, stride_w, dilation_h, dilation_w, groups);
  return 0;
}
