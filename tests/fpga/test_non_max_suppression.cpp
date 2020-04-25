#include <CL/opencl.h>
#include <iostream>
#include <string>
#include <vector>
#include <iostream>
#include <fstream>
#include <algorithm>

#include <string.h>
#include "util.hpp"
using namespace std;


#define FORMAT_CORNER 1
#define FORMAT_CENTER 2

int64_t iou(const int32_t *rect1, const int32_t *rect2, const int32_t format){
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

void nms_cpu(const int *inputs, const int *valid_count, int *outputs, 
    const int batch, const int N, const int K, 
    const bool force_suppress, const int iou_threshold,
    const int max_output_size, const int top_k){
  //auto X = CVMArg2Data<int32_t>(args[0]);
  //auto valid_count = CVMArg2Data<int32_t>(args[1]);
  //auto Y = CVMArg2Data<int32_t>(args[2]);
  //auto params = CVMArg2Attr<top::NonMaximumSuppressionParam>(args[3]);

  // X's shape must be (B, N, K), K = 6
  //auto x_shape = CVMArgShape(args[0]);
  int32_t B = batch;//x_shape[0];

  for (int32_t b = 0; b < B; ++b) {
    int32_t T = std::max(std::min(N, valid_count[b]), 0);
    std::vector<const int32_t*> R(T); // sorted X in score descending order
    for (int i = 0; i < T; ++i) R[i] = inputs + b * N * K + i * K;

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
    int32_t *y_batch = outputs + b * N * K; // temporary variable
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
}

void nms_fpga(const int *inputs, const int *valid_count, int *outputs, 
      const int batch, const int N, const int K, 
      const bool force_suppress, const int iou_threshold,
      const int max_output_size, const int top_k){

  int32_t B = batch;

  for (int32_t b = 0; b < B; ++b) {
    int32_t T = std::max(std::min(N, valid_count[b]), 0);
//    std::vector<const int32_t*> R(T); // sorted X in score descending order
//    for (int i = 0; i < T; ++i) R[i] = inputs + b * N * K + i * K;
//
//    std::stable_sort(R.begin(), R.end(), 
//        [](const int32_t* a, const int32_t* b) -> bool {
//        return a[1] > b[1];
//        });

    
    int32_t n_max = T; // n_max = min{T, MOS}
    if (max_output_size >= 0)
      n_max = std::min(n_max, max_output_size);
    int32_t p_max = T; // p_max = min{TK, T}
    if (top_k >= 0) 
      p_max = std::min(p_max, top_k);

    int32_t n = 0; // dynamic calculate union U, as Y index.
    int32_t *y_batch = outputs + b * N * K; // temporary variable

    int i_offset = b * N * K;
    int o_offset = b * N * K;
    cl_int code;
    cl_mem bufI = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(int)*N*K, NULL, &code);
    cl_mem bufO = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(int)*N*K, NULL, &code);
    assert(code == CL_SUCCESS);

    cl_kernel kernel_sort = clCreateKernel(program, "recursion_sort", &code);
    clEnqueueWriteBuffer(queue, bufI, CL_TRUE, 0, sizeof(int)*K*N, inputs + i_offset, 0, nullptr, nullptr);
    int index = 0;
    clSetKernelArg(kernel_sort, index++, sizeof(cl_mem), (void*)&bufI);
    clSetKernelArg(kernel_sort, index++, sizeof(cl_mem), (void*)&bufO);
    clSetKernelArg(kernel_sort, index++, sizeof(int), (void*)&N);
    clSetKernelArg(kernel_sort, index++, sizeof(int), (void*)&K);
    int score_index = 1;
    clSetKernelArg(kernel_sort, index++, sizeof(int), (void*)&score_index);
    clEnqueueTask(queue, kernel_sort, 0, NULL, NULL);
    //for(int i = 0; i < R.size(); i++){
    //  clEnqueueWriteBuffer(queue, bufI, CL_TRUE, sizeof(int)*(i_offset + i * K), sizeof(int)*K, R[i], 0, nullptr, nullptr);
    //}

    index = 0;
    cl_kernel kernel = clCreateKernel(program, "non_max_suppression", &code);
    assert(code == CL_SUCCESS);
    clSetKernelArg(kernel, index++, sizeof(cl_mem), (void*)&bufI);
    clSetKernelArg(kernel, index++, sizeof(cl_mem), (void*)&bufO);
    clSetKernelArg(kernel, index++, sizeof(int), (void*)&n_max);
    clSetKernelArg(kernel, index++, sizeof(int), (void*)&p_max);
    clSetKernelArg(kernel, index++, sizeof(bool), (void*)&force_suppress);
    clSetKernelArg(kernel, index++, sizeof(int), (void*)&iou_threshold);
    clSetKernelArg(kernel, index++, sizeof(int), (void*)&N);
    clSetKernelArg(kernel, index++, sizeof(int), (void*)&K);
    clSetKernelArg(kernel, index++, sizeof(int), (void*)&i_offset);
    clSetKernelArg(kernel, index++, sizeof(int), (void*)&o_offset);

    clEnqueueTask(queue, kernel, 0, NULL, NULL); 
    clEnqueueReadBuffer(queue, bufO, CL_TRUE, sizeof(int) * o_offset, sizeof(int) *N*K, outputs + o_offset, 0, nullptr, nullptr); 
  }
}

int main(){
  init_opencl("ops.xclbin");
  int batchs = 1;
  int n = 4;
  int k = 6;
  bool force_suppress = false;
  int iou_threshold = 20;
  int top_k = -1;
  int max_output_size = -1;
  int *inputs = new int[batchs * n * k];
  int *valid_count = new int[batchs];
  int *outputs = new int[batchs * n * k];
  int *outputs2 = new int[batchs * n * k];

  for(int i = 0; i < batchs * n * k; i++){
    inputs[i] = i % 127;
  }

  nms_cpu(inputs, valid_count, outputs, batchs, n, k, force_suppress, iou_threshold, max_output_size, top_k);
  nms_fpga(inputs, valid_count, outputs2, batchs, n, k, force_suppress, iou_threshold, max_output_size, top_k);

  verify(outputs, outputs2, batchs * n * k);
  return 0;
}
