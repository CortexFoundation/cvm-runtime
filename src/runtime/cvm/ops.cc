#include "graph_runtime.h"

#include <tvm/runtime/ndarray.h>
#include <tvm/runtime/packed_func.h>
#include <tvm/runtime/registry.h>
#include <tvm/runtime/serializer.h>

#include <algorithm>
#include <functional>
#include <numeric>
#include <vector>
#include <string>
#include <memory>
#include <utility>

#include "cuda_ops.h"

namespace tvm {
namespace runtime {

#define DEBUG_OP false
inline void parseToIntPair(std::string str, int* ret){
	char a,b;
    sscanf(str.c_str(), "%c%d,%d%c", &a,ret, ret + 1, &b);
}

inline uint32_t getSize(DLTensor *dlTensor){
    uint32_t size = 1;
    for(int i = 0; i < dlTensor->ndim; i++){
        size *= dlTensor->shape[i];
    }
    return size;
}
/**
* x
* y
* a_min -127
* a_max 127
*/
TVM_REGISTER_GLOBAL("tvm.runtime.cvm.clip").set_body([](TVMArgs args, TVMRetValue* rv) {
   CHECK(args.num_args == 4);
   DLTensor *x = args[0];
   DLTensor *y = args[1];
   std::string min_str = args[2];
   std::string max_str = args[3];
   int min = std::atoi(min_str.c_str());
   int max = std::atoi(max_str.c_str());
   for (uint32_t i = 0; i < getSize(x); i++) {
 		static_cast<int32_t*>(y->data)[i] = std::max(std::min(max, static_cast<int32_t*>(x->data)[i]), min);
   }
 });

 TVM_REGISTER_GLOBAL("tvm.runtime.cvm.relu").set_body([](TVMArgs args, TVMRetValue* rv) {
   CHECK(args.num_args == 2);
   DLTensor *x = args[0];
   DLTensor *y = args[1];
   for (uint32_t i = 0; i < getSize(x); i++) {
 		static_cast<int32_t*>(y->data)[i] = std::max(static_cast<int32_t*>(x->data)[i], 0);
   }
 });

/*
* x
* w
* b
* y
* units 1000
* use_bias True
*/
TVM_REGISTER_GLOBAL("tvm.runtime.cvm.dense").set_body([](TVMArgs args, TVMRetValue* rv) {
  int ndim = args.num_args;
  CHECK(ndim == 6 || ndim == 5);
  DLTensor *x = args[0];
  DLTensor *w = args[1];
  DLTensor *b = nullptr;
  DLTensor *y = nullptr;
  int32_t* db = nullptr;
  if(ndim == 6){
	b = args[2];
    CHECK(b->ndim == 1) << "dense requires 1-D bias";
	y = args[3];
    db = static_cast<int32_t*>(b->data);
  }else{
	y = args[2];
  }
  CHECK(x->ndim == 2) << "dense requires 2-D data";
  CHECK(w->ndim == 2) << "dense reuqires 2-D weight";

  auto dx = static_cast<int32_t*>(x->data);
  auto dy = static_cast<int32_t*>(y->data);
  auto dw = static_cast<int32_t*>(w->data);
  // assert(y->shape[0] == 1); // not tested yet
  for (uint32_t di = 0; di < y->shape[0]; di++) {
      for (uint32_t oi = 0; oi < y->shape[1]; oi++) {
          int32_t sum = 0;
          for (uint32_t xi = 0; xi < x->shape[1]; xi++) {
              sum += dx[di * y->shape[1] + xi] * dw[oi * w->shape[1] + xi];
          }
		  if(db != nullptr){
			  sum += db[oi];
		  }
          dy[di * y->shape[1] + oi] = sum;
      }
  }

});

TVM_REGISTER_GLOBAL("tvm.runtime.cvm.flatten").set_body([]
(TVMArgs args, TVMRetValue* rv){
     CHECK(args.num_args == 2);
     DLTensor *x = args[0];
     DLTensor *y = args[1];
     for (uint32_t i = 0; i < getSize(x); i++) {
         static_cast<int32_t*>(y->data)[i] = static_cast<int32_t*>(x->data)[i];
     }

});
inline void depthwise_conv2d(
        int32_t *x_data, int32_t n_batch, int32_t in_channels, int32_t x_h, int32_t x_w,
        int32_t *w_data, int32_t filter_c, int32_t filter_h, int32_t filter_w,
        int32_t *y_data, int32_t out_channels, int32_t o_h, int32_t o_w,
        int32_t *b_data,
        int32_t padding[2], int32_t stride_h, int32_t stride_w, int32_t dilation_h, int32_t dilation_w,
        int32_t groups){
    for(int n = 0; n < n_batch; ++n){
        for(int c = 0; c < in_channels; ++c){
            for(int h = 0; h < o_h; ++h){
                for(int w = 0; w < o_w; ++w){
                    int32_t sum = 0;
                    for(int fh = 0; fh < filter_h; ++fh){
                        for(int fw = 0; fw < filter_w; ++fw){
                            int th = h * stride_h + fh*dilation_h - padding[0];
                            int tw = w * stride_h + fw*dilation_w - padding[1];
                            if(th < 0 || tw < 0 || th >= x_h || tw >= x_w)
                                continue;
                            sum += x_data[n * in_channels * x_h * x_w + c * x_h * x_w + th * x_w + tw]
                                * w_data[c * filter_h * filter_w + fh * filter_w + fw];
                        }
                    }
                    y_data[n * in_channels * o_h * o_w + c * o_h * o_w + h * o_w + w] = sum + (b_data != nullptr ? b_data[c] : 0);
                }
            }
        }
    }
}
/*
input
weight
bias
output
groups 1
dilation (1, 1)
channels 512
layout NCHW
kernel_layout OIHW
kernel_size [1, 1]
padding (0, 0)
use_bias True
strides (1, 1)
*/
TVM_REGISTER_GLOBAL("tvm.runtime.cvm.conv2d").set_body([]
 (TVMArgs args, TVMRetValue* rv){
    CHECK(args.num_args == 12 || args.num_args == 13);
    DLTensor *x = args[0];
    CHECK(x->ndim == 4);
    DLTensor *w = args[1];
    CHECK(w->ndim == 4);
    int dlIndex = 2;
	DLTensor *b = nullptr; //args[2];
    if(args.num_args == 13){
        b = args[dlIndex++];
    }
    DLTensor *y = args[dlIndex++];
    //auto time_start = clock();
	std::string groups_str = args[dlIndex++];
	std::string dilation_str = args[dlIndex++];
	std::string channels_str = args[dlIndex++];
	std::string layout_str = args[dlIndex++];
	std::string kernel_layout_str = args[dlIndex++];
	std::string kernel_size_str = args[dlIndex++];
	std::string padding_str = args[dlIndex++];
	std::string use_bias_str = args[dlIndex++];
	std::string strides_str = args[dlIndex++];
	int groups = std::atoi(groups_str.c_str());
	int dilation[2] = {0};
	parseToIntPair(dilation_str, dilation);
	//int channels = std::atoi(channels_str.c_str());
	int kernel_size[2] = {0};
	parseToIntPair(kernel_size_str, kernel_size);
	int padding[2] = {0};
	parseToIntPair(padding_str, padding);
	int strides[2] = {0};
	parseToIntPair(strides_str, strides);

    int stride_h = strides[0];
    int stride_w = strides[1];
    int dilation_h = dilation[0];
    int dilation_w = dilation[1];

    int32_t* x_data = (int32_t*)x->data;
    int32_t* w_data = (int32_t*)w->data;
    int32_t* y_data = (int32_t*)y->data;
	int32_t* b_data = b != nullptr ? (int32_t*)b->data : nullptr;

    int out_channels = static_cast<int>(w->shape[0]);
    int filter_c = static_cast<int>(w->shape[1]);
    int filter_h = static_cast<int>(w->shape[2]);
    int filter_w = static_cast<int>(w->shape[3]);
	filter_h = (filter_h - 1) * dilation[0] + 1;
	filter_w = (filter_w - 1) * dilation[1] + 1;

    int n_batch = static_cast<int>(x->shape[0]);
    int in_channels = static_cast<int>(x->shape[1]);
    int x_h = static_cast<int>(x->shape[2]);
    int x_w = static_cast<int>(x->shape[3]);
	int o_h = (x_h + 2 * padding[0] - filter_h) / strides[0] + 1;
	int o_w = (x_w + 2 * padding[1] - filter_w) / strides[1] + 1;
    if(n_batch < 1 || in_channels < 1 || x_h < 1 || x_w < 1 || filter_c < 1 || filter_h < 1 || filter_w < 1 ||
            padding[0] < 0 || padding[1] < 0 || stride_h < 1 || stride_w < 1 || dilation_h < 1 || dilation_w < 1 ||
             out_channels < 1 || o_h < 1 || o_w < 1){
        CHECK(false) << "error args";
    }
//	int o_h = static_cast<int>(y->shape[2]);
//	int o_w = static_cast<int>(y->shape[3]);
//	std::cout << o_h << " " << o_w << " "
//              << (x_h + 2 * padding[0] - filter_h) / strides[0] + 1 << " "
//              << (x_w + 2 * padding[1] - filter_w) / strides[1] + 1 << "\n";
//    std::cout << "dim = " << b->ndim << " shape = " << b->shape[0] << "\n";
//    std::cout << "padding = " << padding[0] << " " << padding[1] << "\n";


    if(groups > 1){
        depthwise_conv2d(
                x_data, n_batch, in_channels, x_h, x_w,
                w_data, filter_c, filter_h, filter_w,
                y_data, out_channels, o_h, o_w,
                b_data,
                padding, stride_h, stride_w, dilation[0], dilation[1],
                groups);
    }else{
        const int y_n_offset = out_channels * o_h * o_w;
        const int y_c_offset = o_h * o_w;
        const int y_h_offset = o_w;
        //const int x_n_offset = in_channels * x_h * x_w;
        const int x_c_offset = x_h * x_w;
        const int x_h_offset = x_w;
        const int w_o_offset = in_channels * filter_h * filter_w;
        const int w_i_offset = filter_h * filter_w;
        const int w_h_offset = filter_w;
#define CONV2d_X(n, c, h, w) x_data[(n) * y_n_offset + (c) * x_c_offset + (h) * x_h_offset + (w)]
#define CONV2d_W(o, i, h, w) w_data[(o) * w_o_offset + (i) * w_i_offset + (h) * w_h_offset + (w)]
#define CONV2d_Y(n, c, h, w) y_data[(n) * y_n_offset + (c) * y_c_offset + (h) * y_h_offset + (w)]
        auto calc_func = [&](int n, int k, int p, int q) {
            int y_sum = 0;
            for (int c = 0; c < in_channels; ++c) {
                for (int r = 0; r < filter_h; ++r) {
                    auto tp = p * stride_h + r*dilation_h - padding[0];
                    if (tp < 0 || tp >= x_h)
                        continue;
                    auto tq_start = q * stride_w - padding[1];
                    auto tq_end = q * stride_w - padding[1] + filter_w;
                    for (auto tq = std::max(tq_start, 0); tq < std::min(tq_end, x_h); ++tq) {
                        auto s = tq - tq_start;
                        y_sum += CONV2d_X(n, c, tp, tq-s+s*dilation_w) * CONV2d_W(k, c, r, s);
                    }
                }
            }
            return y_sum;

        };
        auto calc_func1x1 = [&](int n, int k, int p, int q, int r = 0, int s = 0) {
            int y_sum = 0;
            for (int c = 0; c < in_channels; ++c) {
                y_sum += CONV2d_X(n, c, p, q) * CONV2d_W(k, c, r, s);
            }
            return y_sum;
        };
        if (filter_w == 1) {
            for (int n = 0; n < n_batch; ++n) {
                for (int k = 0; k < out_channels; ++k) {
                    for (int p = 0; p < o_h; ++p) {
                        auto tp = p * stride_h - padding[0];
                        if (tp < 0 || tp >= x_h)
                            continue;
                        for (int q = 0; q < o_w; ++q) {
                            auto tq = q * stride_w - padding[1];
                            if (tq < 0 || tq >= x_w)
                                continue;
                            CONV2d_Y(n, k, p, q) = (b_data != nullptr ? b_data[k] : 0) + calc_func1x1(n, k, tp, tq);
                        }
                    }
                }
            }
        } else if (filter_w == 3) {
            std::vector<int32_t> y_sum(in_channels * o_h * o_w, 0);
//            std::cout << "buff " << y_sum.size() << "\n";
            for (int n = 0; n < n_batch; ++n) {
                for (int k = 0; k < out_channels; ++k) {
                    std::fill(y_sum.begin(), y_sum.end(), 0);
                    for (int c = 0; c < in_channels; ++c) {
                        auto conv2d_w_kc = w_data + (k) * w_o_offset + (c) * w_i_offset;
                        for (int p = 0; p < o_h; ++p) {
                            for (int q = 0; q < o_w; ++q) {
                                const int y_idx = c * o_w * o_h + p * o_w + q;
                                auto tq_start = q * stride_w - padding[1];
                                auto tq_begin = std::max(tq_start, 0);
                                auto tq_end = std::min(q * stride_w - padding[1] + filter_w, x_w);
                                {
                                    int r = 0;
                                    auto tp = p * stride_h + r - padding[0];
                                    if (tp >= 0) {
                                        if (tp >= x_h)
                                            continue;
                                        for (auto tq = tq_begin; tq < tq_end; ++tq) {
                                            auto s = tq - tq_start;
                                            y_sum[y_idx] += CONV2d_X(n, c, tp, tq-s+s*dilation_w) * conv2d_w_kc[r * filter_h + s];
                                        }
                                    }
                                }
                                {
                                    int r = 1*dilation_h;
                                    auto tp = p * stride_h + r - padding[0];
                                    if (tp >= 0) {
                                        if (tp >= x_h)
                                            continue;
                                        for (auto tq = tq_begin; tq < tq_end; ++tq) {
                                            auto s = tq - tq_start;
                                            y_sum[y_idx] += CONV2d_X(n, c, tp, tq-s+s*dilation_w) * conv2d_w_kc[r * filter_h + s];
                                        }
                                    }
                                }
                                {
                                    int r = 2*dilation_h;
                                    auto tp = p * stride_h + r - padding[0];
                                    if (tp >= 0) {
                                        if (tp >= x_h)
                                            continue;
                                        for (auto tq = tq_begin; tq < tq_end; ++tq) {
                                            auto s = tq - tq_start;
                                            y_sum[y_idx] += CONV2d_X(n, c, tp, tq-s+s*dilation_w) * conv2d_w_kc[r * filter_h + s];
                                        }
                                    }
                                }
                            }
                        }
                    }
                    for (int p = 0; p < o_h; ++p) {
                        for (int q = 0; q < o_w; ++q) {
                            uint32_t tmp = 0;
                            for (int c = 0; c < in_channels; ++c) {
                                tmp += y_sum[c * o_h * o_w + p * o_h + q];
                            }
                            CONV2d_Y(n, k, p, q) = b_data[k] + tmp;
                        }
                    }
                }
            }
        } else {
            for (int n = 0; n < n_batch; ++n) {
                for (int k = 0; k < out_channels; ++k) {
                    for (int p = 0; p < o_h; ++p) {
                        for (int q = 0; q < o_w; ++q) {
                            CONV2d_Y(n, k, p, q) = (b_data != nullptr ? b_data[k] : 0) + calc_func(n, k, p, q);
                        }
                    }
                }
            }
        }
    }

//    std::cout << o_h << " " << o_w << " (" << filter_h << "," << " " << filter_w << ")"
//              << in_channels << " " << out_channels << " "
//              << (clock() - time_start + .0) / CLOCKS_PER_SEC << "\n";
 });

inline int32_t broadcast_o_index(int64_t* oshape, int odim, int& o_index){
    if(o_index == -1){
        o_index = 0;
        return o_index;
    }
    int tmp_o_index = o_index;
    for(int i = 0; i < odim; i++){
        int idx = odim - 1 - i;
        int ovar = tmp_o_index % oshape[idx];
        if(ovar + 1 != oshape[idx]){
            o_index += 1;
            break;
        }
        tmp_o_index /= oshape[idx];
    }
    return o_index;
}
inline int32_t broadcast_i_index(int64_t* oshape, int o_index, int64_t* ishape, int idim){
    int index = 0;
    int allIndex = 0;
    for(int i = 0; i < idim; i++){
        int idx = idim - 1 - i;
        int ovar = o_index % oshape[idx];
        if(ovar < ishape[idx]){
            index += i == 0 ? ovar : allIndex * ovar;
        }else if(ishape[idx] == 1){
        }else{
        }
        allIndex = (i == 0 ? ishape[idim-1] : allIndex * ishape[idx]);
        o_index /= oshape[idx];
    }
    return index;
}

TVM_REGISTER_GLOBAL("tvm.runtime.cvm.broadcast_add")
    .set_body([](TVMArgs args, TVMRetValue *ret){
        CHECK(args.num_args == 3);
        DLTensor *args0 = args[0];
        DLTensor *args1 = args[1];
        DLTensor *args2 = args[2];
        int32_t *a = static_cast<int32_t*>(args0->data);
        int32_t *b = static_cast<int32_t*>(args1->data);
        int32_t *c = static_cast<int32_t*>(args2->data);

        int o_index = -1;
        for(uint32_t i = 0; i < getSize(args0); ++i){
            o_index = i;//broadcast_o_index(args2->shape, args2->ndim, o_index);
            int32_t a_index = broadcast_i_index(args2->shape, o_index, args0->shape, args0->ndim);
            int32_t b_index = broadcast_i_index(args2->shape, o_index, args1->shape, args1->ndim);
            c[i] = a[a_index] + b[b_index];
        }
    });

TVM_REGISTER_GLOBAL("tvm.runtime.cvm.broadcast_sub")
    .set_body([](TVMArgs args, TVMRetValue *ret){
        CHECK(args.num_args == 3);
        DLTensor *args0 = args[0];
        DLTensor *args1 = args[1];
        DLTensor *args2 = args[2];
        int32_t *a = static_cast<int32_t*>(args0->data);
        int32_t *b = static_cast<int32_t*>(args1->data);
        int32_t *c = static_cast<int32_t*>(args2->data);

        int o_index = -1;
        for(uint32_t i = 0; i < getSize(args0); ++i){
            o_index = broadcast_o_index(args2->shape, args2->ndim, o_index);
            int32_t a_index = broadcast_i_index(args2->shape, o_index, args0->shape, args0->ndim);
            int32_t b_index = broadcast_i_index(args2->shape, o_index, args1->shape, args1->ndim);
            c[i] = a[a_index] - b[b_index];
        }
    });

TVM_REGISTER_GLOBAL("tvm.runtime.cvm.broadcast_mul")
    .set_body([](TVMArgs args, TVMRetValue *ret){
        CHECK(args.num_args == 3);
        DLTensor *args0 = args[0];
        DLTensor *args1 = args[1];
        DLTensor *args2 = args[2];
        int32_t *a = static_cast<int32_t*>(args0->data);
        int32_t *b = static_cast<int32_t*>(args1->data);
        int32_t *c = static_cast<int32_t*>(args2->data);
        int o_index = -1;
        for(uint32_t i = 0; i < getSize(args0); ++i){
            o_index = broadcast_o_index(args2->shape, args2->ndim, o_index);
            int32_t a_index = broadcast_i_index(args2->shape, o_index, args0->shape, args0->ndim);
            int32_t b_index = broadcast_i_index(args2->shape, o_index, args1->shape, args1->ndim);
            c[i] = a[a_index] * b[b_index];
        }

    });
TVM_REGISTER_GLOBAL("tvm.runtime.cvm.broadcast_div")
    .set_body([](TVMArgs args, TVMRetValue *ret){
        CHECK(args.num_args == 3);
        DLTensor *args0 = args[0];
        DLTensor *args1 = args[1];
        DLTensor *args2 = args[2];
        int32_t *a = static_cast<int32_t*>(args0->data);
        int32_t *b = static_cast<int32_t*>(args1->data);
        int32_t *c = static_cast<int32_t*>(args2->data);

        int o_index = -1;
        for(uint32_t i = 0; i < getSize(args0); ++i){
            o_index = broadcast_o_index(args2->shape, args2->ndim, o_index);
            int32_t a_index = broadcast_i_index(args2->shape, o_index, args0->shape, args0->ndim);
            int32_t b_index = broadcast_i_index(args2->shape, o_index, args1->shape, args1->ndim);
            CHECK(b[b_index] != 0);
            c[i] = a[a_index] / b[b_index];
        }
    });
TVM_REGISTER_GLOBAL("tvm.runtime.cvm.broadcast_right_shift")
    .set_body([](TVMArgs args, TVMRetValue *ret){
        CHECK(args.num_args == 3);
        DLTensor *args0 = args[0];
        DLTensor *args1 = args[1];
        DLTensor *args2 = args[2];
        int32_t *a = static_cast<int32_t*>(args0->data);
        int32_t *b = static_cast<int32_t*>(args1->data);
        int32_t *c = static_cast<int32_t*>(args2->data);

        int o_index = -1;
        for(uint32_t i = 0; i < getSize(args0); ++i){
            o_index = broadcast_o_index(args2->shape, args2->ndim, o_index);
            int32_t a_index = broadcast_i_index(args2->shape, o_index, args0->shape, args0->ndim);
            int32_t b_index = broadcast_i_index(args2->shape, o_index, args1->shape, args1->ndim);
            c[i] = a[a_index] >> b[b_index];
        }
    });
TVM_REGISTER_GLOBAL("tvm.runtime.cvm.broadcast_left_shift")
    .set_body([](TVMArgs args, TVMRetValue *ret){
        CHECK(args.num_args == 3);
        DLTensor *args0 = args[0];
        DLTensor *args1 = args[1];
        DLTensor *args2 = args[2];
        int32_t *a = static_cast<int32_t*>(args0->data);
        int32_t *b = static_cast<int32_t*>(args1->data);
        int32_t *c = static_cast<int32_t*>(args2->data);

        int o_index = -1;
        for(uint32_t i = 0; i < getSize(args0); ++i){
            o_index = broadcast_o_index(args2->shape, args2->ndim, o_index);
            int32_t a_index = broadcast_i_index(args2->shape, o_index, args0->shape, args0->ndim);
            int32_t b_index = broadcast_i_index(args2->shape, o_index, args1->shape, args1->ndim);
            c[i] = a[a_index] << b[b_index];
        }
    });

/*
* strides (2, 2)
* pool_size [3, 3]
* ceil_mode False
* padding (1, 1)
*/
TVM_REGISTER_GLOBAL("tvm.runtime.cvm.max_pool2d")
    .set_body([](TVMArgs args, TVMRetValue *ret){
    CHECK(args.num_args == 6);
	DLTensor *x = args[0];
	DLTensor *y = args[1];
	std::string strides_str = args[2];
	std::string pool_size_str = args[3];
	std::string ceil_mode = args[4];
	std::string padding_str = args[5];
	int strides[2] = {0};
	parseToIntPair(strides_str, strides);
	int pool_size[2] = {0};
	parseToIntPair(pool_size_str, pool_size);
	int padding[2] = {0};
	parseToIntPair(padding_str, padding);

    int stride_h = strides[0];
    int stride_w = strides[1];

    int32_t* x_data = (int32_t*)x->data;
    int32_t* y_data = (int32_t*)y->data;

    int filter_h = pool_size[0];
    int filter_w = pool_size[1];

    int n_batch = static_cast<int>(x->shape[0]);
    int in_channels = static_cast<int>(x->shape[1]);
	int out_channels = in_channels;
    int x_h = static_cast<int>(x->shape[2]);
    int x_w = static_cast<int>(x->shape[3]);
//	int o_h = (x_h + 2 * padding[0] - filter_h) / strides[0] + 1;
//	int o_w = (x_w + 2 * padding[1] - filter_w) / strides[1] + 1;
	int o_h = static_cast<int>(y->shape[2]);
	int o_w = static_cast<int>(y->shape[3]);
	#define GETX(n, c, h, w) x_data[(n) * in_channels * x_h * x_w + (c) * x_h * x_w + (h) * x_w + (w)]
	#define GETW(o, i, h, w) w_data[(o) * in_channels * filter_h * filter_w + (i) * filter_h * filter_w + (h) * filter_w + (w)]
	#define GETY(n, c, h, w) y_data[(n) * out_channels * o_h * o_w + (c) * o_h * o_w + (h) * o_w + (w)]
	auto calc_func = [&](int n, int k, int p, int q) {
		int y_sum = int(1)<<31;
		for (int r = 0; r < filter_h; ++r) {
			for (int s = 0; s < filter_w; ++s) {
				auto tp = p * stride_h + r - padding[0];
				auto tq = q * stride_w + s - padding[1];
				int32_t x_tmp = 0;
				if (!(tp < 0 || tq < 0 || tp >= x_h || tq >= x_w))
					x_tmp = GETX(n, k, tp, tq);
				y_sum = std::max(x_tmp, y_sum);
			}
		}
		return y_sum;

	};
    for (int n = 0; n < n_batch; ++n) {
        for (int k = 0; k < out_channels; ++k) {
            for (int p = 0; p < o_h; ++p) {
                for (int q = 0; q < o_w; ++q) {
                    GETY(n, k, p, q) = calc_func(n, k, p, q);
                }
            }
        }
    }

});

/*
* axis (2, 3)
*/
TVM_REGISTER_GLOBAL("tvm.runtime.cvm.sum")
    .set_body([](TVMArgs args, TVMRetValue *ret){
        CHECK(args.num_args == 3);
		DLTensor *x = args[0];
		DLTensor *y = args[1];
		std::string axis_str = args[2];
		int axis[2] = {0};
		parseToIntPair(axis_str, axis);

		int32_t *x_data = static_cast<int32_t*>(x->data);
		int32_t *y_data = static_cast<int32_t*>(y->data);
		int n_batch = static_cast<int>(x->shape[0]);
		int channels = static_cast<int>(x->shape[1]);
		int x_h = static_cast<int>(x->shape[2]);
		int x_w = static_cast<int>(x->shape[3]);
		for(int i = 0; i < n_batch; i++){
			for(int j = 0; j < channels; j++){
				int32_t sum = 0;
				for(int h = 0; h < x_h; h++){
					for(int w = 0; w < x_w; w++){
						sum += x_data[i * channels * x_h * x_w + j * x_h * x_w + h * x_w + w];
					}
				}
				y_data[i*channels + j] = sum;
			}
		}
    });


TVM_REGISTER_GLOBAL("tvm.runtime.cvm.elemwise_add")
    .set_body([](TVMArgs args, TVMRetValue *ret){
        CHECK(args.num_args == 3);
        DLTensor *args0 = args[0];
        DLTensor *args1 = args[1];
        DLTensor *args2 = args[2];
        int32_t *a = static_cast<int32_t*>(args0->data);
        int32_t *b = static_cast<int32_t*>(args1->data);
        int32_t *c = static_cast<int32_t*>(args2->data);

        for(uint32_t i = 0; i < getSize(args0); i++){
            c[i] = a[i] + b[i];
        }
    });

TVM_REGISTER_GLOBAL("tvm.runtime.cvm.reshape")
    .set_body([](TVMArgs args, TVMRetValue *ret){
        CHECK(args.num_args == 3);
         DLTensor *x = args[0];
		 DLTensor *y = args[1];
         std::string newshape = args[2];
		 if(x->data == y->data) return;
		 std::memcpy(y->data, x->data, getSize(x) * sizeof(int32_t));
    });

/*\brief:
 * x, input data
 * y, output data
 * precision, clip precision
 */
TVM_REGISTER_GLOBAL("tvm.runtime.cvm.cvm_clip")
    .set_body([](TVMArgs args, TVMRetValue *ret){
         CHECK(args.num_args == 3);
         DLTensor *x = args[0];
		 DLTensor *y = args[1];
         int32_t *x_data = static_cast<int32_t*>(x->data);
         int32_t *y_data = static_cast<int32_t*>(y->data);
         std::string str_precision = args[2];
         int32_t precision = std::atoi(str_precision.c_str());
         CHECK(precision > 0) << "precision must greater zero";
         int32_t min = -((1 << (precision-1))-1);
         int32_t max = -min;
         for(uint32_t i = 0; i < getSize(x); i++){
            y_data[i] = std::max(std::min(x_data[i], max), min);
         }
//         const char* errorStr = cuda_cvm_clip(
//                 x_data,
//                 precision,
//                 y_data,
//                 getSize(x),
//                 DEBUG_OP);
//         CHECK(errorStr == NULL) << errorStr;
    });

/*
 * a, input data
 * c, output data
 * precision, clip precision
 * b, shift b
 * */
TVM_REGISTER_GLOBAL("tvm.runtime.cvm.cvm_right_shift")
    .set_body([](TVMArgs args, TVMRetValue *ret){
        CHECK(args.num_args == 4);
        DLTensor *a = args[0];
        DLTensor *c = args[1];
        std::string str_precision = args[2];
        std::string str_b = args[3];
        int32_t precision = std::atoi(str_precision.c_str());
        int32_t b = std::atoi(str_b.c_str());
        int32_t* a_data = static_cast<int32_t*>(a->data);
        int32_t* c_data = static_cast<int32_t*>(c->data);
        CHECK_GT(precision, 0) << "precision must greater zero";
        int32_t min = -((1 << (precision-1)) - 1);
        int32_t max = -min;

        for(uint32_t i = 0; i < getSize(a); i++){
            int32_t shift_a = a_data[i];
            if(b == 0)
                c_data[i] = shift_a;
            else{
                shift_a = ((a_data[i] >> (b-1)) +1) >> 1;
                c_data[i] = std::max(std::min(shift_a, max), min);
            }
        }
    });
TVM_REGISTER_GLOBAL("tvm.runtime.cvm.cvm_left_shift")
    .set_body([](TVMArgs args, TVMRetValue *ret){
        CHECK(args.num_args == 4);
        DLTensor *a = args[0];
        DLTensor *c = args[1];
        std::string str_precision = args[2];
        std::string str_b = args[3];
        int32_t precision = std::atoi(str_precision.c_str());
        int32_t b = std::atoi(str_b.c_str());
        int32_t* a_data = static_cast<int32_t*>(a->data);
        int32_t* c_data = static_cast<int32_t*>(c->data);
        CHECK_GT(precision, 0) << "precision must greater zero";
        int32_t min = -((1 << (precision-1)) - 1);
        int32_t max = -min;

        for(uint32_t i = 0; i < getSize(a); i++){
            int32_t shift_a = a_data[i];
            if(b == 0) c_data[i] = shift_a;
            else {
                shift_a = a_data[i] << b;
                c_data[i] = std::max(std::min(shift_a, max), min);
            }
        }
    });
TVM_REGISTER_GLOBAL("tvm.runtime.cvm.log2")
    .set_body([](TVMArgs args, TVMRetValue *ret){
        CHECK(args.num_args == 2);
//        std::string x_str = args[0];
        DLTensor *dlx = args[0];
        DLTensor *y = args[1];
        int32_t *y_data = static_cast<int32_t*>(y->data);
        int32_t *x = static_cast<int32_t*>(dlx->data);
        CHECK(x[0] != 0);
        for(int i = 0; i < 64; i++){
            int64_t tmp = (int64_t)1 << i;
            if(x[0] < tmp){
                y_data[0] = i;
                return;
            }
        }
        y_data[0] = 64;
    });
TVM_REGISTER_GLOBAL("tvm.runtime.cvm.__div_scalar__")
    .set_body([](TVMArgs args, TVMRetValue *ret){
        CHECK(args.num_args == 3);
        DLTensor *dlx = args[0];
        DLTensor *y = args[1];
        std::string scalar_str = args[2];
        int32_t *y_data = static_cast<int32_t*>(y->data);
        int32_t scalar = std::atoi(scalar_str.c_str());
        int32_t* x = static_cast<int32_t*>(dlx->data);
        for(uint32_t i = 0; i < getSize(dlx); i++){
            y_data[i] = x[i] / scalar;
        }
    });
TVM_REGISTER_GLOBAL("tvm.runtime.cvm.abs")
    .set_body([](TVMArgs args, TVMRetValue *ret){
        CHECK(args.num_args == 2);
        DLTensor *dlx = args[0];
        DLTensor *y = args[1];
        int32_t *y_data = static_cast<int32_t*>(y->data);
        int32_t* x = static_cast<int32_t*>(dlx->data);
        for(uint32_t i = 0; i < getSize(dlx); i++){
            y_data[i] = std::abs(x[i]);
        }
    });
TVM_REGISTER_GLOBAL("tvm.runtime.cvm.max")
    .set_body([](TVMArgs args, TVMRetValue *ret){
        CHECK(args.num_args == 2);
        DLTensor *dlx = args[0];
        DLTensor *y = args[1];
        int32_t *y_data = static_cast<int32_t*>(y->data);
        int32_t* x = static_cast<int32_t*>(dlx->data);
        int max = x[0];
        for(uint32_t i = 1; i < getSize(dlx); i++){
            if(max < x[i]) max = x[i];
        }
        y_data[0] = max;
    });
TVM_REGISTER_GLOBAL("tvm.runtime.cvm.broadcast_max")
    .set_body([](TVMArgs args, TVMRetValue *ret){
        CHECK(args.num_args == 3);
        DLTensor *a = args[0];
        DLTensor *b = args[1];
        DLTensor *c = args[2];
        int32_t *a_data = static_cast<int32_t*>(a->data);
        int32_t* b_data = static_cast<int32_t*>(b->data);
        int32_t* c_data = static_cast<int32_t*>(c->data);

        int o_index = -1;
        for(uint32_t i = 0; i < getSize(a); i++){
            o_index = broadcast_o_index(c->shape, c->ndim, o_index);
            int32_t a_index = broadcast_i_index(c->shape, o_index, a->shape, a->ndim);
            int32_t b_index = broadcast_i_index(c->shape, o_index, b->shape, b->ndim);
            //c_data[i] = (a_data[i] > b_data[i] ? a_data[i] : b_data[i]);
            c_data[i] = a_data[a_index] > b_data[b_index] ? a_data[a_index] : b_data[b_index];
        }
    });

TVM_REGISTER_GLOBAL("tvm.runtime.cvm.concatenate")
.set_body([](TVMArgs args, TVMRetValue *ret){
        int len = args.num_args;
        CHECK(len >= 3);
        DLTensor *input0 = args[0];
        std::string str_axis = args[--len];
        DLTensor *out = args[--len];
        int32_t axis = std::atoi(str_axis.c_str());
        int32_t ndim = static_cast<int32_t>(input0->ndim);
        CHECK(-ndim <= axis && axis < ndim);
        if(axis < 0) axis += ndim;
        CHECK(axis < input0->ndim) << "axis out of bounds.";

 //       std::cout << "call concatenate: " << args.num_args << " " << axis  << " " << input0->shape[1] << " " << out->shape[1]<< std::endl;
 //       for(int i = 0; i < args.num_args-1; i++){
 //           DLTensor* dl = args[i];
 //           for(int j = 0; j < dl->ndim; j++){
 //               std::cout << dl->shape[j] << " ";
 //           }
 //           std::cout << std::endl;
 //       }

        int32_t *out_data = static_cast<int32_t*>(out->data);
        int tmpi = 0;
        for(int i = 0; i < getSize(out); i++){
            int32_t o_i = i, in_i = 0, in_i2 = 0, shapeSize = 0;
            for(int j = out->ndim-1; j >= 0; j--){
                int32_t col = o_i % out->shape[j];
                o_i /= out->shape[j];
                int32_t tmpcol = col;
                if(j == axis){
                    int32_t allShapeSize = 0;
                    for(int k = 0; k < len; k++){
                        tmpcol = col - allShapeSize;
                        DLTensor *input = args[k];
                        allShapeSize += input->shape[axis];
                        if(col < allShapeSize){
                            in_i = k;
                            break;
                        }
                    }
                }
                in_i2 += (j == out->ndim-1 ? tmpcol : tmpcol * shapeSize);
                DLTensor* input = args[in_i];
                shapeSize = (j == out->ndim-1 ? input->shape[j] : shapeSize * input->shape[j]);
            }
//            if(tmpi != in_i) {std::cout << in_i << " " << in_i2<< std::endl; tmpi = in_i;}
            DLTensor *input = args[in_i];
            int32_t *input_data = static_cast<int32_t*>(input->data);
            out_data[i] = input_data[in_i2];
        }
});

/*********************************cuda op*********************************************/
//#ifdef CVM_RUNTIME_CUDA
TVM_REGISTER_GLOBAL("tvm.runtime.cvm_cuda.elemwise_add")
.set_body([](tvm::runtime::TVMArgs args, tvm::runtime::TVMRetValue *rv){
    CHECK(args.num_args == 3);
    DLTensor *a = args[0];
    DLTensor *b = args[1];
    DLTensor *c = args[2];
    int32_t *a_data = static_cast<int32_t*>(a->data);
    int32_t *b_data = static_cast<int32_t*>(b->data);
    int32_t *c_data = static_cast<int32_t*>(c->data);
    uint32_t n = getSize(a);
    const char *errorStr = cuda_elemwise_add(a_data, b_data, c_data, n, DEBUG_OP);
    CHECK_EQ(errorStr == NULL, true) << errorStr;
});

TVM_REGISTER_GLOBAL("tvm.runtime.cvm_cuda.conv2d")
.set_body([](TVMArgs args, TVMRetValue* rv){
    CHECK(args.num_args == 13 || args.num_args == 12);
    DLTensor *x = args[0];
    DLTensor *w = args[1];
    int dlIndex = 2;
	DLTensor *b = nullptr;
    if(args.num_args == 13){
        b = args[dlIndex++];
    }
    DLTensor *y = args[dlIndex++];
	std::string groups_str = args[dlIndex++];
	std::string dilation_str = args[dlIndex++];
	std::string channels_str = args[dlIndex++];
	std::string layout_str = args[dlIndex++];
	std::string kernel_layout_str = args[dlIndex++];
	std::string kernel_size_str = args[dlIndex++];
	std::string padding_str = args[dlIndex++];
	std::string use_bias_str = args[dlIndex++];
	std::string strides_str = args[dlIndex++];
	int groups = std::atoi(groups_str.c_str());
	int dilation[2] = {0};
	parseToIntPair(dilation_str, dilation);
	//int channels = std::atoi(channels_str.c_str());
	int kernel_size[2] = {0};
	parseToIntPair(kernel_size_str, kernel_size);
	int padding[2] = {0};
	parseToIntPair(padding_str, padding);
	int strides[2] = {0};
	parseToIntPair(strides_str, strides);

    int32_t* x_data = (int32_t*)x->data;
    int32_t* w_data = (int32_t*)w->data;
    int32_t* y_data = (int32_t*)y->data;
	int32_t* b_data = b != nullptr ? (int32_t*)b->data : nullptr;

    int out_channels = static_cast<int>(w->shape[0]);
    int filter_h = static_cast<int>(w->shape[2]);
    int filter_w = static_cast<int>(w->shape[3]);
	filter_h = (filter_h - 1) * dilation[0] + 1;
	filter_w = (filter_w - 1) * dilation[1] + 1;

    int n_batch = static_cast<int>(x->shape[0]);
    int in_channels = static_cast<int>(x->shape[1]);
    int x_h = static_cast<int>(x->shape[2]);
    int x_w = static_cast<int>(x->shape[3]);
	int o_h = (x_h + 2 * padding[0] - filter_h) / strides[0] + 1;
	int o_w = (x_w + 2 * padding[1] - filter_w) / strides[1] + 1;
//	int o_h = static_cast<int>(y->shape[2]);
//	int o_w = static_cast<int>(y->shape[3]);

    if(groups == 1){
        const char* errorStr = cuda_conv2d(
                x_data, n_batch, in_channels, x_h, x_w,
                w_data, out_channels, in_channels, filter_h, filter_w,
                b_data,
                padding[0], padding[1],
                strides[0], strides[1],
                dilation[0], dilation[1],
                groups,
                y_data, n_batch, out_channels, o_h, o_w, x->ctx.device_id, DEBUG_OP);
        CHECK_EQ(errorStr == NULL, true) << errorStr;
    }else{
        const char* errorStr = cuda_depthwise_conv2d(
                x_data, n_batch, in_channels, x_h, x_w,
                w_data, out_channels, in_channels, filter_h, filter_w,
                b_data,
                padding[0], padding[1],
                strides[0], strides[1],
                dilation[0], dilation[1],
                groups,
                y_data, n_batch, out_channels, o_h, o_w, x->ctx.device_id, DEBUG_OP);
        CHECK_EQ(errorStr == NULL, true) << errorStr;

    }
 });

TVM_REGISTER_GLOBAL("tvm.runtime.cvm.cuda_max_pool2d")
    .set_body([](TVMArgs args, TVMRetValue *ret){
    CHECK(args.num_args == 6);
	DLTensor *x = args[0];
	DLTensor *y = args[1];
	std::string strides_str = args[2];
	std::string pool_size_str = args[3];
	std::string ceil_mode = args[4];
	std::string padding_str = args[5];
	int strides[2] = {0};
	parseToIntPair(strides_str, strides);
	int pool_size[2] = {0};
	parseToIntPair(pool_size_str, pool_size);
	int padding[2] = {0};
	parseToIntPair(padding_str, padding);

    int32_t* x_data = (int32_t*)x->data;
    int32_t* y_data = (int32_t*)y->data;

    int filter_h = pool_size[0];
    int filter_w = pool_size[1];

    int n_batch = static_cast<int>(x->shape[0]);
    int in_channels = static_cast<int>(x->shape[1]);
	int out_channels = in_channels;
    int x_h = static_cast<int>(x->shape[2]);
    int x_w = static_cast<int>(x->shape[3]);
//	int o_h = (x_h + 2 * padding[0] - filter_h) / strides[0] + 1;
//	int o_w = (x_w + 2 * padding[1] - filter_w) / strides[1] + 1;
	int o_h = static_cast<int>(y->shape[2]);
	int o_w = static_cast<int>(y->shape[3]);

    const char* errorStr = cuda_max_pool(
            x_data, n_batch, in_channels, x_h, x_w,
            filter_h, filter_w,
            padding[0], padding[1],
            strides[0], strides[1],
            y_data, n_batch, out_channels, o_h, o_w, x->ctx.device_id, DEBUG_OP);
    CHECK_EQ(errorStr == NULL, true) << errorStr;
    });

TVM_REGISTER_GLOBAL("tvm.runtime.cvm_cuda.dense")
.set_body([](TVMArgs args, TVMRetValue* rv) {
  CHECK(args.num_args == 6 || args.num_args == 5);
  int ndim = args.num_args;
  DLTensor *x = args[0];
  DLTensor *w = args[1];
  DLTensor *b = nullptr;
  DLTensor *y = nullptr;
  int32_t* db = nullptr;
  if(ndim == 6){
	b = args[2];
	y = args[3];
    db = static_cast<int32_t*>(b->data);
  }else{
	y = args[2];
  }
  auto dx = static_cast<int32_t*>(x->data);
  auto dy = static_cast<int32_t*>(y->data);
  auto dw = static_cast<int32_t*>(w->data);

  const char* errorStr = cuda_dense(
          dx, dw, dy,
          static_cast<int32_t>(x->shape[0]),
          static_cast<int32_t>(x->shape[1]),
          static_cast<int32_t>(y->shape[1]),
          db,
          DEBUG_OP);
    CHECK_EQ(errorStr == NULL, true) << errorStr;
});

TVM_REGISTER_GLOBAL("tvm.runtime.cvm_cuda.clip").set_body([](TVMArgs args, TVMRetValue* rv) {
   CHECK(args.num_args == 4);
   DLTensor *x = args[0];
   DLTensor *y = args[1];
   std::string min_str = args[2];
   std::string max_str = args[3];
   int min = std::atoi(min_str.c_str());
   int max = std::atoi(max_str.c_str());

   const char *errorStr = cuda_clip(
           static_cast<int32_t*>(x->data),
           static_cast<int32_t*>(y->data),
           getSize(x),
           max, min, DEBUG_OP);
   CHECK_EQ(errorStr == NULL, true) << errorStr;
 });

 TVM_REGISTER_GLOBAL("tvm.runtime.cvm_cuda.relu").set_body([](TVMArgs args, TVMRetValue* rv) {
   CHECK(args.num_args == 2);
   DLTensor *x = args[0];
   DLTensor *y = args[1];
   const char* errorStr = cuda_relu(
           static_cast<int32_t*>(x->data),
           static_cast<int32_t*>(y->data),
           getSize(x),
           DEBUG_OP);
    CHECK_EQ(errorStr == NULL, true) << errorStr;
 });
TVM_REGISTER_GLOBAL("tvm.runtime.cvm_cuda.flatten").set_body([]
(TVMArgs args, TVMRetValue* rv){
     CHECK(args.num_args == 2);
     DLTensor *x = args[0];
     DLTensor *y = args[1];

     const char* errorStr = cuda_flatten(
            static_cast<int32_t*>(x->data),
            static_cast<int32_t*>(y->data),
            getSize(x),
            DEBUG_OP);
     CHECK_EQ(errorStr == NULL, true) << errorStr;
});
TVM_REGISTER_GLOBAL("tvm.runtime.cvm_cuda.broadcast_add")
    .set_body([](TVMArgs args, TVMRetValue *ret){
        CHECK(args.num_args == 3);
        DLTensor *args0 = args[0];
        DLTensor *args1 = args[1];
        DLTensor *args2 = args[2];
        int32_t *a = static_cast<int32_t*>(args0->data);
        int32_t *b = static_cast<int32_t*>(args1->data);
        int32_t *c = static_cast<int32_t*>(args2->data);

        const char* errorStr = cuda_broadcast_add(a, b, c, getSize(args0),
                args0->shape, args0->ndim,
                args1->shape, args1->ndim,
                args2->shape, args2->ndim,
                DEBUG_OP);
        CHECK_EQ(errorStr == NULL, true) << errorStr;
    });

TVM_REGISTER_GLOBAL("tvm.runtime.cvm_cuda.broadcast_sub")
    .set_body([](TVMArgs args, TVMRetValue *ret){
        CHECK(args.num_args == 3);
        DLTensor *args0 = args[0];
        DLTensor *args1 = args[1];
        DLTensor *args2 = args[2];
        int32_t *a = static_cast<int32_t*>(args0->data);
        int32_t *b = static_cast<int32_t*>(args1->data);
        int32_t *c = static_cast<int32_t*>(args2->data);

        const char* errorStr = cuda_broadcast_sub(a, b, c, getSize(args0), DEBUG_OP);
        CHECK_EQ(errorStr == NULL, true) << errorStr;
    });
TVM_REGISTER_GLOBAL("tvm.runtime.cvm_cuda.broadcast_mul")
    .set_body([](TVMArgs args, TVMRetValue *ret){
        CHECK(args.num_args == 3);
        DLTensor *args0 = args[0];
        DLTensor *args1 = args[1];
        DLTensor *args2 = args[2];
        int32_t *a = static_cast<int32_t*>(args0->data);
        int32_t *b = static_cast<int32_t*>(args1->data);
        int32_t *c = static_cast<int32_t*>(args2->data);

        const char* errorStr = cuda_broadcast_mul(a, b, c, getSize(args0), DEBUG_OP);
        CHECK_EQ(errorStr == NULL, true) << errorStr;
    });
TVM_REGISTER_GLOBAL("tvm.runtime.cvm_cuda.broadcast_div")
    .set_body([](TVMArgs args, TVMRetValue *ret){
        CHECK(args.num_args == 3);
        DLTensor *args0 = args[0];
        DLTensor *args1 = args[1];
        DLTensor *args2 = args[2];
        int32_t *a = static_cast<int32_t*>(args0->data);
        int32_t *b = static_cast<int32_t*>(args1->data);
        int32_t *c = static_cast<int32_t*>(args2->data);

        const char* errorStr = cuda_broadcast_div(a, b, c, getSize(args0), DEBUG_OP);
        CHECK_EQ(errorStr == NULL, true) << errorStr;
    });
TVM_REGISTER_GLOBAL("tvm.runtime.cvm_cuda.broadcast_right_shift")
    .set_body([](TVMArgs args, TVMRetValue *ret){
        CHECK(args.num_args == 3);
        DLTensor *args0 = args[0];
        DLTensor *args1 = args[1];
        DLTensor *args2 = args[2];
        int32_t *a = static_cast<int32_t*>(args0->data);
        int32_t *b = static_cast<int32_t*>(args1->data);
        int32_t *c = static_cast<int32_t*>(args2->data);

        const char* errorStr = cuda_broadcast_right_shift(a, b, c, getSize(args0), DEBUG_OP);
        CHECK_EQ(errorStr == NULL, true) << errorStr;
    });
TVM_REGISTER_GLOBAL("tvm.runtime.cvm_cuda.broadcast_left_shift")
    .set_body([](TVMArgs args, TVMRetValue *ret){
        CHECK(args.num_args == 3);
        DLTensor *args0 = args[0];
        DLTensor *args1 = args[1];
        DLTensor *args2 = args[2];
        int32_t *a = static_cast<int32_t*>(args0->data);
        int32_t *b = static_cast<int32_t*>(args1->data);
        int32_t *c = static_cast<int32_t*>(args2->data);

        const char* errorStr = cuda_broadcast_left_shift(a, b, c, getSize(args0), DEBUG_OP);
        CHECK_EQ(errorStr == NULL, true) << errorStr;
    });

/*
* strides (2, 2)
* pool_size [3, 3]
* ceil_mode False
* padding (1, 1)
*/
TVM_REGISTER_GLOBAL("tvm.runtime.cvm_cuda.max_pool2d")
    .set_body([](TVMArgs args, TVMRetValue *ret){
            CHECK(args.num_args == 6);
            DLTensor *x = args[0];
            DLTensor *y = args[1];
            std::string strides_str = args[2];
            std::string pool_size_str = args[3];
            std::string ceil_mode = args[4];
            std::string padding_str = args[5];
            int strides[2] = {0};
            parseToIntPair(strides_str, strides);
            int pool_size[2] = {0};
            parseToIntPair(pool_size_str, pool_size);
            int padding[2] = {0};
            parseToIntPair(padding_str, padding);

            int32_t* x_data = (int32_t*)x->data;
            int32_t* y_data = (int32_t*)y->data;

            int filter_h = pool_size[0];
            int filter_w = pool_size[1];

            int n_batch = static_cast<int>(x->shape[0]);
            int in_channels = static_cast<int>(x->shape[1]);
            int out_channels = in_channels;
            int x_h = static_cast<int>(x->shape[2]);
            int x_w = static_cast<int>(x->shape[3]);
            //	int o_h = (x_h + 2 * padding[0] - filter_h) / strides[0] + 1;
            //	int o_w = (x_w + 2 * padding[1] - filter_w) / strides[1] + 1;
            int o_h = static_cast<int>(y->shape[2]);
            int o_w = static_cast<int>(y->shape[3]);
            const char* errorStr = cuda_max_pool(
                    x_data, n_batch, in_channels, x_h, x_w,
                    filter_h, filter_w,
                    padding[0], padding[1],
                    strides[0], strides[1],
                    y_data, n_batch, out_channels, o_h, o_w, x->ctx.device_id, DEBUG_OP);
            CHECK_EQ(errorStr == NULL, true) << errorStr;
});

/*
* axis (2, 3)
*/
TVM_REGISTER_GLOBAL("tvm.runtime.cvm_cuda.sum")
    .set_body([](TVMArgs args, TVMRetValue *ret){
        CHECK(args.num_args == 3);
		DLTensor *x = args[0];
		DLTensor *y = args[1];
		std::string axis_str = args[2];
		int axis[2] = {0};
		parseToIntPair(axis_str, axis);

		int32_t *x_data = static_cast<int32_t*>(x->data);
		int32_t *y_data = static_cast<int32_t*>(y->data);
		int n_batch = static_cast<int>(x->shape[0]);
		int channels = static_cast<int>(x->shape[1]);
		int x_h = static_cast<int>(x->shape[2]);
		int x_w = static_cast<int>(x->shape[3]);
        const char* errorStr = cuda_sum(x_data, n_batch, channels, x_h, x_w, y_data, DEBUG_OP);
        CHECK_EQ(errorStr == NULL, true) << errorStr;
    });

TVM_REGISTER_GLOBAL("tvm.runtime.cvm_cuda.reshape")
    .set_body([](TVMArgs args, TVMRetValue *ret){
        CHECK(args.num_args == 3);
         DLTensor *x = args[0];
		 DLTensor *y = args[1];
         std::string newshape = args[2];
         const char* errorStr = cuda_reshape(
                 static_cast<int32_t*>(x->data),
                 static_cast<int32_t*>(y->data),
                 getSize(x),
                 DEBUG_OP);
         CHECK_EQ(errorStr == NULL, true) << errorStr;
    });
/*\brief:
 * x, input data
 * y, output data
 * precision, clip precision
 */
TVM_REGISTER_GLOBAL("tvm.runtime.cvm_cuda.cvm_clip")
    .set_body([](TVMArgs args, TVMRetValue *ret){
        CHECK(args.num_args == 3);
         DLTensor *x = args[0];
		 DLTensor *y = args[1];
         int32_t *x_data = static_cast<int32_t*>(x->data);
         int32_t *y_data = static_cast<int32_t*>(y->data);
         std::string str_precision = args[2];
         int32_t precision = std::atoi(str_precision.c_str());
         CHECK(precision > 0) << "precision must greater zero";
         const char* errorStr = cuda_cvm_clip(
                 x_data,
                 precision,
                 y_data,
                 getSize(x),
                 DEBUG_OP);
         CHECK(errorStr == NULL) << errorStr;
    });

/*
 * a, input data
 * c, output data
 * precision, clip precision
 * b, shift b
 * */
TVM_REGISTER_GLOBAL("tvm.runtime.cvm_cuda.cvm_right_shift")
    .set_body([](TVMArgs args, TVMRetValue *ret){
        CHECK(args.num_args == 4);
        DLTensor *a = args[0];
        DLTensor *c = args[1];
        std::string str_precision = args[2];
        std::string str_b = args[3];
        int32_t precision = std::atoi(str_precision.c_str());
        int32_t b = std::atoi(str_b.c_str());
        int32_t* a_data = static_cast<int32_t*>(a->data);
        int32_t* c_data = static_cast<int32_t*>(c->data);
        CHECK_GT(precision, 0) << "precision must greater zero";
        const char* errorStr = cuda_cvm_right_shift(
                a_data,
                b,
                precision,
                c_data,
                getSize(a),
                DEBUG_OP);
        CHECK(errorStr == NULL) << errorStr;

    });
TVM_REGISTER_GLOBAL("tvm.runtime.cvm_cuda.cvm_left_shift")
    .set_body([](TVMArgs args, TVMRetValue *ret){
        CHECK(args.num_args == 4);
        DLTensor *a = args[0];
        DLTensor *c = args[1];
        std::string str_precision = args[2];
        std::string str_b = args[3];
        int32_t precision = std::atoi(str_precision.c_str());
        int32_t b = std::atoi(str_b.c_str());
        int32_t* a_data = static_cast<int32_t*>(a->data);
        int32_t* c_data = static_cast<int32_t*>(c->data);
        CHECK_GT(precision, 0) << "precision must greater zero";
        const char* errorStr = cuda_cvm_left_shift(
                a_data,
                b,
                precision,
                c_data,
                getSize(a),
                DEBUG_OP);
        CHECK(errorStr == NULL) << errorStr;
    });
TVM_REGISTER_GLOBAL("tvm.runtime.cvm_cuda.log2")
    .set_body([](TVMArgs args, TVMRetValue *ret){
//        std::string x_str = args[0];
        CHECK(args.num_args == 2);
        DLTensor *dlx = args[0];
        DLTensor *y = args[1];
        int32_t *y_data = static_cast<int32_t*>(y->data);
        int32_t *x = static_cast<int32_t*>(dlx->data);
        const char* errorStr = cuda_log(x, y_data, DEBUG_OP);
        CHECK(errorStr == NULL) << errorStr;
    });
TVM_REGISTER_GLOBAL("tvm.runtime.cvm_cuda.abs")
    .set_body([](TVMArgs args, TVMRetValue *ret){
        CHECK(args.num_args == 2);
        DLTensor *dlx = args[0];
        DLTensor *y = args[1];
        int32_t *y_data = static_cast<int32_t*>(y->data);
        int32_t* x = static_cast<int32_t*>(dlx->data);
        const char* errorStr = cuda_abs(x, y_data, getSize(dlx), DEBUG_OP);
        CHECK(errorStr == NULL) << errorStr;
    });
TVM_REGISTER_GLOBAL("tvm.runtime.cvm_cuda.max")
    .set_body([](TVMArgs args, TVMRetValue *ret){
        CHECK(args.num_args == 2);
        DLTensor *dlx = args[0];
        DLTensor *y = args[1];
        int32_t *y_data = static_cast<int32_t*>(y->data);
        int32_t* x = static_cast<int32_t*>(dlx->data);
        const char* errorStr = cuda_max(x, y_data, getSize(dlx), DEBUG_OP);
        CHECK(errorStr == NULL) << errorStr;
    });
TVM_REGISTER_GLOBAL("tvm.runtime.cvm_cuda.broadcast_max")
    .set_body([](TVMArgs args, TVMRetValue *ret){
        CHECK(args.num_args == 3);
        DLTensor *a = args[0];
        DLTensor *b = args[1];
        DLTensor *c = args[2];
        int32_t *a_data = static_cast<int32_t*>(a->data);
        int32_t* b_data = static_cast<int32_t*>(b->data);
        int32_t* c_data = static_cast<int32_t*>(c->data);
        const char* errorStr = cuda_broadcast_max(
                a_data,
                b_data,
                c_data,
                getSize(a),
                DEBUG_OP);
        CHECK(errorStr == NULL) << errorStr;
    });
//#endif // end of CVM_RUNTIME_CUDA
}
}

