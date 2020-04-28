#include <cvm/runtime/packed_func.h>
#include <cvm/runtime/registry.h>

#include <cvm/op.h>
#include <cvm/node.h>
#include <cvm/top/tensor.h>
#include <cvm/top/nn.h>

#include <algorithm>
#include <functional>
#include <numeric>
#include <vector>
#include <string>
#include <memory>
#include <utility>

#include "ops.h"

namespace cvm{
  namespace runtime{

inline uint64_t getSize(DLTensor *dlTensor){
  uint64_t size = 1;
  for(int i = 0; i < dlTensor->ndim; i++){
    size *= dlTensor->shape[i];
  }
  return size;
}

//CVM_REGISTER_GLOBAL("cvm.runtime.opencl.elemwise_add")
//  .set_body([](cvm::runtime::CVMArgs args, cvm::runtime::CVMRetValue *rv){
//      DLTensor *a = args[0];
//      DLTensor *b = args[1];
//      DLTensor *c = args[2];
//      void *a_data = (a->data);
//      void *b_data = (b->data);
//      void *c_data = (c->data);
//      uint64_t n = getSize(a);
//      //int error_code = 0;
//      opencl_elemwise(a_data, b_data, c_data, n, 0);
//});
//
CVM_REGISTER_GLOBAL("cvm.runtime.opencl.conv2d")
  .set_body([](CVMArgs args, CVMRetValue* rv){
      DLTensor *x = args[0];
      DLTensor *w = args[1];
      DLTensor *b = nullptr;
      DLTensor *y = nullptr;
      void *_attr;

      if(args.num_args == 5){
      b = args[2];
      y = args[3];
      _attr = args[4];
      } else {
      y = args[2];
      _attr = args[3];
      }
      auto *attr = static_cast<cvm::NodeAttrs*>(_attr);
      auto &param = cvm::get<cvm::top::Conv2DParam>(attr->parsed);
      int groups = param.groups;
      int dilation[2] = {static_cast<int>(param.dilation[0]), static_cast<int>(param.dilation[1])};
      int padding[2] = {static_cast<int>(param.padding[0]), static_cast<int>(param.padding[1])};
      int strides[2] = {static_cast<int>(param.strides[0]), static_cast<int>(param.strides[1])};
      bool use_bias = param.use_bias;


      void* x_data = x->data;
      void* w_data = w->data;
      void* y_data = y->data;
      void* b_data = b != nullptr ? b->data : nullptr;

      int out_channels = static_cast<int>(w->shape[0]);
      int filter_c = static_cast<int>(w->shape[1]);
      int filter_h = static_cast<int>(w->shape[2]);
      int filter_w = static_cast<int>(w->shape[3]);
      int t_filter_h = (filter_h - 1) * dilation[0] + 1;
      int t_filter_w = (filter_w - 1) * dilation[1] + 1;

      int n_batch = static_cast<int>(x->shape[0]);
      int in_channels = static_cast<int>(x->shape[1]);
      int x_h = static_cast<int>(x->shape[2]);
      int x_w = static_cast<int>(x->shape[3]);
      int o_h = (x_h + 2 * padding[0] - t_filter_h) / strides[0] + 1;
      int o_w = (x_w + 2 * padding[1] - t_filter_w) / strides[1] + 1;

      if(groups == 1){
        void *ext_space = args.ext_space->data;
        int32_t ext_space_size = args.ext_space->shape[0];
        opencl_conv2d(x_data, w_data, b_data, y_data, 
            n_batch, in_channels, x_h, x_w, 
            out_channels, filter_h, filter_w, 
            o_h, o_w, 
            padding[0], padding[1],
            strides[0], strides[1],
            dilation[0], dilation[1],
            use_bias,
            ext_space, ext_space_size);
      }else{
        opencl_groupwise_conv2d(x_data, n_batch, in_channels, x_h, x_w, 
            w_data, filter_c, filter_h, filter_w,
            y_data, out_channels, o_h, o_w, 
            b_data,
            padding[0], padding[1],
            strides[0], strides[1],
            dilation[0], dilation[1],
            groups, 
            use_bias);
      }
  });

CVM_REGISTER_GLOBAL("cvm.runtime.opencl.dense")
  .set_body([](CVMArgs args, CVMRetValue* rv) {
      int ndim = args.num_args;
      DLTensor *x = args[0];
      DLTensor *w = args[1];
      DLTensor *bias = nullptr;
      DLTensor *y = nullptr;
      void* bias_data = nullptr;
      void* _attr;
      if(ndim == 5){
        bias = args[2];
        y = args[3];
        bias_data = bias->data;
        _attr = args[4];
      } else{
        y = args[2];
        _attr = args[3];
      }

      auto *attr = static_cast<cvm::NodeAttrs*>(_attr);
      auto &param = cvm::get<cvm::top::DenseParam>(attr->parsed);
      void* x_data = x->data;
      void* y_data = y->data;
      void* w_data = w->data;
      bool use_bias = param.use_bias;
      opencl_dense(x_data, w_data, bias_data, y_data, y->shape[0], y->shape[1], x->shape[1], use_bias);
      //int error_code = NON_ERROR;
      //const char* errorStr = opencl_dense(
      //    x_data, w_data, y_data,
      //    static_cast<int32_t>(x->shape[0]),
      //    static_cast<int32_t>(x->shape[1]),
      //    static_cast<int32_t>(y->shape[1]),
      //    bias_data,
      //    error_code);

      //deal_error(error_code, errorStr);
  });

CVM_REGISTER_GLOBAL("cvm.runtime.opencl.relu")
  .set_body([](CVMArgs args, CVMRetValue* rv) {
      DLTensor *x = args[0];
      DLTensor *y = args[1];

      opencl_relu(x->data, y->data, getSize(x));
      //int error_code = NON_ERROR;
      //const char* errorStr = opencl_relu(
      //    static_cast<int32_t*>(x->data),
      //    static_cast<int32_t*>(y->data),
      //    getSize(x),
      //    error_code);
      //deal_error(error_code, errorStr);
  });

CVM_REGISTER_GLOBAL("cvm.runtime.opencl.flatten")
  .set_body([](CVMArgs args, CVMRetValue* rv){
      DLTensor *x = args[0];
      DLTensor *y = args[1];

      opencl_flatten(x->data, y->data, getSize(x));
      //int error_code = NON_ERROR;
      //const char* errorStr = opencl_flatten(
      //    static_cast<int32_t*>(x->data),
      //    static_cast<int32_t*>(y->data),
      //    getSize(x),
      //    error_code);
      //deal_error(error_code, errorStr);
  });
CVM_REGISTER_GLOBAL("cvm.runtime.opencl.reshape")
  .set_body([](CVMArgs args, CVMRetValue* rv){
      DLTensor *x = args[0];
      DLTensor *y = args[1];

      opencl_flatten(x->data, y->data, getSize(x));
  });

CVM_REGISTER_GLOBAL("cvm.runtime.opencl.broadcast_add")
  .set_body([](CVMArgs args, CVMRetValue *ret){
      DLTensor *args0 = args[0];
      DLTensor *args1 = args[1];
      DLTensor *args2 = args[2];
      void *a = args0->data;
      void *b = args1->data;
      void *c = args2->data;
      int64_t *ashape = static_cast<int64_t*>(args0->shape);
      int32_t adim = static_cast<int32_t>(args0->ndim);
      int64_t *bshape = static_cast<int64_t*>(args1->shape);
      int32_t bdim = static_cast<int32_t>(args1->ndim);
      int64_t *cshape = static_cast<int64_t*>(args2->shape);
      int32_t cdim = static_cast<int32_t>(args2->ndim);

     // if(bdim == 1 && bshape[0] == 1)
     //   opencl_broadcast_mul(a, b, c, getSize(args0));
     // else printf("no support b dim > 1\n");
     opencl_broadcast(a, b, c, ashape, bshape, cshape, adim, bdim, cdim, getSize(args0), getSize(args1), getSize(args2), 0);
  });

CVM_REGISTER_GLOBAL("cvm.runtime.opencl.broadcast_sub")
  .set_body([](CVMArgs args, CVMRetValue *ret){
      DLTensor *args0 = args[0];
      DLTensor *args1 = args[1];
      DLTensor *args2 = args[2];
      void *a = args0->data;
      void *b = args1->data;
      void *c = args2->data;
      int64_t *ashape = static_cast<int64_t*>(args0->shape);
      int32_t adim = static_cast<int32_t>(args0->ndim);
      int64_t *bshape = static_cast<int64_t*>(args1->shape);
      int32_t bdim = static_cast<int32_t>(args1->ndim);
      int64_t *cshape = static_cast<int64_t*>(args2->shape);
      int32_t cdim = static_cast<int32_t>(args2->ndim);

     // if(bdim == 1 && bshape[0] == 1)
     //   opencl_broadcast_mul(a, b, c, getSize(args0));
     // else printf("no support b dim > 1\n");
     opencl_broadcast(a, b, c, ashape, bshape, cshape, adim, bdim, cdim, getSize(args0), getSize(args1), getSize(args2), 1);
  });

CVM_REGISTER_GLOBAL("cvm.runtime.opencl.broadcast_mul")
  .set_body([](CVMArgs args, CVMRetValue *ret){
      DLTensor *args0 = args[0];
      DLTensor *args1 = args[1];
      DLTensor *args2 = args[2];
      void *a = args0->data;
      void *b = args1->data;
      void *c = args2->data;
      int64_t *ashape = static_cast<int64_t*>(args0->shape);
      int32_t adim = static_cast<int32_t>(args0->ndim);
      int64_t *bshape = static_cast<int64_t*>(args1->shape);
      int32_t bdim = static_cast<int32_t>(args1->ndim);
      int64_t *cshape = static_cast<int64_t*>(args2->shape);
      int32_t cdim = static_cast<int32_t>(args2->ndim);

      if(bdim == 1 && bshape[0] == 1){
        opencl_broadcast_mul(a, b, c, getSize(args0));
      }
      else{ 
        opencl_broadcast(a, b, c, ashape, bshape, cshape, adim, bdim, cdim, getSize(args0), getSize(args1), getSize(args2), 2);
      }
  });

CVM_REGISTER_GLOBAL("cvm.runtime.opencl.broadcast_max")
  .set_body([](CVMArgs args, CVMRetValue *ret){
      DLTensor *args0 = args[0];
      DLTensor *args1 = args[1];
      DLTensor *args2 = args[2];
      void *a = args0->data;
      void *b = args1->data;
      void *c = args2->data;
      int64_t *ashape = static_cast<int64_t*>(args0->shape);
      int32_t adim = static_cast<int32_t>(args0->ndim);
      int64_t *bshape = static_cast<int64_t*>(args1->shape);
      int32_t bdim = static_cast<int32_t>(args1->ndim);
      int64_t *cshape = static_cast<int64_t*>(args2->shape);
      int32_t cdim = static_cast<int32_t>(args2->ndim);

     // if(bdim == 1 && bshape[0] == 1)
     //   opencl_broadcast_mul(a, b, c, getSize(args0));
     // else printf("no support b dim > 1\n");
     opencl_broadcast(a, b, c, ashape, bshape, cshape, adim, bdim, cdim, getSize(args0), getSize(args1), getSize(args2), 3);
  });

CVM_REGISTER_GLOBAL("cvm.runtime.opencl.broadcast_div")
  .set_body([](CVMArgs args, CVMRetValue *ret){
      DLTensor *args0 = args[0];
      DLTensor *args1 = args[1];
      DLTensor *args2 = args[2];
      void *a = args0->data;
      void *b = args1->data;
      void *c = args2->data;
      int64_t *ashape = static_cast<int64_t*>(args0->shape);
      int32_t adim = static_cast<int32_t>(args0->ndim);
      int64_t *bshape = static_cast<int64_t*>(args1->shape);
      int32_t bdim = static_cast<int32_t>(args1->ndim);
      int64_t *cshape = static_cast<int64_t*>(args2->shape);
      int32_t cdim = static_cast<int32_t>(args2->ndim);

     // if(bdim == 1 && bshape[0] == 1)
     //   opencl_broadcast_mul(a, b, c, getSize(args0));
     // else printf("no support b dim > 1\n");
     opencl_broadcast(a, b, c, ashape, bshape, cshape, adim, bdim, cdim, getSize(args0), getSize(args1), getSize(args2), 4);
  });

CVM_REGISTER_GLOBAL("cvm.runtime.opencl.broadcast_greater")
  .set_body([](CVMArgs args, CVMRetValue *ret){
      DLTensor *args0 = args[0];
      DLTensor *args1 = args[1];
      DLTensor *args2 = args[2];
      void *a = args0->data;
      void *b = args1->data;
      void *c = args2->data;
      int64_t *ashape = static_cast<int64_t*>(args0->shape);
      int32_t adim = static_cast<int32_t>(args0->ndim);
      int64_t *bshape = static_cast<int64_t*>(args1->shape);
      int32_t bdim = static_cast<int32_t>(args1->ndim);
      int64_t *cshape = static_cast<int64_t*>(args2->shape);
      int32_t cdim = static_cast<int32_t>(args2->ndim);

     // if(bdim == 1 && bshape[0] == 1)
     //   opencl_broadcast_mul(a, b, c, getSize(args0));
     // else printf("no support b dim > 1\n");
     opencl_broadcast(a, b, c, ashape, bshape, cshape, adim, bdim, cdim, getSize(args0), getSize(args1), getSize(args2), 5);
  });

CVM_REGISTER_GLOBAL("cvm.runtime.opencl.max_pool2d")
  .set_body([](CVMArgs args, CVMRetValue *ret){
      DLTensor *x = args[0];
      DLTensor *y = args[1];
      void *_attr = args[2];
      auto *attr = static_cast<cvm::NodeAttrs*>(_attr);
      auto &param = cvm::get<cvm::top::MaxPool2DParam>(attr->parsed);
      int strides[2] = {static_cast<int>(param.strides[0]), static_cast<int>(param.strides[1])};
      int pool_size[2] = {static_cast<int>(param.pool_size[0]), static_cast<int>(param.pool_size[1])};
      int padding[2] = {static_cast<int>(param.padding[0]), static_cast<int>(param.padding[0])};
      if(param.padding.ndim() == 2){
        padding[1] = static_cast<int>(param.padding[1]);
      }
      // bool ceil_mode = param.ceil_mode;

      int stride_h = strides[0];
      int stride_w = strides[1];

      void* x_data = x->data;
      void* y_data = y->data;

      int filter_h = pool_size[0];
      int filter_w = pool_size[1];

      int n_batch = static_cast<int>(x->shape[0]);
      int in_channels = static_cast<int>(x->shape[1]);
      //int out_channels = in_channels;
      int x_h = static_cast<int>(x->shape[2]);
      int x_w = static_cast<int>(x->shape[3]);
      //  int o_h = (x_h + 2 * padding[0] - filter_h) / strides[0] + 1;
      //  int o_w = (x_w + 2 * padding[1] - filter_w) / strides[1] + 1;
      int o_h = static_cast<int>(y->shape[2]);
      int o_w = static_cast<int>(y->shape[3]);

      opencl_max_pool2d(x_data, y_data, 
          n_batch, in_channels, x_h, x_w,
          filter_h, filter_w,
          o_h, o_w,
          padding[0], padding[1],
          stride_h, stride_w);
      //int error_code = NON_ERROR;
      //const char* errorStr = opencl_max_pool(
      //    x_data, n_batch, in_channels, x_h, x_w,
      //    filter_h, filter_w,
      //    padding[0], padding[1],
      //    stride_h, stride_w,
      //    y_data, n_batch, out_channels, o_h, o_w, x->ctx.device_id, error_code);
      //deal_error(error_code, errorStr);
  });


CVM_REGISTER_GLOBAL("cvm.runtime.opencl.clip")
  .set_body([](CVMArgs args, CVMRetValue* rv) {
      DLTensor *x = args[0];
      DLTensor *y = args[1];
      void *_attr = args[2];
      auto *attr = static_cast<cvm::NodeAttrs*>(_attr);
      auto& param = cvm::get<cvm::top::ClipParam>(attr->parsed);
      int max = param.a_max;
      int min = param.a_min;

      opencl_clip(
          x->data,
          y->data,
          getSize(x),
          max, min);
  });

CVM_REGISTER_GLOBAL("cvm.runtime.opencl.cvm_clip")
  .set_body([](CVMArgs args, CVMRetValue *ret){
      DLTensor *x = args[0];
      DLTensor *y = args[1];
      void *x_data = x->data;
      void *y_data = y->data;
      void *_attr = args[2];
      auto *attr = static_cast<cvm::NodeAttrs*>(_attr);
      auto &param = cvm::get<cvm::top::CVMClipParam>(attr->parsed);
      int32_t precision = param.precision;

      opencl_cvm_clip(x_data, y_data, getSize(x), precision);
  });

CVM_REGISTER_GLOBAL("cvm.runtime.opencl.cvm_right_shift")
  .set_body([](CVMArgs args, CVMRetValue *ret){
      DLTensor *a = args[0];
      DLTensor *c = args[1];
      void *_attr = args[2];
      auto *attr = static_cast<cvm::NodeAttrs*>(_attr);
      auto &param = cvm::get<cvm::top::CVMRightShiftParam>(attr->parsed);
      int32_t precision = param.precision;
      int32_t b = param.shift_bit;
      void* a_data = a->data;
      void* c_data = c->data;

      opencl_cvm_right_shift(a_data, c_data, b, getSize(a), precision);
  });

CVM_REGISTER_GLOBAL("cvm.runtime.opencl.cvm_left_shift")
  .set_body([](CVMArgs args, CVMRetValue *ret){
      DLTensor *a = args[0];
      DLTensor *c = args[1];
      void *_attr = args[2];
      auto *attr = static_cast<cvm::NodeAttrs*>(_attr);
      auto &param = cvm::get<cvm::top::CVMLeftShiftParam>(attr->parsed);
      int32_t precision = param.precision;
      int32_t b = param.shift_bit;std::string str_precision = args[2];
      void* a_data = a->data;
      void* c_data = c->data;
      opencl_cvm_left_shift(a_data, c_data, b, getSize(a), precision);
  });


CVM_REGISTER_GLOBAL("cvm.runtime.opencl.concatenate")
  .set_body([](CVMArgs args, CVMRetValue *ret){
      int len = args.num_args;
      DLTensor *input0 = args[0];
      void *_attr = args[--len];
      auto *attr = static_cast<cvm::NodeAttrs*>(_attr);
      auto &param = cvm::get<cvm::top::ConcatenateParam>(attr->parsed);
      DLTensor *output = args[--len];
      int32_t axis = param.axis;
      int32_t ndim = static_cast<int32_t>(input0->ndim);
      if(axis < 0) axis += ndim;

      std::vector<void*> input_data(len);
      std::vector<int> input_shape(len*ndim);
      std::vector<int32_t > axisSize(len);
      int64_t preSize = 0;
      for(int i = 0; i < len; i++){
        DLTensor *input = args[i];
        input_data[i] = input->data;
        //memcpy(&input_shape[i*ndim], input->shape, ndim * sizeof(int));
        for(int j = 0; j < ndim; j++){
          input_shape[i*ndim + j] = input->shape[j];
        }
        axisSize[i] = preSize;
        preSize += input->shape[axis];
      }
      void *out_data = output->data;
      opencl_concatenate(input_data.data(), input_shape.data(), len, ndim, out_data, output->shape, axis, axisSize.data());
  });

CVM_REGISTER_GLOBAL("cvm.runtime.opencl.sum")
  .set_body([](CVMArgs args, CVMRetValue *ret){
      DLTensor *dlx = args[0];
      DLTensor *y = args[1];
      void *y_data = y->data;
      void * x = dlx->data;
      void* _attr = args[2];
      auto *attr = static_cast<cvm::NodeAttrs*>(_attr);
      auto &param = cvm::get<cvm::top::ReduceParam>(attr->parsed);
      TShape axis = param.axis;
      int64_t *axis_data = axis.begin();
      for(size_t i = 0; i < axis.ndim(); i++){
        if(axis_data[i] < 0) axis_data[i] += dlx->ndim;
      }
      //bool keepdims = param.keepdims;
      std::vector<int> raxis;
      bool exclude = param.exclude;
      if(!exclude){
        for(size_t i = 0; i < axis.ndim(); i++){
          raxis.push_back(axis[i]);
        }
      }else{
        raxis.resize(dlx->ndim - axis.ndim());
        for(int32_t i = 0, k = 0; i < dlx->ndim; i++){
          bool flag = false;
          for(uint32_t j = 0; j < axis.ndim(); j++){
            if(axis_data[j] == i) {
              flag = true;
              break;
            }
          }
          if(!flag){
            raxis[k++] = i;
          }
        }
      }

      if(exclude && raxis.size() == 0){
        opencl_flatten(x, y_data, getSize(dlx));
      }
      else if(raxis.size() == 0){
        opencl_reduce(x, y_data, getSize(dlx), getSize(y),
            dlx->shape, y->shape, NULL, NULL,
            NULL, 0, dlx->ndim, y->ndim, raxis.size(), REDUCE_SUM);
      }else{
        std::vector<int32_t> realAxis(raxis.size());
        std::shared_ptr<int32_t> flag(new int32_t[dlx->ndim]);
        std::memset(flag.get(), 0, sizeof(int32_t)*dlx->ndim);
        for(uint32_t i = 0; i < raxis.size(); i++){
          int32_t val = raxis[i];
          realAxis[i] = val;
          flag.get()[val] = 1;
        }
        std::sort(realAxis.begin(), realAxis.end());
        realAxis.resize(std::unique(realAxis.begin(), realAxis.end()) - realAxis.begin());

        int axis_size = 1;
        for(uint32_t i = 0; i < realAxis.size(); i++){
          axis_size *= dlx->shape[realAxis[i]];
        }
        std::vector<int> every_xdim_size(dlx->ndim, 1);
        for(int i = dlx->ndim-2; i >= 0; i--){
          every_xdim_size[i] = dlx->shape[i+1] * every_xdim_size[i+1];
        }

        int32_t yndim = y->ndim;
        std::vector<int64_t> yshape(y->ndim);
        for(int32_t i = 0; i < y->ndim; i++){
          yshape[i] = y->shape[i];
        }
        for(int32_t i = 0, j = 0; i < y->ndim; i++){
          if(y->shape[i] == 1) {
            yndim -= 1;
          }else{
            yshape[j++] = y->shape[i];
          }
        }
        opencl_reduce(x, y_data, getSize(dlx), getSize(y),
            dlx->shape, yshape.data(), realAxis.data(), flag.get(),
            every_xdim_size.data(), axis_size, dlx->ndim, yndim, raxis.size(), REDUCE_SUM);
      }
  });

CVM_REGISTER_GLOBAL("cvm.runtime.opencl.max")
  .set_body([](CVMArgs args, CVMRetValue *ret){
      DLTensor *dlx = args[0];
      DLTensor *y = args[1];
      void *y_data = y->data;
      void* x = dlx->data;
      void* _attr = args[2];
      auto *attr = static_cast<cvm::NodeAttrs*>(_attr);
      auto &param = cvm::get<cvm::top::ReduceParam>(attr->parsed);
      TShape axis = param.axis;
      int64_t *axis_data = axis.begin();
      for(size_t i = 0; i < axis.ndim(); i++){
      if(axis_data[i] < 0) axis_data[i] += dlx->ndim;
      }
      //bool keepdims = param.keepdims;
      std::vector<int> raxis;
      bool exclude = param.exclude;
      if(!exclude){
        for(size_t i = 0; i < axis.ndim(); i++){
        raxis.push_back(axis[i]);
        }
      }else{
        raxis.resize(dlx->ndim - axis.ndim());
        for(int i = 0, k = 0; i < dlx->ndim; i++){
          bool flag = false;
          for(size_t j = 0; j < axis.ndim(); j++){
            if(axis_data[j] == i) {
              flag = true;
              break;
            }
          }
          if(!flag){
            raxis[k++] = i;
          }
        }
      }

      if(exclude && raxis.size() == 0){
        opencl_flatten(x, y_data, getSize(dlx));
      }
      else if(raxis.size() == 0){
        opencl_reduce(x, y_data, getSize(dlx), getSize(y),
            dlx->shape, y->shape, NULL, NULL,
            NULL, 0, dlx->ndim, y->ndim, raxis.size(), REDUCE_MAX);
      }else{
        std::vector<int32_t> realAxis(raxis.size());
        std::shared_ptr<int32_t> flag(new int32_t[dlx->ndim]);
        std::memset(flag.get(), 0, sizeof(int32_t)*dlx->ndim);
        for(uint32_t i = 0; i < raxis.size(); i++){
          int32_t val = raxis[i];
          realAxis[i] = val;
          flag.get()[val] = 1;
        }
        std::sort(realAxis.begin(), realAxis.end());
        realAxis.resize(std::unique(realAxis.begin(), realAxis.end()) - realAxis.begin());

        int axis_size = 1;
        for(uint32_t i = 0; i < realAxis.size(); i++){
          axis_size *= dlx->shape[realAxis[i]];
        }
        std::vector<int> every_xdim_size(dlx->ndim, 1);
        for(int i = dlx->ndim-2; i >= 0; i--){
          every_xdim_size[i] = dlx->shape[i+1] * every_xdim_size[i+1];
        }

        int32_t yndim = y->ndim;
        std::vector<int64_t> yshape(y->ndim);
        for(int i = 0; i < y->ndim; i++){
          yshape[i] = y->shape[i];
        }
        for(int i = 0, j = 0; i < y->ndim; i++){
          if(y->shape[i] == 1) {
            yndim -= 1;
          }else{
            yshape[j++] = y->shape[i];
          }
        }
        opencl_reduce(x, y_data, getSize(dlx), getSize(y),
            dlx->shape, yshape.data(), realAxis.data(), flag.get(),
            every_xdim_size.data(), axis_size, dlx->ndim, yndim, raxis.size(), REDUCE_MAX);
      }
  });

CVM_REGISTER_GLOBAL("cvm.runtime.opencl.get_valid_counts")
  .set_body([](cvm::runtime::CVMArgs args, cvm::runtime::CVMRetValue *rv){
      DLTensor *x = args[0];
      DLTensor *valid_count = args[1];
      DLTensor *y = args[2];
      void* _attr = args[3];
      auto *attr = static_cast<cvm::NodeAttrs*>(_attr);
      auto &param = cvm::get<cvm::top::GetValidCountsParam>(attr->parsed);

      int32_t score_threshold = param.score_threshold;

      int32_t batchs = x->shape[0];
      int32_t n = x->shape[1];
      int32_t k = x->shape[2];

      void *x_data = static_cast<int32_t*>(x->data);
      void *valid_count_data = static_cast<int32_t*>(valid_count->data);
      void *y_data = static_cast<int32_t*>(y->data);
//      void *ext_space = static_cast<int32_t*>(args.ext_space->data);

      opencl_get_valid_count(x_data, y_data, valid_count_data, batchs, n, k, score_threshold);
  });

CVM_REGISTER_GLOBAL("cvm.runtime.opencl.non_max_suppression")
.set_body([](cvm::runtime::CVMArgs args, cvm::runtime::CVMRetValue *rv){
    DLTensor *dlx = args[0];
    DLTensor *dlv = args[1];
    DLTensor *dly = args[2];
    void *X = dlx->data;
    void *valid_count = dlv->data;
    void *Y = dly->data;
    void* _attr = args[3];
    auto *attr = static_cast<cvm::NodeAttrs*>(_attr);
    auto &params = cvm::get<cvm::top::NonMaximumSuppressionParam>(attr->parsed);
    //auto params = CVMArg2Attr<top::NonMaximumSuppressionParam>(args[3]);
    auto x_shape = dlx->shape;
    int32_t B = x_shape[0];
    int32_t N = x_shape[1];
    int32_t K = x_shape[2];
    opencl_non_max_suppression(X, valid_count, Y, B, N, K, params.force_suppress, params.iou_threshold, params.max_output_size, params.top_k);
  });

CVM_REGISTER_GLOBAL("cvm.runtime.opencl.repeat")
  .set_body([](CVMArgs args, CVMRetValue *ret){
      DLTensor *x = args[0];
      DLTensor *y = args[1];
      void *_attr = args[2];
      auto *attr = static_cast<cvm::NodeAttrs*>(_attr);
      auto &param = cvm::get<cvm::top::RepeatParam>(attr->parsed);
      void *x_data = x->data;
      void *y_data = y->data;
      int32_t axis = param.axis;
      int32_t repeat = param.repeats;
      int ndim = x->ndim;
      if(axis < 0) axis = axis + ndim;

      opencl_repeat(
          x_data, y_data, x->shape, y->shape, getSize(y), x->ndim, y->ndim, axis, repeat);
  });

CVM_REGISTER_GLOBAL("cvm.runtime.opencl.tile")
  .set_body([](CVMArgs args, CVMRetValue *ret){
      DLTensor *x = args[0];
      DLTensor *y = args[1];

      void *x_data = x->data;
      void *y_data = y->data;

      int32_t yndim = y->ndim;
      int32_t xndim = x->ndim;

      opencl_tile(x_data, y_data, getSize(y), yndim, xndim, x->shape, y->shape);
  });

CVM_REGISTER_GLOBAL("cvm.runtime.opencl.transpose")
  .set_body([](CVMArgs args, CVMRetValue *ret){
      DLTensor *x = args[0];
      DLTensor *y = args[1];
      void *_attr = args[2];
      auto *attr = static_cast<cvm::NodeAttrs*>(_attr);
      auto &param = cvm::get<cvm::top::TransposeParam>(attr->parsed);

      TShape axes = param.axes;
      int64_t *axes_data = axes.begin();
      for(uint32_t i = 0; i < axes.ndim(); i++){
        if(axes_data[i] < 0) axes_data[i] += x->ndim;
      }

      void *x_data = x->data;
      void *y_data = y->data;
      int ndim = y->ndim;
      opencl_transpose(x_data, axes_data, y_data, x->shape, y->shape, ndim, getSize(y), axes.ndim());
  });

CVM_REGISTER_GLOBAL("cvm.runtime.opencl.strided_slice")
  .set_body([](CVMArgs args, CVMRetValue *ret){
      DLTensor *x = args[0];
      DLTensor *y = args[1];
      void *_attr = args[2];
      auto *attr = static_cast<cvm::NodeAttrs*>(_attr);
      auto &param = cvm::get<cvm::top::StridedSliceParam>(attr->parsed);

      void *x_data = x->data;
      void *y_data = y->data;
      TShape begin = param.begin;
      TShape end = param.end;
      TShape stride = param.stride;
      //int ndim = y->ndim;
      int32_t num_axis = x->ndim;
      int64_t *dshp = x->shape;
      std::vector<int64_t> begin_vec;
      std::copy(begin.begin(), begin.end(), std::back_inserter(begin_vec));
      for (dim_t i = begin_vec.size(); i < num_axis; ++i) {
        begin_vec.push_back(0);
      }

      std::vector<int64_t> stride_vec;
      std::copy(stride.begin(), stride.end(), std::back_inserter(stride_vec));
      for (dim_t i = stride_vec.size(); i < num_axis; ++i) {
        stride_vec.push_back(1);
      }

      for (size_t i = 0; i < begin_vec.size(); ++i) {
        int64_t begin_range = stride_vec[i] < 0 ? -1 : 0;
        int64_t end_range = stride_vec[i] < 0 ? dshp[i] -1 : dshp[i];
        int64_t begin = begin_vec[i];
        if (begin < 0) begin += dshp[i];
        begin_vec[i]= std::min(std::max(begin, begin_range), end_range);
      }

      opencl_stride_slice(x_data, y_data, begin_vec.data(), begin.ndim(), stride_vec.data(),
          x->shape, y->shape, stride.ndim(), y->ndim, getSize(y), x->ndim);
      //int error_code = NON_ERROR;
      //const char *errorStr = cuda_stride_slice(x_data, y_data, begin_vec.data(), begin.ndim(), stride_vec.data(),
      //    x->shape, y->shape, stride.ndim(), y->ndim, getSize(y), x->ndim, error_code);
      //deal_error(error_code, errorStr);
  });

CVM_REGISTER_GLOBAL("cvm.runtime.opencl.slice_like")
  .set_body([](CVMArgs args, CVMRetValue *ret){
      DLTensor *x = args[0];
      //DLTensor *shape = args[1];
      DLTensor *y = args[2];
      void* _attr = args[3];
      auto *attr = static_cast<cvm::NodeAttrs*>(_attr);
      auto &param = cvm::get<cvm::top::SliceLikeParam>(attr->parsed);
      Tuple<int> axis = param.axis;
      // int *axis_data = axis.begin();

      void *x_data = x->data;
      void *y_data = y->data;
      int ndim = x->ndim;

      opencl_slice_like(x_data, y_data, x->shape, y->shape, getSize(y), ndim);
  });

CVM_REGISTER_GLOBAL("cvm.runtime.opencl.take")
  .set_body([](cvm::runtime::CVMArgs args, cvm::runtime::CVMRetValue *rv){
      DLTensor *x = args[0];
      DLTensor *indices = args[1];
      DLTensor *y = args[2];
      void *_attr = args[3];
      auto *attr = static_cast<cvm::NodeAttrs*>(_attr);
      auto &param = cvm::get<cvm::top::TakeParam>(attr->parsed);

      void *x_data = x->data;
      void *indices_data = indices->data;
      void *y_data = y->data;

      if(param.axis.has_value()){
      int32_t axis = param.axis.value();
      if(axis < 0){
        axis += x->ndim;
      }
      opencl_take(x_data, indices_data, y_data, x->shape, y->shape,
          indices->shape, y->ndim, x->ndim, indices->ndim, getSize(y), axis);
      }else{
        opencl_take(x_data, indices_data, y_data, getSize(y), getSize(x));
      }
  });

CVM_REGISTER_GLOBAL("cvm.runtime.opencl.cvm_lut")
  .set_body([](cvm::runtime::CVMArgs args, cvm::runtime::CVMRetValue *rv){
      DLTensor *x = args[0];
      DLTensor *indices = args[1];
      DLTensor *y = args[2];

      void *x_data = x->data;
      void *indices_data = indices->data;
      void *y_data = y->data;
      //    take(indices, x, y);
      opencl_take(indices_data, x_data, y_data, getSize(y), getSize(x));
  });

CVM_REGISTER_GLOBAL("cvm.runtime.opencl.upsampling")
  .set_body([](CVMArgs args, CVMRetValue *ret){
      DLTensor *x = args[0];
      DLTensor *y = args[1];

      void *_attr = args[2];
      auto *attr = static_cast<cvm::NodeAttrs*>(_attr);
      auto &param = cvm::get<cvm::top::UpSamplingParam>(attr->parsed);
      uint32_t scale = {(uint32_t)param.scale};
      uint32_t h = x->shape[2], w = x->shape[3];
      uint32_t oh = y->shape[2], ow = y->shape[3];
      uint32_t n_batch = x->shape[0], n_channels = x->shape[1];

      auto x_data = x->data;
      auto y_data = y->data;

      opencl_upsampling_nearest(x_data, y_data, scale, h, w, oh, ow, n_batch, n_channels);
  });

CVM_REGISTER_GLOBAL("cvm.runtime.opencl.squeeze")
  .set_body([](CVMArgs args, CVMRetValue *ret){
      DLTensor *ishape = args[0];
      DLTensor *oshape = args[1];
      void *ishape_data = ishape->data;
      void *oshape_data = oshape->data;
      if(ishape_data == oshape_data){
        return;
      }
      opencl_flatten(ishape_data, oshape_data, getSize(ishape));
});

CVM_REGISTER_GLOBAL("cvm.runtime.opencl.where")
  .set_body([](CVMArgs args, CVMRetValue *ret){
    DLTensor *condition = args[0];
    DLTensor *x = args[1];
    DLTensor *y = args[2];
    DLTensor *result = args[3];

    void *x_data = x->data;
    void *y_data = y->data;
    void *condition_data = condition->data;
    void *result_data = result->data;

    uint64_t size = 1;
    for(int32_t i = 1; i < result->ndim; i++){
      size *= result->shape[i];
    }
    
    bool same_shape = x->ndim == condition->ndim;
    uint64_t n = same_shape ? getSize(result) : size;

    opencl_where(x_data, y_data, condition_data, result_data, same_shape, n, result->shape[0]);
  });

CVM_REGISTER_GLOBAL("cvm.runtime.opencl.expand_dims")
  .set_body([](CVMArgs args, CVMRetValue *ret){
      DLTensor *ishape = args[0];
      DLTensor *oshape = args[1];
      //void *_attr = args[2];
      //auto *attr = static_cast<cvm::NodeAttrs*>(_attr);
      //auto &param = cvm::get<cvm::top::ExpandDimsParam>(attr->parsed);

      //int32_t axis = param.axis;
      void *ishape_data = ishape->data;
      void *oshape_data = oshape->data;

      opencl_flatten(ishape_data, oshape_data, getSize(oshape));
  });

CVM_REGISTER_GLOBAL("cvm.runtime.opencl.negative")
  .set_body([](CVMArgs args, CVMRetValue *ret){
      DLTensor *x = args[0];
      DLTensor *y = args[1];
      void *x_data = x->data;
      void *y_data = y->data;

      opencl_negative(x_data, y_data, getSize(y));
  });

CVM_REGISTER_GLOBAL("cvm.runtime.opencl.cvm_precision")
  .set_body([](CVMArgs args, CVMRetValue *ret){
      DLTensor *dlx = args[0];
      DLTensor *y = args[1];
      void *y_data = y->data;
      void *x = dlx->data;
      opencl_log(x, y_data, getSize(y));
  });

CVM_REGISTER_GLOBAL("cvm.runtime.opencl.elemwise_add")
  .set_body([](cvm::runtime::CVMArgs args, cvm::runtime::CVMRetValue *rv){
      DLTensor *a = args[0];
      DLTensor *b = args[1];
      DLTensor *c = args[2];
      void *a_data = a->data;
      void *b_data = b->data;
      void *c_data = c->data;
      uint64_t n = getSize(a);
      opencl_elemwise(a_data, b_data, c_data, n, 0);

});

CVM_REGISTER_GLOBAL("cvm.runtime.opencl.elemwise_sub")
  .set_body([](CVMArgs args, CVMRetValue *ret){
      DLTensor *a = args[0];
      DLTensor *b = args[1];
      DLTensor *c = args[2];
      void *a_data = a->data;
      void *b_data = b->data;
      void *c_data = c->data;
      uint64_t n = getSize(a);
      opencl_elemwise(a_data, b_data, c_data, n, 1);
  });

CVM_REGISTER_GLOBAL("cvm.runtime.opencl.abs")
  .set_body([](CVMArgs args, CVMRetValue *ret){
      DLTensor *dlx = args[0];
      DLTensor *y = args[1];
      opencl_abs(dlx->data, y->data, getSize(dlx));
  });
}
}
