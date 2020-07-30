// #include "ops.h"
#include <cvm/runtime/forward.h>
#include <cvm/runtime/registry.h>
#include <cvm/top/tensor.h>
#include <cvm/top/nn.h>

namespace cvm {
namespace runtime {

CVM_REGISTER_GLOBAL("cvm.runtime.formal.relu")
.set_body([](CVMArgs args, CVMRetValue* rv){
  auto X = args[0];
  auto Y = args[1];

  auto X_data = CVMArg2Data<int32_t>(X);
  auto Y_data = CVMArg2Data<int32_t>(Y);
  TShape const& shp = CVMArgShape(X);

  for (size_t i = 0; i < shp.Size(); ++i) {
    Y_data[i] = std::max(X_data[i], 0);
  }
});

CVM_REGISTER_GLOBAL("cvm.runtime.formal.dense")
.set_body([](CVMArgs args, CVMRetValue* rv) {

  // inputs X, W, B 
  // outpus Y 
  auto X = args[0];
  auto W = args[1];
  auto param = CVMArg2Attr<top::DenseParam>(args[args.size()-1]);
  auto Y = (param.use_bias) ? args[3] : args[2];

  // X.shape = (M, K)
  // W.shape = (N, K)
  // B.shape = (N,)
  // Y.shape = (M, N)
  auto X_shape = CVMArgShape(X);
  auto W_shape = CVMArgShape(W);
  auto Y_shape = CVMArgShape(Y);

  auto X_data = CVMArg2Data<int32_t>(X); 
  auto W_data = CVMArg2Data<int32_t>(W); 
  auto Y_data = CVMArg2Data<int32_t>(Y); 
  // Y = X * WT
  for (int64_t m = 0; m < Y_shape[0]; ++m) {
    // Y(m, n) = X(m, k) * WT(k, n) = X(m, k) * W(n, k)
    int32_t Y_offset = m * Y_shape[1], X_offset = m * X_shape[1];
    for (int64_t n = 0; n < Y_shape[1]; ++n) {
      int32_t sum = 0, W_offset = n * W_shape[1];
      for (int64_t k = 0; k < X_shape[1]; ++k) {
        sum += X_data[X_offset + k] * W_data[W_offset + k];
      }
      Y_data[Y_offset + n] = sum;
    }
  }
  // if B is not None, Y = X * WT + B
  if (param.use_bias) {
    auto B = args[2];
    auto B_data = CVMArg2Data<int32_t>(B); 
    for (int64_t m = 0; m < Y_shape[0]; ++m) {
      int32_t Y_offset = m * Y_shape[1];
      for (int64_t n = 0; n < Y_shape[1]; ++n) {
        Y_data[Y_offset + n] += B_data[n];
      }
    }
  }
});

void conv2d(
    int32_t *x_data, int32_t n_batch, int32_t in_channels, int32_t x_h, int32_t x_w,
    int32_t *w_data, int32_t filter_c, int32_t filter_h, int32_t filter_w,
    int32_t *y_data, int32_t out_channels, int32_t o_h, int32_t o_w,
    int32_t *b_data,
    int32_t padding[2], int32_t stride_h, int32_t stride_w, int32_t dilation_h, int32_t dilation_w){
  for(int32_t n = 0; n < n_batch; n++){
    for(int32_t oc = 0; oc < out_channels; oc++){
      for(int32_t oh = 0; oh < o_h; oh++){
        for(int32_t ow = 0; ow < o_w; ow++){
          int32_t sum = 0;
          for(int32_t ic = 0; ic < in_channels; ic++){
            for(int32_t fh = 0; fh < filter_h; fh++){
              for(int32_t fw = 0; fw < filter_w; fw++){
                int32_t ih = oh * stride_h + fh * dilation_h- padding[0];
                int32_t iw = ow * stride_w + fw * dilation_w- padding[1];
                if(ih < 0 || ih >= x_h || iw < 0 || iw >= x_w){
                  continue;
                }
                int32_t w_index = oc * filter_c * filter_h * filter_w + ic * filter_h * filter_w + fh * filter_w + fw;
                int32_t x_index = n * in_channels * x_h * x_w + ic * x_h * x_w + ih * x_w + iw;
                sum += w_data[w_index] * x_data[x_index];
              }
            }
          }
          int32_t y_index = n * out_channels * o_h * o_w + oc * o_h * o_w + oh * o_w + ow;
          y_data[y_index] = sum + (b_data != nullptr ? b_data[oc] : 0);
        }
      }
    }
  }
}

static void groupwise_conv2d(
   int32_t *x_data, int32_t n_batch, int32_t in_channels, int32_t x_h, int32_t x_w,
   int32_t *w_data, int32_t filter_c, int32_t filter_h, int32_t filter_w,
   int32_t *y_data, int32_t out_channels, int32_t o_h, int32_t o_w,
   int32_t *b_data,
   int32_t padding[2], int32_t stride_h, int32_t stride_w, int32_t dilation_h, int32_t dilation_w,
   int32_t groups=1){
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
                int32_t th = oh * stride_h + fh*dilation_h - padding[0];
                int32_t tw = ow * stride_w + fw*dilation_w - padding[1];
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

CVM_REGISTER_GLOBAL("cvm.runtime.formal.conv2d")
    .set_body([](CVMArgs args, CVMRetValue* rv)
{
  DLTensor *x = args[0];
  DLTensor *w = args[1];
  DLTensor *b = nullptr; //args[2];
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
  int dilation[2] = {(int)param.dilation[0], (int)param.dilation[1]};
  // int kernel_size[2] = {(int)param.kernel_size[0], (int)param.kernel_size[1]};
  int padding[2] = {(int)param.padding[0], (int)param.padding[1]};
  int strides[2] = {(int)param.strides[0], (int)param.strides[1]};

  int stride_h = strides[0];
  int stride_w = strides[1];
  //int dilation_h = dilation[0];
  //int dilation_w = dilation[1];

  int32_t* x_data = (int32_t*)x->data;
  int32_t* w_data = (int32_t*)w->data;
  int32_t* y_data = (int32_t*)y->data;
  int32_t* b_data = b != nullptr ? (int32_t*)b->data : nullptr;

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

  if(groups > 1){
    groupwise_conv2d(
        x_data, n_batch, in_channels, x_h, x_w,
        w_data, filter_c, filter_h, filter_w,
        y_data, out_channels, o_h, o_w,
        b_data,
        padding, stride_h, stride_w, dilation[0], dilation[1],
        groups);
  } else {
    groupwise_conv2d(
        x_data, n_batch, in_channels, x_h, x_w,
        w_data, filter_c, filter_h, filter_w,
        y_data, out_channels, o_h, o_w,
        b_data,
        padding, stride_h, stride_w, dilation[0], dilation[1]);
  }
  print_to_file(y, "conv2d.txt");
});



/*
* strides (2, 2)
* pool_size [3, 3]
* ceil_mode False
* padding (1, 1)
*/
CVM_REGISTER_GLOBAL("cvm.runtime.formal.max_pool2d")
    .set_body([](CVMArgs args, CVMRetValue *ret)
{
  DLTensor *x = args[0];
  DLTensor *y = args[1];
  void *_attr = args[2];
  auto *attr = static_cast<cvm::NodeAttrs*>(_attr);
  auto &param = cvm::get<cvm::top::MaxPool2DParam>(attr->parsed);
  int padding[2] = {(int)param.padding[0], (int)param.padding[0]};
  if(param.padding.ndim() == 2){
    padding[1] = (int)param.padding[1];
  }

  int stride_h = param.strides[0];
  int stride_w = param.strides[1];

  int32_t* x_data = (int32_t*)x->data;
  int32_t* y_data = (int32_t*)y->data;

  int filter_h = param.pool_size[0];
  int filter_w = param.pool_size[1];

  int n_batch = static_cast<int>(x->shape[0]);
  int in_channels = static_cast<int>(x->shape[1]);
  int out_channels = in_channels;
  int x_h = static_cast<int>(x->shape[2]);
  int x_w = static_cast<int>(x->shape[3]);
  int o_h = static_cast<int>(y->shape[2]);
  int o_w = static_cast<int>(y->shape[3]);
#define GETX(n, c, h, w) x_data[(n) * in_channels * x_h * x_w + (c) * x_h * x_w + (h) * x_w + (w)]
#define GETY(n, c, h, w) y_data[(n) * out_channels * o_h * o_w + (c) * o_h * o_w + (h) * o_w + (w)]
  auto calc_func = [&](int n, int k, int p, int q) {
    const int32_t minV = int32_t(1) << 31;
    int32_t y_max = minV;
    for (int r = 0; r < filter_h; ++r) {
      for (int s = 0; s < filter_w; ++s) {
        int32_t tp = p * stride_h + r - padding[0];
        int32_t tq = q * stride_w + s - padding[1];
        int32_t x_tmp = minV; 
        if (0 <= tp && tp < x_h && 0 <= tq && tq < x_w)
          x_tmp = GETX(n, k, tp, tq);
        y_max = std::max(x_tmp, y_max);
      }
    }
    return y_max;
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
  print_to_file(y, "max_pool.txt");

});

CVM_REGISTER_GLOBAL("cvm.runtime.formal.cvm_precision")
.set_body([](CVMArgs args, CVMRetValue *ret){
    auto X_data = CVMArg2Data<int32_t>(args[0]);
    auto Y_data = CVMArg2Data<int32_t>(args[1]);

    for(size_t j = 0; j < CVMArgSize(args[0]); j++){
      // The case of x[j] == 0 is considered.
      // By right-shifting 1 bit at 1 time, how many bits x[j] takes is
      // how many times the right-shifting is performed until x[j] is 0
      int y = 1;
      int32_t absx = X_data[j] < 0 ? -X_data[j] : X_data[j];
      while (absx >> 1) {
        absx >>= 1;
        y++;
      }
      Y_data[j] = y;
    }
});

CVM_REGISTER_GLOBAL("cvm.runtime.formal.abs")
.set_body([](CVMArgs args, CVMRetValue *ret){
    auto X_data = CVMArg2Data<int32_t>(args[0]); 
    auto Y_data = CVMArg2Data<int32_t>(args[1]); 
    for (auto i = CVMShapeBegin(args[1]); i < CVMShapeEnd(args[1]); i++){
      Y_data[i] = std::abs(X_data[i]);
    }
});

// CVM_REGISTER_GLOBAL("cvm.runtime.formal.sqrt")
// .set_body([](CVMArgs args, CVMRetValue *ret){
    // DLTensor *x = args[0];
    // DLTensor *y = args[1];
    // int32_t *y_data = static_cast<int32_t*>(y->data);
    // int32_t* x_data = static_cast<int32_t*>(x->data);
    // for(uint64_t i = 0; i < getSize(x); i++){
      // y_data[i] = x_data[i] < 0 ? 0 : static_cast<int32_t>(std::sqrt(x_data[i]));
    // }
// });

CVM_REGISTER_GLOBAL("cvm.runtime.formal.concatenate")
.set_body([](CVMArgs args, CVMRetValue *ret){
    int M = args.num_args - 2; // I^0, I^1, ... I^M-1
    auto Y = CVMArg2Data<int32_t>(args[M]);
    auto params = CVMArg2Attr<top::ConcatenateParam>(args[M+1]);

    auto y_shape = CVMArgShape(args[M]);

    int32_t axis = params.axis;
    if(axis < 0) axis += y_shape.ndim();

    int64_t y_size = 1;
    for (int i = 0; i < axis; ++i) y_size *= y_shape[i];
    int32_t axis_batch = 1;
    for (size_t i = axis+1; i < y_shape.ndim(); ++i) axis_batch *= y_shape[i];

    // all axes after the axis we want to concatenate on can be copied as 
    // a batch at once thanks to cpp's row-major order standard.
    int64_t y_start_idx = 0;
    int64_t y_axis_batch = y_shape[axis] * axis_batch;
    for (int m = 0; m < M; ++m) {
      auto Ix = CVMArg2Data<int32_t>(args[m]);
      auto x_shape = CVMArgShape(args[m]);
      auto x_axis_batch = x_shape[axis] * axis_batch;

      for (int64_t y_iter = 0; y_iter < y_size; ++y_iter) {
        memcpy(Y+y_iter*y_axis_batch+y_start_idx,
               Ix+y_iter*x_axis_batch,
               x_axis_batch*sizeof(int32_t));
      }

      y_start_idx += x_axis_batch;
    }

});

CVM_REGISTER_GLOBAL("cvm.runtime.formal.repeat")
.set_body([](CVMArgs args, CVMRetValue *ret){

    // inputs: X
    // attr: repeats, axis
    // outputs: Y
    // X.shape = (n_0, n_1,,n_axis,, n_{N-1})
    // Y.shape = (n_0, n_1,,n_axis * repeats, n_{N-1})
    auto X = args[0];
    auto Y = args[1];
    auto X_shape = CVMArgShape(X);
    auto Y_shape = CVMArgShape(Y);
    auto X_data = CVMArg2Data<int32_t>(X);
    auto Y_data = CVMArg2Data<int32_t>(Y);
    auto param = CVMArg2Attr<cvm::top::RepeatParam>(args[args.size()-1]);
    int32_t axis = param.axis;
    int32_t repeats = param.repeats;
    if(axis < 0) axis = axis + X_shape.ndim();
    // y_k, x_k represent the coordinate index of Y.shape, X.shape, respectively
    std::vector<int64_t> Y_k(Y_shape.ndim(), 0), X_k(X_shape.ndim(), 0);
    for (auto i = CVMShapeBegin(Y); i < CVMShapeEnd(Y); i++){
      int index1 = Index2Number(X_shape, X_k);
      Y_data[i] = X_data[index1];
      IndexBaseShapeAddOne(Y_shape, Y_k);
      // Y[n_0, n_1,,n_axis,, n_{N-1}] = X[n_0, n_1,,n_axis/repeats,, n_{N-1}]
      X_k = Y_k;
      X_k[axis] /= repeats;
    }
});

CVM_REGISTER_GLOBAL("cvm.runtime.formal.negative")
.set_body([](CVMArgs args, CVMRetValue *ret){
    // inputs: x_data
    // outputs: y_data
    auto x_data = CVMArg2Data<int32_t>(args[0]); 
    auto y_data = CVMArg2Data<int32_t>(args[1]); 
    // y_data = -x_data
    for(size_t i = 0; i < CVMArgSize(args[0]); i++){
        y_data[i] = -x_data[i];
    }
});

CVM_REGISTER_GLOBAL("cvm.runtime.formal.tile")
.set_body([](CVMArgs args, CVMRetValue *ret){
    // inputs: X, reps
    // X.shape = (n_0, n_1,,, n_{N-1})
    // reps = (m_0, m_1,,, m_{M-1})
    // outputs: Y
    // Y.shape = max(X.shape, reps)
    // Y.shape.size = K
    auto X = args[0];
    auto Y = args[1];
    auto X_data = CVMArg2Data<int32_t>(X); 
    auto Y_data = CVMArg2Data<int32_t>(Y); 
    auto X_shape = CVMArgShape(X);
    auto Y_shape = CVMArgShape(Y);
    // X_k, Y_k represent the coordinate index of X.shape, Y.shape, respectively
    std::vector<int64_t> Y_k(Y_shape.ndim(), 0), X_k(X_shape.ndim(), 0);
    for (auto j = CVMShapeBegin(Y); j < CVMShapeEnd(Y); j++){
      // Y[k0, k1,,,k_{K-N}, k_{K-N+1},,,k_{K-1}] =
      // X[k_{K-N+0} mod n_0, k_{K-N+1} mod n_1,,, k_{K-N+N-1} mod n_{N-1}]
      for (uint32_t i = 0; i < X_shape.ndim(); i++){
        X_k[i] = Y_k[Y_shape.ndim() - X_shape.ndim() + i] % X_shape[i];
      }
      int index0 = Index2Number(X_shape, X_k);
      Y_data[j] = X_data[index0];
      IndexBaseShapeAddOne(Y_shape, Y_k);
    }
});

// CVM_REGISTER_GLOBAL("cvm.runtime.formal.pad")
// .set_body([](CVMArgs args, CVMRetValue *ret){
    // DLTensor *x = args[0];
    // DLTensor *y = args[1];
    // void *_attr = args[2];
    // auto *attr = static_cast<cvm::NodeAttrs*>(_attr);
    // auto &param = cvm::get<cvm::top::PadParam>(attr->parsed);

    // TShape pad_width = param.pad_width;
    // int pad_value = param.pad_value;
    // int32_t *x_data = static_cast<int32_t*>(x->data);
    // int32_t *y_data = static_cast<int32_t*>(y->data);

    // int32_t yndim = y->ndim;
    // for (uint64_t i = 0; i < getSize(y); i++) {
      // uint64_t o_i = i, in_i = 0, shapeSize = 1;
      // bool flag = true;
      // for (int j = xndim-1; j >= 0; j--) {
        // int col = o_i % y->shape[j];
        // int lower = pad_width[2*j], upper = x->shape[j]+pad_width[2*j];
        // if (col < lower || col >= upper) {
          // flag = false;
          // break;
        // }
        // o_i /= y->shape[j];
        // in_i += (col-lower) * shapeSize;
        // shapeSize *= x->shape[j];
      // }
      // y_data[i] = flag ? x_data[in_i] : pad_value;
    // }

    // print_to_file(y, "tile.txt");
// });

CVM_REGISTER_GLOBAL("cvm.runtime.formal.expand_dims")
.set_body([](CVMArgs args, CVMRetValue *ret)
{
  auto X_data = CVMArg2Data<int32_t>(args[0]);
  auto Y_data = CVMArg2Data<int32_t>(args[1]);

  if (X_data == Y_data) return ;
  memcpy(Y_data, X_data, CVMArgSize(args[0]) * sizeof(int32_t));
});

CVM_REGISTER_GLOBAL("cvm.runtime.formal.squeeze")
.set_body([](CVMArgs args, CVMRetValue *ret)
{
  auto X_data = CVMArg2Data<int32_t>(args[0]);
  auto Y_data = CVMArg2Data<int32_t>(args[1]);

  if (X_data == Y_data) return ;
  memcpy(Y_data, X_data, CVMArgSize(args[0]) * sizeof(int32_t));
});

CVM_REGISTER_GLOBAL("cvm.runtime.formal.transpose")
.set_body([](CVMArgs args, CVMRetValue *ret){
    auto X_data = CVMArg2Data<int32_t>(args[0]);
    auto Y_data = CVMArg2Data<int32_t>(args[1]);

//<<<<<<< HEAD
//    int32_t ndim = y->ndim;
//    // assert y->ndim == x->ndim
//    //TShape axes = param.axes;
//=======
    auto &param = CVMArg2Attr<top::TransposeParam>(args[2]);

    TShape const& X_shape = CVMArgShape(args[0]);
    TShape const& Y_shape = CVMArgShape(args[1]);

    int32_t ndim = Y_shape.ndim();
//>>>>>>> wlt
    std::vector<int32_t> axes(ndim);
    for(int32_t i = 0; i < ndim; i++){
        if(param.axes.ndim() == 0){
          axes[i] = ndim - 1 - i;
        }else{
          int32_t axis = param.axes[i];
          axes[i] = axis < 0 ? axis + ndim : axis;
        }
    }

//<<<<<<< HEAD
//    TShape xShape(x->shape, x->shape + x->ndim),
//        yShape(y->shape, y->shape + y->ndim);
//    for (uint32_t i = 0; i < getSize(x); i++) {
//      auto xIndex = VectorIndex(xShape, i);
//      TShape yIndex(y->ndim);
//      for (int j = yIndex.ndim() - 1; j >= 0; j--) {
//        yIndex[j] = xIndex[axes[j]];
//      }
//      y_data[ScalarIndex(yShape, yIndex)] = x_data[i];
//=======
    for (Indices Y_indices(Y_shape); !Y_indices.End(); ++Y_indices) {
      Indices X_indices(X_shape);
      for (uint32_t i = 0; i < Y_indices.ndim(); ++i) {
        X_indices[axes[i]] = Y_indices[i];
      }
      Y_data[Y_indices.Index()] = X_data[X_indices.Index()];
//>>>>>>> wlt
    }

    // for(uint64_t i = 0; i < Y_shape.Size(); i++) {
      // uint64_t o_i = i, in_i = 0;
      // for(int j = ndim - 1; j >= 0; j--){
        // uint64_t col = o_i % Y_shape[j];
        // o_i /= Y_shape[j];
        // int xi = 1;
        // for(int tx = ndim-1; tx > axes[j]; tx--){
          // xi *= X_shape[tx];
        // }
        // in_i += col * xi;
      // }
      // Y_data[i] = X_data[in_i];
    // }
});

CVM_REGISTER_GLOBAL("cvm.runtime.formal.strided_slice")
.set_body([](CVMArgs args, CVMRetValue *ret){
    DLTensor *x = args[0];
    DLTensor *y = args[1];
    void *_attr = args[2];
    auto *attr = static_cast<cvm::NodeAttrs*>(_attr);
    auto &param = cvm::get<cvm::top::StridedSliceParam>(attr->parsed);

    int32_t *x_data = static_cast<int32_t*>(x->data);
    int32_t *y_data = static_cast<int32_t*>(y->data);
    TShape begin = param.begin;
    TShape end = param.end;
    TShape stride = param.stride;
    int ndim = y->ndim;
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

//<<<<<<< HEAD
//    TShape xShape(x->shape, x->shape + x->ndim),
//        yShape(y->shape, y->shape + y->ndim);
//    for(uint64_t i = 0; i < getSize(y); i++){
//      TShape yVecIndex = VectorIndex(yShape, i);
//      TShape xVecIndex(ndim);
//      for (int j = 0; j < xVecIndex.ndim(); j++) {
//        xVecIndex[j] = begin_vec[j] + stride_vec[j] * yVecIndex[j];
//      }
//      y_data[i] = x_data[ScalarIndex(xShape, xVecIndex)];
//=======
    for(uint64_t i = 0; i < CVMArgSize(args[1]); i++){
        uint64_t o_i = i, in_i = 0, shapeSize = 1;
        for(int j = ndim-1; j >= 0; j--){
            uint64_t col = o_i % y->shape[j];
            o_i /= y->shape[j];
            int64_t tbegin = begin_vec[j];
            int64_t tstep = stride_vec[j];
            col = tbegin + col * tstep;
            in_i += col * shapeSize;
            shapeSize *= x->shape[j];
        }
        y_data[i] = x_data[in_i];
//>>>>>>> wlt
    }
    // print_to_file(y, "stride_slice.txt");
});

CVM_REGISTER_GLOBAL("cvm.runtime.formal.slice_like")
.set_body([](CVMArgs args, CVMRetValue *ret){
    auto X_shape = CVMArgShape(args[0]);
    auto Y_shape = CVMArgShape(args[2]);
    auto X_data = CVMArg2Data<int32_t>(args[0]); 
    auto Y_data = CVMArg2Data<int32_t>(args[2]); 
    // d_k represent the coordinate index
    // Y[d_0, d_1,,,] = X[d_0, d_1,,,]
    for (Indices Y_indices(Y_shape); !Y_indices.End(); ++Y_indices) {
      Indices X_indices(X_shape, Y_indices);
      Y_data[Y_indices.Index()] = X_data[X_indices.Index()];
    }

    // int K = Y_shape.ndim();
    // std::vector<int64_t> d_k(K, 0);
    // auto size = CVMArgSize(args[2]);
    // for(uint64_t i = 0; i < size; i++){
      // int index0 = Index2Number(X_shape, d_k);
      // Y_data[i] = X_data[index0];
      // IndexBaseShapeAddOne(Y_shape, d_k);
    // }
});

static void take(CVMArgValue x,
                 CVMArgValue indices,
                 CVMArgValue y){
    auto x_data = CVMArg2Data<int32_t>(x);
    auto indices_data = CVMArg2Data<int32_t>(indices);
    auto y_data = CVMArg2Data<int32_t>(y);
    uint64_t xs = CVMArgSize(x);

    for(size_t i = 0; i < CVMArgSize(y); i++){
        uint64_t in_i = std::min((uint64_t)std::max(indices_data[i], 0), xs-1);
        y_data[i] = x_data[in_i];
    }
}

//<<<<<<< HEAD
//static void take(DLTensor *x, 
//                 DLTensor *indices, 
//                 DLTensor *y, 
//                 const int32_t axis){
//  int32_t *x_data = static_cast<int32_t*>(x->data);
//  int32_t *indices_data = static_cast<int32_t*>(indices->data);
//  int32_t *y_data = static_cast<int32_t*>(y->data);
//
//  int32_t xndim = x->ndim;
//  int32_t indices_ndim = indices->ndim;
//  TShape xShape(x->shape, x->shape + x->ndim),
//      yShape(y->shape, y->shape + y->ndim),
//      idxShape(indices->shape, indices->shape + indices->ndim);
//
//  for (size_t i = 0; i < getSize(y); ++i) {
//    TShape yIndex = VectorIndex(yShape, i);
//    TShape indIndex(indices_ndim);
//    TShape xIndex(xndim);
//    for (uint j = 0; j < axis; j++) {
//      xIndex[j] = yIndex[j];
//    }
//    for (uint j = 0; j < indices_ndim; j++) {
//      indIndex[j] = yIndex[j + axis];
//    }
//    int32_t axisIndex = indices_data[ScalarIndex(idxShape, indIndex)];
//    xIndex[axis] = std::min(std::max(axisIndex, 0), (int32_t)x->shape[axis]-1);
//    for (uint j = axis + 1; j < xndim; j++) {
//      xIndex[j] = yIndex[j + indices_ndim - 1];
//=======
static void take(CVMArgValue x,
                 CVMArgValue indices,
                 CVMArgValue y,
                 int32_t axis){
    auto x_data = CVMArg2Data<int32_t>(x);
    auto indices_data = CVMArg2Data<int32_t>(indices);
    auto y_data = CVMArg2Data<int32_t>(y);

    auto X_shape = CVMArgShape(x);
    auto Indices_shape = CVMArgShape(indices);
    auto Y_shape = CVMArgShape(y);

    int32_t yndim = Y_shape.ndim();
    int32_t xndim = X_shape.ndim();
    int32_t indices_ndim = Indices_shape.ndim();

    if(axis < 0) { axis += xndim; }

    std::vector<size_t> x_shape_size(xndim, 1), indices_shape_size(indices_ndim, 1);
    for (int i = xndim-2; i >= 0; --i) {
      x_shape_size[i] = x_shape_size[i+1] * X_shape[i+1];
    }
    for (int i = indices_ndim-2; i >= 0; --i) {
      indices_shape_size[i] = indices_shape_size[i+1] * Indices_shape[i+1];
    }
    for (size_t i = 0; i < Y_shape.Size(); ++i) {
      size_t oi = i, xi = 0, idxi = 0;
      for(int j = yndim - 1; j>=0; --j){
        size_t col = oi % Y_shape[j];
        oi /= Y_shape[j];
        if (axis <= j && j < axis+indices_ndim) {
          idxi += col * indices_shape_size[j - axis];
        } else {
          int xidx = j < axis ? j : j - indices_ndim + 1;
          xi += col * x_shape_size[xidx];
        }

        if (axis == j) {
          int64_t idxx = std::min(std::max(indices_data[idxi], 0), 
              (int32_t)X_shape[j]-1);
          xi += idxx * x_shape_size[j];
        }
      }
      y_data[i] = x_data[xi];
//>>>>>>> wlt
    }
    //y_data[i] = x_data[ScalarIndex(xShape, xIndex)];
  //}
}

CVM_REGISTER_GLOBAL("cvm.runtime.formal.take")
.set_body([](cvm::runtime::CVMArgs args, cvm::runtime::CVMRetValue *rv){
    auto& param = CVMArg2Attr<top::TakeParam>(args[3]);
    if(!param.axis.has_value()){
      take(args[0], args[1], args[2]);
    }else{
      int32_t axis = param.axis.value();
      take(args[0], args[1], args[2], axis);
    }
    // print_to_file(y, "take.txt");
});

CVM_REGISTER_GLOBAL("cvm.runtime.formal.cvm_lut")
.set_body([](cvm::runtime::CVMArgs args, cvm::runtime::CVMRetValue *rv){
    take(args[1], args[0], args[2]);
});

CVM_REGISTER_GLOBAL("cvm.runtime.formal.upsampling")
.set_body([](CVMArgs args, CVMRetValue *ret)
{
  DLTensor *x = args[0];
  DLTensor *y = args[1];

  void *_attr = args[2];
  auto *attr = static_cast<cvm::NodeAttrs*>(_attr);
  auto &param = cvm::get<cvm::top::UpSamplingParam>(attr->parsed);

  uint32_t scale = {(uint32_t)param.scale};
  uint32_t h = x->shape[2], w = x->shape[3];
  uint32_t oh = y->shape[2], ow = y->shape[3];
  uint32_t n_batch = x->shape[0], n_channels = x->shape[1];

  auto x_data = static_cast<int32_t*>(x->data);
  auto y_data = static_cast<int32_t*>(y->data);

  for (uint32_t batch = 0; batch < n_batch; ++batch) {
    for (uint32_t c = 0; c< n_channels; ++c) {
      auto bc_y_data = y_data + batch * n_channels * oh * ow + c * oh * ow;
      auto bc_x_data = x_data + batch * n_channels *  h *  w + c *  h *  w;
      for(uint32_t y = 0; y < oh; ++y){
        for(uint32_t x = 0; x < ow; ++x){
            bc_y_data[y * ow + x] = bc_x_data[y/scale * w + x/scale];
        }
      }
    }
  }
});

CVM_REGISTER_GLOBAL("cvm.runtime.formal.where")
.set_body([](CVMArgs args, CVMRetValue *ret){
    DLTensor *condition = args[0];
    DLTensor *x = args[1];
    DLTensor *y = args[2];
    DLTensor *result = args[3];

    int32_t *x_data = static_cast<int32_t*>(x->data);
    int32_t *y_data = static_cast<int32_t*>(y->data);
    int32_t *condition_data = static_cast<int32_t*>(condition->data);
    int32_t *result_data = static_cast<int32_t*>(result->data);

    if(x->ndim == condition->ndim){
      for(uint64_t i = 0; i < CVMArgSize(args[3]); ++i){
        result_data[i] = condition_data[i] == 0 ? y_data[i] : x_data[i];
      }
    }else{
      uint64_t size = 1;
      for(int32_t i = 1; i < result->ndim; i++){
        size *= result->shape[i];
      }
      for(int32_t i = 0; i < result->shape[0]; ++i){
        memcpy(&result_data[i*size], (condition_data[i] == 0 ? &y_data[i*size] : &x_data[i*size]), size); 
      } 
    } 
    print_to_file(result, "where.txt");
});

}
}



