// #include "ops.h"
#include <cvm/runtime/forward.h>
#include <cvm/runtime/registry.h>
#include <cvm/top/tensor.h>
#include <cvm/top/nn.h>

namespace cvm {
namespace runtime {


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

  auto &param = CVMArg2Attr<top::TransposeParam>(args[2]);

  TShape const& X_shape = CVMArgShape(args[0]);
  TShape const& Y_shape = CVMArgShape(args[1]);

  int32_t ndim = Y_shape.ndim();
  std::vector<int32_t> axes(ndim);
  for(int32_t i = 0; i < ndim; i++){
      if(param.axes.ndim() == 0){
        axes[i] = ndim - 1 - i;
      }else{
        int32_t axis = param.axes[i];
        axes[i] = axis < 0 ? axis + ndim : axis;
      }
  }

  for (Indices Y_indices(Y_shape); !Y_indices.End(); ++Y_indices) {
    Indices X_indices(X_shape);
    for (uint32_t i = 0; i < Y_indices.ndim(); ++i) {
      X_indices.Ref(axes[i]) = Y_indices[i];
    }
    Y_data[Y_indices.Index()] = X_data[X_indices.Index()];
  }
});

CVM_REGISTER_GLOBAL("cvm.runtime.formal.strided_slice")
.set_body([](CVMArgs args, CVMRetValue *ret){
  int32_t* x_data = CVMArg2Data<int32_t>(args[0]);
  int32_t* y_data = CVMArg2Data<int32_t>(args[1]);
  auto& param = CVMArg2Attr<cvm::top::StridedSliceParam>(args[2]);
  TShape begin = param.begin;
  TShape end = param.end;
  TShape stride = param.stride;
  TShape const& xShape = CVMArgShape(args[0]);
  TShape const& yShape = CVMArgShape(args[1]);
  int num_axis = xShape.ndim();

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
    int64_t end_range = stride_vec[i] < 0 ? xShape[i] -1 : xShape[i];
    int64_t begin = begin_vec[i];
    if (begin < 0) begin += xShape[i];
    begin_vec[i]= std::min(std::max(begin, begin_range), end_range);
  }

  Indices xIdx(xShape);
  for (Indices yIdx(yShape); !yIdx.End(); yIdx++) {
    for (int i = 0; i < xIdx.ndim(); i++) {
      xIdx.Ref(i) = begin_vec[i] + stride_vec[i] * yIdx[i];
    }
    y_data[yIdx.Index()] = x_data[xIdx.Index()];
  }
});

CVM_REGISTER_GLOBAL("cvm.runtime.formal.slice_like")
.set_body([](CVMArgs args, CVMRetValue *ret){
  auto X_shape = CVMArgShape(args[0]);
  auto Y_shape = CVMArgShape(args[2]);
  auto X_data = CVMArg2Data<int32_t>(args[0]); 
  auto Y_data = CVMArg2Data<int32_t>(args[2]); 
  // d_k represent the coordinate index
  // Y[d_0, d_1,,,] = X[d_0, d_1,,,]
  Indices X_indices(X_shape), Y_indices(Y_shape);
  for (; !Y_indices.End(); ++Y_indices) {
    X_indices.CopyIndicesFrom(Y_indices);
    Y_data[Y_indices.Index()] = X_data[X_indices.Index()];
  }
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

  /**
  * takes the input data on this axis for every coordinates in other axes.
  *
  *
  *                   idxIdx[0, 1, ...................., indices_ndim - 1]
  *                        ||               /\
  *  yIdx:                 ||               ||
  *  0, 1, ..., axis-1, | axis, axis+1, ..., axis + ind.ndim - 1, | axis + ind.ndim, ..., yIdx.ndim
  *           ||           ||                                                    ||
  *           ||           ||               ++===================================++
  *           ||           ||               ||
  *  xIdx:    \/           \/               \/
  *  0, 1, ..., axis-1, | axis | axis+1, ..., xIdx.ndim
  */
  for (Indices yIdx(Y_shape), xIdx(X_shape), idxIdx(Indices_shape);
        !yIdx.End(); yIdx++) {
    // all elements in xIdx and idxIdx assigned to some other values.
    // no need to reset explictly.
    for (int j = 0; j < axis; j++) {
      xIdx.Ref(j) = yIdx[j];
    }
    for (int j = 0; j < indices_ndim; j++) {
      idxIdx.Ref(j) = yIdx[j + axis];
    }
    int32_t axisIndex = indices_data[idxIdx.Index()];
    xIdx.Ref(axis) = std::min(std::max(axisIndex, 0), (int32_t)X_shape[axis] - 1);
    for (int j = axis + 1; j < xndim; j++) {
      xIdx.Ref(j) = yIdx[j + indices_ndim - 1];
    }
    y_data[yIdx.Index()] = x_data[xIdx.Index()];
  }
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


CVM_REGISTER_GLOBAL("cvm.runtime.formal.where")
.set_body([](CVMArgs args, CVMRetValue *ret){

  int32_t *cond_data = CVMArg2Data<int32_t>(args[0]);
  int32_t *x_data = CVMArg2Data<int32_t>(args[1]);
  int32_t *y_data = CVMArg2Data<int32_t>(args[2]);
  int32_t *res_data = CVMArg2Data<int32_t>(args[3]);

  TShape condShape = CVMArgShape(args[0]);
  TShape xShape = CVMArgShape(args[1]);
  TShape resShape = CVMArgShape(args[3]);

  if(xShape.ndim() == condShape.ndim()){
    for (Indices resIdx(resShape); !resIdx.End(); resIdx++) {
      int idx = resIdx.Index();
      res_data[idx] = cond_data[idx] ? x_data[idx] : y_data[idx];
    }
  } else {
    uint64_t size = resShape.Size() / resShape[0];
    Indices resIdx(resShape);
    for (int i = 0; i < condShape[0]; i++) {
      resIdx.Ref(0) = i;
      int offset = resIdx.Index();
      memcpy(res_data + offset,
             (cond_data[i] ? x_data + offset : y_data + offset), size);
    }
  } 
  //print_to_file(result, "where.txt");
});

}
}



