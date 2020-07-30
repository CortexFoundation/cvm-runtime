// #include "ops.h"

#include <cvm/runtime/forward.h>
#include <cvm/runtime/registry.h>
#include <cvm/top/tensor.h>
#include <cvm/top/nn.h>

namespace cvm {
namespace runtime {

inline std::vector<int64_t> GetRealAxis(TShape& axis, bool exclude, uint32_t ndim){
  // axis has been checked, it must be in range [-N, N)
  // CODE STYLE: indent using 2 or 4 spaces???
  for(size_t i = 0; i < axis.ndim(); i++){
    if(axis[i] < 0) axis[i] += ndim;
  }
  std::vector<int64_t> raxis;
  if(!exclude){
    for(size_t i = 0; i < axis.ndim(); i++){
      raxis.push_back(axis[i]);
    }
  }else{
    raxis.resize(ndim - axis.ndim());
    std::vector<bool> flags(ndim, false);
    for (uint32_t i = 0; i < axis.ndim(); i++) {
      flags[axis[i]] = true;
    }
    for (size_t i = 0, k = 0; i < flags.size(); i++) {
      if (!flags[i]) {
        raxis[k++] = i;
      }
    }
  }
  return raxis;
}

typedef std::function<void(int32_t&, int32_t)> reduce_func;
static void Reduce(CVMArgs args, reduce_func const& f) {
// static void Reduce(DLTensor *x,
                   // DLTensor *y,
                   // TShape& axis,
                   // bool exclude, reduce_func const &f){
  // int32_t *x_data = static_cast<int32_t*>(x->data);
  // int32_t *y_data = static_cast<int32_t*>(y->data);
  auto X_data = CVMArg2Data<int32_t>(args[0]);
  auto Y_data = CVMArg2Data<int32_t>(args[1]);

  auto const& X_shape = CVMArgShape(args[0]);
  auto const& Y_shape = CVMArgShape(args[1]);
  auto& param = CVMArg2Attr<top::ReduceParam>(args[2]);

  auto realAxis = GetRealAxis(
      param.axis, param.exclude, X_shape.ndim());

  if(param.exclude && realAxis.size() == 0){  // do nothing
    if (X_data == Y_data) return ;
    memcpy(Y_data, X_data, X_shape.Size() * sizeof(int32_t));
  } else if (realAxis.size() == 0) {  // do reduce to the whole data
    int32_t tmp = 0;
    for (size_t i = 0; i < X_shape.Size(); ++i) {
      f(tmp, X_data[i]);
    }
    Y_data[0] = tmp;
  } else {
    // if flags[dims] == true, dims are to be reduced
    std::vector<bool> flag(X_shape.ndim(), false);
    for(uint32_t i = 0; i < realAxis.size(); i++){
      int32_t val = realAxis[i];
      flag[val] = true;
    }
    std::sort(realAxis.begin(), realAxis.end());

    uint64_t axis_size = 1;
//<<<<<<< HEAD
//    TShape reducedAxShape(realAxis.size()), reducedYShape(x->ndim - realAxis.size());
//    TShape xShape(x->ndim);
//    for(uint i = 0, ax = 0, yi = 0; i < x->ndim; i++){
//      if (flag[i]) {
//        axis_size *= x->shape[i];
//        reducedAxShape[ax++] = x->shape[i];
//      } else {
//        reducedYShape[yi++] = x->shape[i];
//=======
    for(uint32_t i = 0; i < realAxis.size(); i++){
      axis_size *= X_shape[realAxis[i]];
    }
    // every_xdim_size is used to calculate array polynomial
    std::vector<uint64_t> every_xdim_size(X_shape.ndim(), 1);
    for(int i = X_shape.ndim()-2; i >= 0; i--){
      every_xdim_size[i] = X_shape[i+1] * every_xdim_size[i+1];
    }
    // foreach yshape, remove the dimension equals 1
    // special case: input shape is (1,), considered.
    int32_t yndim = Y_shape.ndim();
    std::vector<int64_t> yshape(Y_shape.ndim(), 1);
    for(uint32_t i = 0, j = 0; i < Y_shape.ndim(); i++){
      if(Y_shape[i] == 1) {
        yndim -= 1;
      }else{
        yshape[j++] = Y_shape[i];
//>>>>>>> wlt
      }
//      xShape[i] = x->shape[i];
    }

    /* For example:
     * xshp : (n1, n2, n3, n4) -> yshp(n1, n4)
     * find x indexes reduce to yi with two steps:
     *  1. x start index, xvar(n1, 0, 0, n4)
     *  2. foreach reduce axis dimension(n2, n3), 
     *      get x possible indexes and add value to result.
     **/
//<<<<<<< HEAD
//    for(uint64_t i = 0; i < getSize(y); i++){
//      // for each y to be filled, find related xs and calculate.
//      // the index for each dim of an x:
//      // for a dim to be reduced, index of this dim differs from each x.
//      // otherwise, it is fixed with y during the traverse.
//      TShape yIndex = VectorIndex(reducedYShape, i);
//      TShape xIndex(x->ndim);
//      for (uint j = 0, yi = 0; j < x->ndim; j++) {
//        xIndex[j] = flag[j] ? 0 : yIndex[yi++];
//      }
//      // the first x is tmp.
//      int32_t tmp = x_data[ScalarIndex(xShape, xIndex)];
//      for (uint64_t xi = 1; xi < axis_size; xi++) {
//        TShape reducedIndex = VectorIndex(reducedAxShape, xi);
//        for (uint j = 0, yi = 0, ri = 0; j < xIndex.ndim(); j++) {
//          xIndex[j] = flag[j] ? reducedIndex[ri++] : yIndex[yi++];
//        }
//        f(tmp, x_data[ScalarIndex(xShape, xIndex)]);
//=======
    for(uint64_t i = 0; i < Y_shape.Size(); i++){
      // in_i will be the base index of X. for Y[a][b] = sum(or max) X[a][*][*][d]
      // in_i = d * 1 + a * n4*n3*n2
      // o_i is a middle variable used for calculating in_i
      uint64_t in_i = 0, o_i = i;
      for(int j = yndim-1, xj = X_shape.ndim()-1; j>=0; j--){
        uint64_t col = o_i % yshape[j];
        o_i /= yshape[j];
        while(xj >= 0 && flag[xj--]); // xj+1 is the dim that remains
        in_i += col * every_xdim_size[xj+1];
      }

      int32_t tmp = X_data[in_i];
      for(uint64_t xi = 1; xi < axis_size; xi++){
        uint64_t o_i = xi, tmp_in_i = 0;
        for(int j = realAxis.size() - 1; j>=0; j--){
          uint64_t col = o_i % X_shape[realAxis[j]];
          o_i /= X_shape[realAxis[j]];
          tmp_in_i += col * every_xdim_size[realAxis[j]];
        }
        f(tmp, X_data[in_i+tmp_in_i]);
//>>>>>>> wlt
      }
      Y_data[i] = tmp;
    }
  }
}

CVM_REGISTER_GLOBAL("cvm.runtime.formal.sum")
  .set_body([](CVMArgs args, CVMRetValue *ret)
{
  reduce_func f = [](int32_t& tmp, int32_t value)->void {
    tmp += value;
  };

  Reduce(args, f);
});

CVM_REGISTER_GLOBAL("cvm.runtime.formal.max")
  .set_body([](CVMArgs args, CVMRetValue *ret)
{
  reduce_func f = [](int32_t& tmp, int32_t value)->void {
    if(tmp < value) tmp = value;
  };

  Reduce(args, f);
});

}
}
