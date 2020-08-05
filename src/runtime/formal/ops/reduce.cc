// #include "ops.h"

#include <cvm/runtime/forward.h>
#include <cvm/runtime/registry.h>
#include <cvm/top/tensor.h>
#include <cvm/top/nn.h>

namespace cvm {
namespace runtime {

inline std::vector<int64_t> GetRealAxis(TShape& axis, bool exclude, uint32_t ndim){
  // axis has been checked, it must be in range [-N, N)
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
  auto X_data = CVMArg2Data<int32_t>(args[0]);
  auto Y_data = CVMArg2Data<int32_t>(args[1]);

  auto const& X_shape = CVMArgShape(args[0]);
  auto& param = CVMArg2Attr<top::ReduceParam>(args[2]);

  auto realAxis = GetRealAxis(param.axis, param.exclude, X_shape.ndim());

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
    // find the shape of sub-space to be reduced.
    // find the true Y shape after reducing
    // - since CVMArgShape(args[1]) will be incorrect when `keepdims` is true
    TShape reducedAxShape(realAxis.size()), reducYShape(X_shape.ndim() - realAxis.size());
    for (uint i = 0, ax = 0, yi = 0; i < X_shape.ndim(); i++) {
      if (flag[i]) {
        reducedAxShape[ax++] = X_shape[i];
      } else {
        reducYShape[yi++] = X_shape[i];
      }
    }

    /* For example:
     * xshp : (n1, n2, n3, n4) -> yshp(n1, n4)
     * find x indexes reduce to yi with two steps:
     *  1. x start index, xvar(n1, 0, 0, n4)
     *  2. foreach reduce axis dimension(n2, n3), 
     *      get x possible indexes and add value to result.
     **/

    for (Indices yIdx(reducYShape); !yIdx.End(); yIdx++) {
      // for each y to be filled, find related xs and calculate.
      // the index for each dim of an x:
      // for a dim to be reduced, index of this dim differs from each x.
      // otherwise, it is fixed with y during the traverse.
      Indices xIdx(X_shape);
      for (uint j = 0, yi = 0; j < X_shape.ndim(); j++) {
        xIdx[j] = flag[j] ? 0 : yIdx[yi++];
      }
      int32_t tmp = X_data[xIdx.Index()];
      // the first x is tmp. we start from the second in the reduced space, so
      // reducIdx++
      Indices reducIdx(reducedAxShape);
      reducIdx++;
      for (; !reducIdx.End(); reducIdx++) {
        for (uint j = 0, yj = 0, rj = 0; j < X_shape.ndim(); j++) {
          xIdx[j] = flag[j] ? reducIdx[rj++] : yIdx[yj++];
        }
        f(tmp, X_data[xIdx.Index()]);
      }
      Y_data[yIdx.Index()] = tmp;
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
