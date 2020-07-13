#include "ops.h"

namespace cvm {
namespace runtime {

inline std::vector<int64_t> GetRealAxis(TShape& axis, bool exclude, DLTensor *x){
  // axis has been checked, it must be in range [-N, N)
  // CODE STYLE: indent using 2 or 4 spaces???
  for(size_t i = 0; i < axis.ndim(); i++){
    if(axis[i] < 0) axis[i] += x->ndim;
  }
  std::vector<int64_t> raxis;
  if(!exclude){
    for(size_t i = 0; i < axis.ndim(); i++){
      raxis.push_back(axis[i]);
    }
  }else{
    raxis.resize(x->ndim - axis.ndim());
    std::vector<bool> flags(x->ndim, false);
    for (int i = 0; i < axis.ndim(); i++) {
      flags[axis[i]] = true;
    }
    for (int i = 0, k = 0; i < flags.size(); i++) {
      if (!flags[i]) {
        raxis[k++] = i;
      }
    }
  }
  return raxis;
}

typedef std::function<void(int32_t&, int32_t)> reduce_func;
static void Reduce(DLTensor *x, 
                   DLTensor *y, 
                   TShape& axis, 
                   bool exclude, reduce_func const &f){
  int32_t *x_data = static_cast<int32_t*>(x->data);
  int32_t *y_data = static_cast<int32_t*>(y->data);

  std::vector<int64_t> realAxis = GetRealAxis(axis, exclude, x);

  if(exclude && realAxis.size() == 0){  // do nothing
    memcpy(y_data, x_data, getSize(x) * sizeof(int32_t));
  } else if (realAxis.size() == 0) {  // do reduce to the whole data
    int32_t tmp = 0;
    for(uint64_t i = 0; i < getSize(x); i++){
      f(tmp, x_data[i]);
    }
    y_data[0] = tmp;
  } else {
    // if flags[dims] == true, dims are to be reduced
    std::vector<bool> flag(x->ndim, false);
    for(uint32_t i = 0; i < realAxis.size(); i++){
      int32_t val = realAxis[i];
      flag[val] = true;
    }
    std::sort(realAxis.begin(), realAxis.end());

    uint64_t axis_size = 1;
    TShape reducedAxShape(realAxis.size()), reducedYShape(x->ndim - realAxis.size());
    TShape xShape(x->ndim);
    for(uint i = 0, ax = 0, yi = 0; i < x->ndim; i++){
      if (flag[i]) {
        axis_size *= x->shape[i];
        reducedAxShape[ax++] = x->shape[i];
      } else {
        reducedYShape[yi++] = x->shape[i];
      }
      xShape[i] = x->shape[i];
    }

    /* For example:
     * xshp : (n1, n2, n3, n4) -> yshp(n1, n4)
     * find x indexes reduce to yi with two steps:
     *  1. x start index, xvar(n1, 0, 0, n4)
     *  2. foreach reduce axis dimension(n2, n3), 
     *      get x possible indexes and add value to result.
     **/
    for(uint64_t i = 0; i < getSize(y); i++){
      // for each y to be filled, find related xs and calculate.
      // the index for each dim of an x:
      // for a dim to be reduced, index of this dim differs from each x.
      // otherwise, it is fixed with y during the traverse.
      TShape yIndex = VectorIndex(reducedYShape, i);
      TShape xIndex(x->ndim);
      for (uint j = 0, yi = 0; j < x->ndim; j++) {
        xIndex[j] = flag[j] ? 0 : yIndex[yi++];
      }
      // the first x is tmp.
      int32_t tmp = x_data[ScalarIndex(xShape, xIndex)];
      for (uint64_t xi = 1; xi < axis_size; xi++) {
        TShape reducedIndex = VectorIndex(reducedAxShape, xi);
        for (uint j = 0, yi = 0, ri = 0; j < xIndex.ndim(); j++) {
          xIndex[j] = flag[j] ? reducedIndex[ri++] : yIndex[yi++];
        }
        f(tmp, x_data[ScalarIndex(xShape, xIndex)]);
      }
      y_data[i] = tmp;
    }
  }
}

CVM_REGISTER_GLOBAL("cvm.runtime.formal.sum")
  .set_body([](CVMArgs args, CVMRetValue *ret)
      {
      DLTensor *x = args[0];
      DLTensor *y = args[1];
      void *_attr = args[2];
      auto *attr = static_cast<cvm::NodeAttrs*>(_attr);
      auto &param = cvm::get<cvm::top::ReduceParam>(attr->parsed);
      TShape axis = param.axis;
      bool exclude = param.exclude;

      reduce_func f = [](int32_t& tmp, int32_t value)->void {
        tmp += value;
      };

      Reduce(x, y, axis, exclude, f);
      print_to_file(y, "sum.txt");
      });

CVM_REGISTER_GLOBAL("cvm.runtime.formal.max")
  .set_body([](CVMArgs args, CVMRetValue *ret)
      {
      DLTensor *x = args[0];
      DLTensor *y = args[1];
      void *_attr = args[2];
      auto *attr = static_cast<cvm::NodeAttrs*>(_attr);
      auto &param = cvm::get<cvm::top::ReduceParam>(attr->parsed);
      TShape axis = param.axis;
      bool exclude = param.exclude;

      reduce_func f = [](int32_t& tmp, int32_t value)->void {
        if(tmp < value) tmp = value;
      };

      Reduce(x, y, axis, exclude, f);
      });

}
}
