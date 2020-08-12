// #include "ops.h"
#include <cvm/runtime/forward.h>
#include <cvm/runtime/registry.h>
#include <cvm/top/tensor.h>
#include <cvm/top/nn.h>

namespace cvm {
namespace runtime {
  

typedef std::function<int32_t(int32_t a, int32_t b)> elemwise_func;

static void elemwise(const cvm::runtime::CVMArgValue& A, 
                     const cvm::runtime::CVMArgValue& B, 
                     const cvm::runtime::CVMArgValue& Y, 
                      elemwise_func const &f){
    // inputs: A, B
    // outputs: Y
    auto a = CVMArg2Data<int32_t>(A);
    auto b = CVMArg2Data<int32_t>(B);
    auto c = CVMArg2Data<int32_t>(Y);
    size_t end = CVMArgShape(A).Size();
    for(size_t i = 0; i < end; i++){
      c[i] = f(a[i], b[i]);
    }
}

CVM_REGISTER_GLOBAL("cvm.runtime.formal.cvm_precision")
.set_body([](CVMArgs args, CVMRetValue *ret) {
  auto X_data = CVMArg2Data<int32_t>(args[0]);
  auto Y_data = CVMArg2Data<int32_t>(args[1]);

  for (size_t j = 0; j < CVMArgSize(args[0]); j++) {
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
.set_body([](CVMArgs args, CVMRetValue *ret) {
  auto X_data = CVMArg2Data<int32_t>(args[0]);
  auto Y_data = CVMArg2Data<int32_t>(args[1]);
  int end = CVMArgShape(args[1]).Size();
  for (int i = 0; i < end; i++) {
    Y_data[i] = std::abs(X_data[i]);
  }
});


CVM_REGISTER_GLOBAL("cvm.runtime.formal.negative")
.set_body([](CVMArgs args, CVMRetValue *ret) {
  // inputs: x_data
  // outputs: y_data
  auto x_data = CVMArg2Data<int32_t>(args[0]);
  auto y_data = CVMArg2Data<int32_t>(args[1]);
  // y_data = -x_data
  size_t end = CVMArgShape(args[0]).Size();
  for (size_t i = 0; i < end; i++) {
    y_data[i] = -x_data[i];
  }
});


CVM_REGISTER_GLOBAL("cvm.runtime.formal.elemwise_add")
    .set_body([](CVMArgs args, CVMRetValue *ret)
{
  elemwise_func f = [](int32_t a, int32_t b) -> int32_t {
    return a + b;
  };

  elemwise(args[0], args[1], args[2], f);
});

CVM_REGISTER_GLOBAL("cvm.runtime.formal.elemwise_sub")
    .set_body([](CVMArgs args, CVMRetValue *ret)
{
  elemwise_func f = [](int32_t a, int32_t b) -> int32_t {
    return a - b;
  };

  elemwise(args[0], args[1], args[2], f);
});

CVM_REGISTER_GLOBAL("cvm.runtime.formal.clip")
.set_body([](CVMArgs args, CVMRetValue* rv){
   auto param = CVMArg2Attr<cvm::top::ClipParam>(args[2]);
   int64_t a_max = param.a_max;
   int64_t a_min = param.a_min;
   auto x_data = CVMArg2Data<int32_t>(args[0]); 
   auto y_data = CVMArg2Data<int32_t>(args[1]); 
   auto size = CVMArgSize(args[0]); 
   for (uint32_t i = 0; i < size; i++) {
      // y = a_max, x >= a_max
      if (x_data[i] >= a_max){
        y_data[i] = a_max;
        // y = a_min, x <= a_min
      } else if (x_data[i] <= a_min) {
        y_data[i] = a_min;
      } else {
        // y = x, a_min < x < a_max
        y_data[i] = x_data[i];
      }
   }
});


CVM_REGISTER_GLOBAL("cvm.runtime.formal.cvm_clip")
    .set_body([](CVMArgs args, CVMRetValue *ret)
{
  auto x_data = CVMArg2Data<int32_t>(args[0]); 
  auto y_data = CVMArg2Data<int32_t>(args[1]); 
  auto param = CVMArg2Attr<cvm::top::CVMClipParam>(args[2]);
  int64_t precision = param.precision;
  // alpha = 2^(precision-1) - 1
  int64_t alpha =  (((int64_t)1 << (precision-1))-1);
  // a_min = -alhpa
  // a_max = alpha
  int64_t a_min = -alpha;
  int64_t a_max = -a_min;
  // Y = clip(X, -alpha, alpha)
  auto size = CVMArgSize(args[0]); 
  for (uint32_t i = 0; i < size; i++) {
      // y = a_max, x >= a_max
      if (x_data[i] >= a_max){
        y_data[i] = a_max;
        // y = a_min, x <= a_min
      } else if (x_data[i] <= a_min) {
        y_data[i] = a_min;
      } else {
        // y = x, a_min < x < a_max
        y_data[i] = x_data[i];
      }
  }
}
);

CVM_REGISTER_GLOBAL("cvm.runtime.formal.cvm_right_shift")
.set_body([](CVMArgs args, CVMRetValue *ret){
    auto x_data = CVMArg2Data<int32_t>(args[0]); 
    auto y_data = CVMArg2Data<int32_t>(args[1]); 
    auto params = CVMArg2Attr<cvm::top::CVMRightShiftParam>(args[2]);
    int32_t precision = params.precision;
    // alpha = 2^(precision-1) - 1
    int32_t alpha =  (((int64_t)1 << (precision-1))-1);
    int32_t a_min = -alpha;
    int32_t a_max = alpha;
    auto size = CVMArgSize(args[0]);
    // T = floor((floor(X >> (shift_bit - 1)) + 1) >> 1)
    // Y = clip(T, -alpha, alpha)
    for (uint32_t i = 0; i < size; i++) {
      int32_t T = ((x_data[i] >> (params.shift_bit - 1)) + 1) >> 1;
      // y = a_max, T >= a_max
      if (T >= a_max){
        y_data[i] = a_max;
        // y = a_min, T <= a_min
      } else if (T <= a_min) {
        y_data[i] = a_min;
      } else {
        // y = T, a_min < T < a_max
        y_data[i] = T;
      }
    }
});

CVM_REGISTER_GLOBAL("cvm.runtime.formal.cvm_left_shift")
.set_body([](CVMArgs args, CVMRetValue *ret){
    auto x_data = CVMArg2Data<int32_t>(args[0]); 
    auto y_data = CVMArg2Data<int32_t>(args[1]); 
    auto params = CVMArg2Attr<cvm::top::CVMLeftShiftParam>(args[2]);
    int32_t precision = params.precision;
    // alpha = 2^(precision-1) - 1
    int32_t alpha =  (((int64_t)1 << (precision-1))-1);
    int32_t a_min = -alpha;
    int32_t a_max = alpha;
    auto size = CVMArgSize(args[0]);
    // T = X << shift_bit
    // Y = clip(T, -alpha, alpha)
    for (uint32_t i = 0; i < size; i++) {
      int32_t T = x_data[i] << (int32_t)params.shift_bit; 
      // y = a_max, T >= a_max
      if (T >= a_max){
        y_data[i] = a_max;
        // y = a_min, T <= a_min
      } else if (T <= a_min) {
        y_data[i] = a_min;
      } else {
        // y = T, a_min < T < a_max
        y_data[i] = T;
      }
    }
});

void FlattenX(int32_t *x, int32_t *y, int Size){
    memcpy(y, x, Size*sizeof(int32_t));
}

CVM_REGISTER_GLOBAL("cvm.runtime.formal.flatten")
    .set_body([](CVMArgs args, CVMRetValue* rv)
{
    auto X = args[0];
    auto x_shape = CVMArgShape(X);
    auto x_data = CVMArg2Data<int32_t>(args[0]); 
    auto y_data = CVMArg2Data<int32_t>(args[1]); 
    if(x_data == y_data) return;
    FlattenX(x_data, y_data, CVMArgSize(X));
});

CVM_REGISTER_GLOBAL("cvm.runtime.formal.reshape")
    .set_body([](CVMArgs args, CVMRetValue *ret)
{
    auto X = args[0];
    auto x_shape = CVMArgShape(X);
    auto x_data = CVMArg2Data<int32_t>(args[0]); 
    auto y_data = CVMArg2Data<int32_t>(args[1]); 
    if(x_data == y_data) return;
    FlattenX(x_data, y_data, CVMArgSize(X));
});

}
}
