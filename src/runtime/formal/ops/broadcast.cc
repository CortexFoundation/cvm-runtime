#include "ops.h"

namespace cvm {
namespace runtime {

double cvm_op_broadcast_cnt = 0;

typedef std::function<int32_t(int32_t a, int32_t b)> broadcast_func;

static void broadcast(DLTensor *args0, 
                      DLTensor* args1, 
                      DLTensor* args2, 
                      broadcast_func const &f){
    // inputs: A (args0), B(args1)
    // outputs: Y (args2)
    // A.shape = (m_0, m_1,,, m_{M-1})
    // B.shape = (n_0, n_1,,, n_{N-1})
    // K = max(M, N)
    int K = args2->ndim;
    std::vector<int64_t> d_k, a_k, b_k, SA, SB;
    for (auto i = 0; i < K; i++){
      a_k.push_back(0);
      b_k.push_back(0);
      d_k.push_back(0);
      SA.push_back(0);
      SB.push_back(0);
    }
    // SA_i = m_{i-K+M}, i >= K - M
    // SA_i = 1, i < K - M
    // SB_i = n_{i-K+N}, i >= K-N
    // SB_i = 1, i < K - N
    for (auto i = 0; i < K; i++){
      if (i < K - args0->ndim){
        SA[i] = 1;
      } else {
        SA[i] = args0->shape[i - K + args0->ndim]; 
      }

      if (i < K - args1->ndim){
        SB[i] = 1;
      } else {
        SB[i] = args1->shape[i - K + args1->ndim]; 
      }
    }
    int32_t *a = static_cast<int32_t*>(args0->data);
    int32_t *b = static_cast<int32_t*>(args1->data);
    int32_t *c = static_cast<int32_t*>(args2->data);
    // Y.shape = (k_0, k_1,,, k_{K-1}), k_i = max(SA_i, SB_i)  
    // For \forall i \in [0, K)], d_{i} \in [0, k_{i})
    for (uint64_t j = 0; j < getSize(args2); j++){
      for (int i = 0; i < K; i++){
        // a_i = min(d_{i}, SA_i-1)
        if (SA[i] - 1 < d_k[i])
          a_k[i] = SA[i] - 1;
        else
          a_k[i] = d_k[i];

        // b_i = min(d_{i}, SB_i-1)
        if (SB[i] - 1 < d_k[i])
          b_k[i] = SB[i] - 1;
        else
          b_k[i] = d_k[i];
      }
      // index0 = the number of (a_0, a_1,,, a_{K-1}) on decimal digit
      int index0 = a_k[0];
      for (int i = 1; i < K; i++){
        index0 = index0 * SA[i] + a_k[i];
      } 
      // index1 = the number of (b_0, b_1,,, b_{K-1}) on decimal digit
      int index1 = b_k[0];
      for (int i = 1; i < K; i++){
        index1 = index1 * SB[i] + b_k[i];
      } 
      // index2 = the number of (d_0, d_1,,, d_{K-1}) on decimal digit
      int index2 = d_k[0];
      for (int i = 1; i < K; i++){
        index2 = index2 * args2->shape[i] + d_k[i];
      }
      // Y[d_0, d_1,,, d_{K-1}] = f(A[a_0, a_1,,, a_{K-1}], B[b_0, b_1,,, b_{K-1}])
      c[index2] = f(a[index0], b[index1]);

      d_k[K-1]++; 
      for (int i = K-1; i > 0; i--){
        if (d_k[i] == args2->shape[i]){
          d_k[i] = 0;
          d_k[i-1]++;
        }
      }
    }

}

CVM_REGISTER_GLOBAL("cvm.runtime.formal.broadcast_add")
.set_body([](CVMArgs args, CVMRetValue *ret){
    DLTensor *args0 = args[0];
    DLTensor *args1 = args[1];
    DLTensor *args2 = args[2];
    broadcast_func f = [](int32_t a, int32_t b) -> int32_t {
      return a + b;
    };

    broadcast(args0, args1, args2, f);
});

CVM_REGISTER_GLOBAL("cvm.runtime.formal.broadcast_sub")
.set_body([](CVMArgs args, CVMRetValue *ret){
    DLTensor *args0 = args[0];
    DLTensor *args1 = args[1];
    DLTensor *args2 = args[2];
    broadcast_func f = [](int32_t a, int32_t b) -> int32_t {
      return a - b;
    };

    broadcast(args0, args1, args2, f);
});

CVM_REGISTER_GLOBAL("cvm.runtime.formal.broadcast_mul")
.set_body([](CVMArgs args, CVMRetValue *ret){
    DLTensor *args0 = args[0];
    DLTensor *args1 = args[1];
    DLTensor *args2 = args[2];
    broadcast_func f = [](int32_t a, int32_t b) -> int32_t {
      return a * b;
    };

    broadcast(args0, args1, args2, f);
});

CVM_REGISTER_GLOBAL("cvm.runtime.formal.broadcast_max")
.set_body([](CVMArgs args, CVMRetValue *ret){
    DLTensor *args0 = args[0];
    DLTensor *args1 = args[1];
    DLTensor *args2 = args[2];
    broadcast_func f = [](int32_t a, int32_t b) -> int32_t {
      return a > b ? a : b;
    };

    broadcast(args0, args1, args2, f);
});

CVM_REGISTER_GLOBAL("cvm.runtime.formal.broadcast_div")
.set_body([](CVMArgs args, CVMRetValue *ret){
    DLTensor *args0 = args[0];
    DLTensor *args1 = args[1];
    DLTensor *args2 = args[2];
    broadcast_func f = [](int32_t a, int32_t b) -> int32_t {
      return b == 0 ? 0 : a/b;
    };

    broadcast(args0, args1, args2, f);
});

CVM_REGISTER_GLOBAL("cvm.runtime.formal.broadcast_greater")
.set_body([](CVMArgs args, CVMRetValue *ret){
    DLTensor *args0 = args[0];
    DLTensor *args1 = args[1];
    DLTensor *args2 = args[2];
    broadcast_func f = [](int32_t a, int32_t b) -> int32_t {
      return a > b;
    };

    broadcast(args0, args1, args2, f);
});
}
}

