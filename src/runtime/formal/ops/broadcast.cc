#include "ops.h"

namespace cvm {
namespace runtime {

double cvm_op_broadcast_cnt = 0;

/*
inline int32_t broadcast_i_index(
    int64_t* oshape, 
    uint64_t o_index, 
    int64_t* ishape, 
    int idim, int odim){
  if(idim == 1 && ishape[0] == 1) return 0;
  uint64_t index = 0;
  uint64_t allIndex = 1;
  for(int i = 0; i < idim; i++){
    int idx = idim - 1 - i;
    int ovar = o_index % oshape[idx+odim-idim];
    if(ovar < ishape[idx]){
      index += allIndex * ovar;
    }
    allIndex =  allIndex * ishape[idx];
    o_index /= oshape[idx + odim-idim];
  }
  return index;
}

typedef std::function<int32_t(int32_t a, int32_t b)> broadcast_func;

static void broadcast(DLTensor *args0, 
                      DLTensor* args1, 
                      DLTensor* args2, 
                      broadcast_func const &f){
  int32_t *a = static_cast<int32_t*>(args0->data);
  int32_t *b = static_cast<int32_t*>(args1->data);
  int32_t *c = static_cast<int32_t*>(args2->data);

  if(getSize(args1) == 1){
    for(uint64_t i = 0; i < getSize(args2); ++i){
      c[i] = f(a[i], b[0]);
    }
  }else{
    for(uint64_t i = 0; i < getSize(args2); ++i){
      uint64_t o_index = i;
      int64_t a_index = broadcast_i_index(args2->shape, o_index, args0->shape, args0->ndim, args2->ndim);
      int64_t b_index = broadcast_i_index(args2->shape, o_index, args1->shape, args1->ndim, args2->ndim);
      c[i] = f(a[a_index], b[b_index]);
    }
  }

}
*/

CVM_REGISTER_GLOBAL("cvm.runtime.formal.broadcast_add")
.set_body([](CVMArgs args, CVMRetValue *ret){
    DLTensor *args0 = args[0];
    DLTensor *args1 = args[1];
    DLTensor *args2 = args[2];
   // broadcast_func f = [](int32_t a, int32_t b) -> int32_t {
   //   return a + b;
   // };
   

    // inputs: A (args0), B(args1)
    // outputs: Y (args2)
    // A.shape = (m_0, m_1,,, m_{M-1})
    // B.shape = (n_0, n_1,,, n_{N-1})
    // K = max(M, N)
    // SA_i = { m_{i-K+M}, i >= K - M; 1, i < K - M }
    // SB_i = { n_{i-K+N}, i >= K-N; 1, i < K - N }
    // Y.shape = (k_0, k_1,,, k_{K-1}), k_i = max(SA_i, SB_i)  
    // For \forall i \in [0, K)], d_{i} \in [0, k_{i})
    // a_i = min(d_{i}, SA_i-1)
    // b_i = min(d_{i}, SB_i-1)
    // Y[d_0, d_1,,, d_{K-1}] = A[a_0, a_1,,, a_{K-1}] + B[b_0, b_1,,, b_{K-1}]
    std::vector<int64_t> d_k, a_k, b_k, SA, SB;
    int K = args2->ndim;
    for (auto i = 0; i < K; i++){
      a_k.push_back(0);
      b_k.push_back(0);
      d_k.push_back(0);
      SA.push_back(0);
      SB.push_back(0);
    }
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
    for (uint64_t j = 0; j < getSize(args2); j++){
      for (int i = 0; i < K; i++){
        if (SA[i] - 1 < d_k[i])
          a_k[i] = SA[i] - 1;
        else
          a_k[i] = d_k[i];

        if (SB[i] - 1 < d_k[i])
          b_k[i] = SB[i] - 1;
        else
          b_k[i] = d_k[i];
      }
      int index0 = a_k[0];
      for (int i = 1; i < K; i++){
        index0 = index0 * SA[i] + a_k[i];
      } 
      int index1 = b_k[0];
      for (int i = 1; i < K; i++){
        index1 = index1 * SB[i] + b_k[i];
      } 
      int index2 = d_k[0];
      for (int i = 1; i < K; i++){
        index2 = index2 * args2->shape[i] + d_k[i];
      }
      c[index2] = a[index0] + b[index1];

      d_k[K-1]++; 
      for (int i = K-1; i > 0; i--){
        if (d_k[i] == args2->shape[i]){
          d_k[i] = 0;
          d_k[i-1]++;
        }
      }
    }
    // broadcast(args0, args1, args2, f);
    // print_to_file(args2, "broadcast_add.txt");
});

CVM_REGISTER_GLOBAL("cvm.runtime.formal.broadcast_sub")
.set_body([](CVMArgs args, CVMRetValue *ret){
    DLTensor *args0 = args[0];
    DLTensor *args1 = args[1];
    DLTensor *args2 = args[2];
    // inputs: A (args0), B(args1)
    // outputs: Y (args2)
    // A.shape = (m_0, m_1,,, m_{M-1})
    // B.shape = (n_0, n_1,,, n_{N-1})
    // K = max(M, N)
    // SA_i = { m_{i-K+M}, i >= K - M; 1, i < K - M }
    // SB_i = { n_{i-K+N}, i >= K-N; 1, i < K - N }
    // Y.shape = (k_0, k_1,,, k_{K-1}), k_i = max(SA_i, SB_i)  
    // For \forall i \in [0, K)], d_{i} \in [0, k_{i})
    // a_i = min(d_{i}, SA_i-1)
    // b_i = min(d_{i}, SB_i-1)
    // Y[d_0, d_1,,, d_{K-1}] = A[a_0, a_1,,, a_{K-1}] - B[b_0, b_1,,, b_{K-1}]
    std::vector<int64_t> d_k, a_k, b_k, SA, SB;
    int K = args2->ndim;
    for (auto i = 0; i < K; i++){
      a_k.push_back(0);
      b_k.push_back(0);
      d_k.push_back(0);
      SA.push_back(0);
      SB.push_back(0);
    }
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
    for (uint64_t j = 0; j < getSize(args2); j++){
      for (int i = 0; i < K; i++){
        if (SA[i] - 1 < d_k[i])
          a_k[i] = SA[i] - 1;
        else
          a_k[i] = d_k[i];

        if (SB[i] - 1 < d_k[i])
          b_k[i] = SB[i] - 1;
        else
          b_k[i] = d_k[i];
      }
      int index0 = a_k[0];
      for (int i = 1; i < K; i++){
        index0 = index0 * SA[i] + a_k[i];
      } 
      int index1 = b_k[0];
      for (int i = 1; i < K; i++){
        index1 = index1 * SB[i] + b_k[i];
      } 
      int index2 = d_k[0];
      for (int i = 1; i < K; i++){
        index2 = index2 * args2->shape[i] + d_k[i];
      }
      c[index2] = a[index0] - b[index1];

      d_k[K-1]++; 
      for (int i = K-1; i > 0; i--){
        if (d_k[i] == args2->shape[i]){
          d_k[i] = 0;
          d_k[i-1]++;
        }
      }
    }
});

CVM_REGISTER_GLOBAL("cvm.runtime.formal.broadcast_mul")
.set_body([](CVMArgs args, CVMRetValue *ret){
    DLTensor *args0 = args[0];
    DLTensor *args1 = args[1];
    DLTensor *args2 = args[2];
    // inputs: A (args0), B(args1)
    // outputs: Y (args2)
    // A.shape = (m_0, m_1,,, m_{M-1})
    // B.shape = (n_0, n_1,,, n_{N-1})
    // K = max(M, N)
    // SA_i = { m_{i-K+M}, i >= K - M; 1, i < K - M }
    // SB_i = { n_{i-K+N}, i >= K-N; 1, i < K - N }
    // Y.shape = (k_0, k_1,,, k_{K-1}), k_i = max(SA_i, SB_i)  
    // For \forall i \in [0, K)], d_{i} \in [0, k_{i})
    // a_i = min(d_{i}, SA_i-1)
    // b_i = min(d_{i}, SB_i-1)
    // Y[d_0, d_1,,, d_{K-1}] = A[a_0, a_1,,, a_{K-1}] * B[b_0, b_1,,, b_{K-1}]
    std::vector<int64_t> d_k, a_k, b_k, SA, SB;
    int K = args2->ndim;
    for (auto i = 0; i < K; i++){
      a_k.push_back(0);
      b_k.push_back(0);
      d_k.push_back(0);
      SA.push_back(0);
      SB.push_back(0);
    }
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
    for (uint64_t j = 0; j < getSize(args2); j++){
      for (int i = 0; i < K; i++){
        if (SA[i] - 1 < d_k[i])
          a_k[i] = SA[i] - 1;
        else
          a_k[i] = d_k[i];

        if (SB[i] - 1 < d_k[i])
          b_k[i] = SB[i] - 1;
        else
          b_k[i] = d_k[i];
      }
      int index0 = a_k[0];
      for (int i = 1; i < K; i++){
        index0 = index0 * SA[i] + a_k[i];
      } 
      int index1 = b_k[0];
      for (int i = 1; i < K; i++){
        index1 = index1 * SB[i] + b_k[i];
      } 
      int index2 = d_k[0];
      for (int i = 1; i < K; i++){
        index2 = index2 * args2->shape[i] + d_k[i];
      }
      c[index2] = a[index0] * b[index1];

      d_k[K-1]++; 
      for (int i = K-1; i > 0; i--){
        if (d_k[i] == args2->shape[i]){
          d_k[i] = 0;
          d_k[i-1]++;
        }
      }
    }
});

CVM_REGISTER_GLOBAL("cvm.runtime.formal.broadcast_max")
.set_body([](CVMArgs args, CVMRetValue *ret){
    DLTensor *args0 = args[0];
    DLTensor *args1 = args[1];
    DLTensor *args2 = args[2];
    // inputs: A (args0), B(args1)
    // outputs: Y (args2)
    // A.shape = (m_0, m_1,,, m_{M-1})
    // B.shape = (n_0, n_1,,, n_{N-1})
    // K = max(M, N)
    // SA_i = { m_{i-K+M}, i >= K - M; 1, i < K - M }
    // SB_i = { n_{i-K+N}, i >= K-N; 1, i < K - N }
    // Y.shape = (k_0, k_1,,, k_{K-1}), k_i = max(SA_i, SB_i)  
    // For \forall i \in [0, K)], d_{i} \in [0, k_{i})
    // a_i = min(d_{i}, SA_i-1)
    // b_i = min(d_{i}, SB_i-1)
    // Y[d_0, d_1,,, d_{K-1}] = max A[a_0, a_1,,, a_{K-1}], B[b_0, b_1,,, b_{K-1}]
    std::vector<int64_t> d_k, a_k, b_k, SA, SB;
    int K = args2->ndim;
    for (auto i = 0; i < K; i++){
      a_k.push_back(0);
      b_k.push_back(0);
      d_k.push_back(0);
      SA.push_back(0);
      SB.push_back(0);
    }
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
    for (uint64_t j = 0; j < getSize(args2); j++){
      for (int i = 0; i < K; i++){
        if (SA[i] - 1 < d_k[i])
          a_k[i] = SA[i] - 1;
        else
          a_k[i] = d_k[i];

        if (SB[i] - 1 < d_k[i])
          b_k[i] = SB[i] - 1;
        else
          b_k[i] = d_k[i];
      }
      int index0 = a_k[0];
      for (int i = 1; i < K; i++){
        index0 = index0 * SA[i] + a_k[i];
      } 
      int index1 = b_k[0];
      for (int i = 1; i < K; i++){
        index1 = index1 * SB[i] + b_k[i];
      } 
      int index2 = d_k[0];
      for (int i = 1; i < K; i++){
        index2 = index2 * args2->shape[i] + d_k[i];
      }
      c[index2] = a[index0] > b[index1] ? a[index0] : b[index1];

      d_k[K-1]++; 
      for (int i = K-1; i > 0; i--){
        if (d_k[i] == args2->shape[i]){
          d_k[i] = 0;
          d_k[i-1]++;
        }
      }
    }
});

CVM_REGISTER_GLOBAL("cvm.runtime.formal.broadcast_div")
.set_body([](CVMArgs args, CVMRetValue *ret){
    DLTensor *args0 = args[0];
    DLTensor *args1 = args[1];
    DLTensor *args2 = args[2];
    // inputs: A (args0), B(args1)
    // outputs: Y (args2)
    // A.shape = (m_0, m_1,,, m_{M-1})
    // B.shape = (n_0, n_1,,, n_{N-1})
    // K = max(M, N)
    // SA_i = { m_{i-K+M}, i >= K - M; 1, i < K - M }
    // SB_i = { n_{i-K+N}, i >= K-N; 1, i < K - N }
    // Y.shape = (k_0, k_1,,, k_{K-1}), k_i = max(SA_i, SB_i)  
    // For \forall i \in [0, K)], d_{i} \in [0, k_{i})
    // a_i = min(d_{i}, SA_i-1)
    // b_i = min(d_{i}, SB_i-1)
    // Y[d_0, d_1,,, d_{K-1}] = A[a_0, a_1,,, a_{K-1}] / B[b_0, b_1,,, b_{K-1}]
    std::vector<int64_t> d_k, a_k, b_k, SA, SB;
    int K = args2->ndim;
    for (auto i = 0; i < K; i++){
      a_k.push_back(0);
      b_k.push_back(0);
      d_k.push_back(0);
      SA.push_back(0);
      SB.push_back(0);
    }
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
    for (uint64_t j = 0; j < getSize(args2); j++){
      for (int i = 0; i < K; i++){
        if (SA[i] - 1 < d_k[i])
          a_k[i] = SA[i] - 1;
        else
          a_k[i] = d_k[i];

        if (SB[i] - 1 < d_k[i])
          b_k[i] = SB[i] - 1;
        else
          b_k[i] = d_k[i];
      }
      int index0 = a_k[0];
      for (int i = 1; i < K; i++){
        index0 = index0 * SA[i] + a_k[i];
      } 
      int index1 = b_k[0];
      for (int i = 1; i < K; i++){
        index1 = index1 * SB[i] + b_k[i];
      } 
      int index2 = d_k[0];
      for (int i = 1; i < K; i++){
        index2 = index2 * args2->shape[i] + d_k[i];
      }
      c[index2] = a[index0] / b[index1];

      d_k[K-1]++; 
      for (int i = K-1; i > 0; i--){
        if (d_k[i] == args2->shape[i]){
          d_k[i] = 0;
          d_k[i-1]++;
        }
      }
    }
});

CVM_REGISTER_GLOBAL("cvm.runtime.formal.broadcast_greater")
.set_body([](CVMArgs args, CVMRetValue *ret){
    DLTensor *args0 = args[0];
    DLTensor *args1 = args[1];
    DLTensor *args2 = args[2];
    // inputs: A (args0), B(args1)
    // outputs: Y (args2)
    // A.shape = (m_0, m_1,,, m_{M-1})
    // B.shape = (n_0, n_1,,, n_{N-1})
    // K = max(M, N)
    // SA_i = { m_{i-K+M}, i >= K - M; 1, i < K - M }
    // SB_i = { n_{i-K+N}, i >= K-N; 1, i < K - N }
    // Y.shape = (k_0, k_1,,, k_{K-1}), k_i = max(SA_i, SB_i)  
    // For \forall i \in [0, K)], d_{i} \in [0, k_{i})
    // a_i = min(d_{i}, SA_i-1)
    // b_i = min(d_{i}, SB_i-1)
    // Y[d_0, d_1,,, d_{K-1}] = A[a_0, a_1,,, a_{K-1}] > B[b_0, b_1,,, b_{K-1}]
    std::vector<int64_t> d_k, a_k, b_k, SA, SB;
    int K = args2->ndim;
    for (auto i = 0; i < K; i++){
      a_k.push_back(0);
      b_k.push_back(0);
      d_k.push_back(0);
      SA.push_back(0);
      SB.push_back(0);
    }
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
    for (uint64_t j = 0; j < getSize(args2); j++){
      for (int i = 0; i < K; i++){
        if (SA[i] - 1 < d_k[i])
          a_k[i] = SA[i] - 1;
        else
          a_k[i] = d_k[i];

        if (SB[i] - 1 < d_k[i])
          b_k[i] = SB[i] - 1;
        else
          b_k[i] = d_k[i];
      }
      int index0 = a_k[0];
      for (int i = 1; i < K; i++){
        index0 = index0 * SA[i] + a_k[i];
      } 
      int index1 = b_k[0];
      for (int i = 1; i < K; i++){
        index1 = index1 * SB[i] + b_k[i];
      } 
      int index2 = d_k[0];
      for (int i = 1; i < K; i++){
        index2 = index2 * args2->shape[i] + d_k[i];
      }
      c[index2] = a[index0] > b[index1];

      d_k[K-1]++; 
      for (int i = K-1; i > 0; i--){
        if (d_k[i] == args2->shape[i]){
          d_k[i] = 0;
          d_k[i-1]++;
        }
      }
    }
});
}
}

