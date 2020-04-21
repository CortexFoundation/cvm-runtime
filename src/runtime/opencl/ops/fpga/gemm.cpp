extern "C"{
void gemm(const char *ext_space, int *C, const int M, const int K, const int N){
#pragma HLS INTERFACE m_axi port=ext_space offset=slave bundle=gmem
#pragma HLS INTERFACE m_axi port=C offset=slave bundle=gmem
#pragma HLS INTERFACE s_axilite port=ext_space bundle=control
#pragma HLS INTERFACE s_axilite port=C bundle=control
#pragma HLS INTERFACE s_axilite port=M bundle=control
#pragma HLS INTERFACE s_axilite port=K bundle=control
#pragma HLS INTERFACE s_axilite port=N bundle=control
#pragma HLS INTERFACE s_axilite port = return bundle = control
  const int BLOCK_SIZE = 8;
  char bufA[BLOCK_SIZE][BLOCK_SIZE];
  char bufB[BLOCK_SIZE][BLOCK_SIZE];
  int bufC[BLOCK_SIZE][BLOCK_SIZE];
  const char *A = ext_space;
  const int offset = M*K;
  const char *B = ext_space + offset;
  
  for(int i = 0; i < M; i += BLOCK_SIZE){
    int chunk_size_m = BLOCK_SIZE;
    if(i + chunk_size_m > M) chunk_size_m = M - i;
    for(int j = 0; j < N; j += BLOCK_SIZE){
      int chunk_size_n = BLOCK_SIZE;
      if(j + chunk_size_n > N) chunk_size_n = N - j;
init:
      for(int ii = 0; ii < chunk_size_m; ii++){
        for(int jj = 0; jj < chunk_size_n; jj++){
#pragma HLS PIPELINE II=1
          bufC[ii][jj] = 0;
        }
      }

      for(int k = 0; k < K; k += BLOCK_SIZE){
        int chunk_size_k = BLOCK_SIZE;
        if(chunk_size_k + k > K) chunk_size_k = K - k;
readA:
        for(int ii = 0; ii < chunk_size_m; ii++){
          for(int kk = 0; kk < chunk_size_k; kk++){
#pragma HLS PIPELINE II=1
            bufA[ii][kk] = A[(i+ii)*K + k + kk];
          }
        }
readB:
        for(int kk = 0; kk < chunk_size_k; kk++){
          for(int jj = 0; jj < chunk_size_n; jj++){
#pragma HLS PIPELINE II=1
            bufB[kk][jj] = B[(k + kk)*N + j + jj];
          }
        }
madd:
        for(int ii = 0; ii < chunk_size_m; ii++){
          for(int jj = 0; jj < chunk_size_n; jj++){
#pragma HLS PIPELINE II=1
            int sum = 0;
            for(int kk = 0; kk < chunk_size_k; kk++){
              sum += bufA[ii][kk] * bufB[kk][jj];  
            }
            bufC[ii][jj] = sum;
          }
        }
      }
write: 
      for(int ii = 0; ii < chunk_size_m; ii++){
        for(int jj = 0; jj < chunk_size_n; jj++){
#pragma HLS PIPELINE II=1
          C[(i+ii)*N + j+jj] = bufC[ii][jj];
        }
      }
    }
  }
}
}
