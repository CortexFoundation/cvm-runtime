
const int BLOCK_SIZE = 16;
const unsigned int c_dim = BLOCK_SIZE;

extern "C"{
void gemm_bias(const char *A, const char* B, const int* bias, int *C, const int M, const int K, const int N){
#pragma HLS INTERFACE m_axi port=A offset=slave bundle=gmem
#pragma HLS INTERFACE m_axi port=B offset=slave bundle=gmem1
#pragma HLS INTERFACE m_axi port=bias offset=slave bundle=gmem
#pragma HLS INTERFACE m_axi port=C offset=slave bundle=gmem
#pragma HLS INTERFACE s_axilite port=A bundle=control
#pragma HLS INTERFACE s_axilite port=B bundle=control
#pragma HLS INTERFACE s_axilite port=bias bundle=control
#pragma HLS INTERFACE s_axilite port=C bundle=control
#pragma HLS INTERFACE s_axilite port=M bundle=control
#pragma HLS INTERFACE s_axilite port=K bundle=control
#pragma HLS INTERFACE s_axilite port=N bundle=control
#pragma HLS INTERFACE s_axilite port = return bundle = control
  char bufA[BLOCK_SIZE][BLOCK_SIZE];
  char bufB[BLOCK_SIZE][BLOCK_SIZE];
  int bufC[BLOCK_SIZE][BLOCK_SIZE];
  const int TM = (M+63)/64*64;
  const int TK = (K+63)/64*64;
  const int TN = (N+63)/64*64;

  int offset = TK*TN;

#pragma HLS ARRAY_PARTITION variable = bufC dim = 2 complete
#pragma HLS ARRAY_PARTITION variable = bufB dim = 2 complete

  for(int i = 0; i < M; i += BLOCK_SIZE){
    int chunk_size_m = BLOCK_SIZE;
    if(i + chunk_size_m > M) chunk_size_m = M - i;
    for(int j = 0; j < N; j += BLOCK_SIZE){
      int chunk_size_n = BLOCK_SIZE;
      if(j + chunk_size_n > N) chunk_size_n = N - j;
#pragma HLS PIPELINE
init:
      for(int ii = 0; ii < BLOCK_SIZE; ii++){
        for(int jj = 0; jj < BLOCK_SIZE; jj++){
#pragma HLS PIPELINE II=1
          bufC[ii][jj] = 0;
        }
      }

      for(int k = 0; k < K; k += BLOCK_SIZE){
        //int chunk_size_k = BLOCK_SIZE;
        //if(chunk_size_k + k > K) chunk_size_k = K - k;

readA:
        for(int ii = 0; ii < BLOCK_SIZE; ii++){
#pragma HLS PIPELINE II=1
          for(int kk = 0; kk < BLOCK_SIZE; kk++){
#pragma HLS UNROLL factor=2
            bufA[ii][kk] = A[offset + (i+ii)*TK + k + kk];
          }
        }

readB:
        for(int kk = 0; kk < BLOCK_SIZE; kk++){
#pragma HLS PIPELINE II=1
          for(int jj = 0; jj < BLOCK_SIZE; jj++){
#pragma HLS UNROLL factor=2
            bufB[kk][jj] = B[(k + kk)*TN + j + jj];
          }
        }

madd:
        for(int ii = 0; ii < BLOCK_SIZE; ii++){
          for(int kk = 0; kk < BLOCK_SIZE; kk+=2){
#pragma HLS PIPELINE 
            int a = bufA[ii][kk];
            int a1 = bufA[ii][kk+1];
            for(int jj = 0; jj < BLOCK_SIZE; jj++){
#pragma HLS UNROLL
              int c = a * bufB[kk][jj];  
              int c1 = a1 * bufB[kk+1][jj];  
              bufC[ii][jj] += c + c1;
            }
          }
        }
      }
write: 
      for(int ii = 0; ii < chunk_size_m; ii++){
        for(int jj = 0; jj < chunk_size_n; jj++){
#pragma HLS PIPELINE II=1
          C[(i+ii)*N + j+jj] = bufC[ii][jj] + bias[i+ii];
        }
      }
    }
  }
}
}
