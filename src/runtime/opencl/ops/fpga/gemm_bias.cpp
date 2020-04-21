
const int BLOCK_SIZE = 16;
const unsigned int c_dim = BLOCK_SIZE;

extern "C"{
void gemm_bias(const int *A, const int* B, const int* bias, int *C, const int M, const int K, const int N){
#pragma HLS INTERFACE m_axi port=A offset=slave bundle=gmem
#pragma HLS INTERFACE m_axi port=B offset=slave bundle=gmem
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
  //const int*B = BA + M*K;
  //int temp_sum[BLOCK_SIZE];

//#pragma HLS ARRAY_PARTITION variable = bufB dim = 2 complete
//#pragma HLS ARRAY_PARTITION variable = bufC dim = 2 complete
//#pragma HLS ARRAY_PARTITION variable = temp_sum dim = 1 complete
//read_A:
//    for (int itr = 0, i = 0, j = 0; itr < M*K; itr++, j++) {
//       #pragma HLS LOOP_TRIPCOUNT min=c_dim*c_dim max=c_dim*c_dim
//       #pragma HLS PIPELINE II=1
//        if (j == K) {
//            j = 0;
//            i++;
//        }
//        bufA[i][j] = A[itr];
//    }
//read_B:
//    for (int itr = 0, i = 0, j = 0; itr < K*N; itr++, j++) {
//       #pragma HLS LOOP_TRIPCOUNT min=c_dim*c_dim max=c_dim*c_dim
//       #pragma HLS PIPELINE II=1
//        if (j == N) {
//            j = 0;
//            i++;
//        }
//        bufB[i][j] = B[itr];
//    }
//arraypart1:
//    for (int row = 0; row < M; row++) {
//       #pragma HLS LOOP_TRIPCOUNT min=c_dim max=c_dim
//    arraypart2:
//        for (int col = 0; col < K; col++) {
//           #pragma HLS LOOP_TRIPCOUNT min=c_dim max=c_dim
//           #pragma HLS PIPELINE II=1
//        arraypart3:
//            for (int j = 0; j < BLOCK_SIZE; j++) {
//               #pragma HLS LOOP_TRIPCOUNT min=c_dim max=c_dim
//                int result = (col == 0) ? 0 : temp_sum[j];
//                result += bufA[row][col] * bufB[col][j];
//                temp_sum[j] = result;
//                if (col == K - 1)
//                    bufC[row][j] = result;
//            }
//        }
//    }
//
//// Write results from local buffer to global memory for out
//writeC:
//    for (int itr = 0, i = 0, j = 0; itr < M*N; itr++, j++) {
//       #pragma HLS LOOP_TRIPCOUNT min=c_dim*c_dim max=c_dim*c_dim
//       #pragma HLS PIPELINE II=1
//        if (j == N) {
//            j = 0;
//            i++;
//        }
//        C[itr] = bufC[i][j] + bias[i];
//    }




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
              sum += (int)bufA[ii][kk] * bufB[kk][jj];  
            }
            bufC[ii][jj] += sum;
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
