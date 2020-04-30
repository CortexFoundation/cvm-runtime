
extern "C" {
void cvm_clip(const int *input, int*output, const int n, const int min, const int max){
#pragma HLS INTERFACE m_axi port=input offset=slave bundle=gmem
#pragma HLS INTERFACE m_axi port=output offset=slave bundle=gmem
#pragma HLS INTERFACE s_axilite port=input bundle=control
#pragma HLS INTERFACE s_axilite port=output bundle=control
#pragma HLS INTERFACE s_axilite port=n bundle=control
#pragma HLS INTERFACE s_axilite port=min bundle=control
#pragma HLS INTERFACE s_axilite port=max bundle=control
#pragma HLS INTERFACE s_axilite port = return bundle = control

  const int BUF_SIZE = 128;
  int buf_i[BUF_SIZE];
  int buf_o[BUF_SIZE];
  for(int i = 0; i < n; i+= BUF_SIZE){
    int chunk_size = BUF_SIZE;
    if(i + BUF_SIZE > n) chunk_size = n - i;

read1:
    for(int j = 0; j < chunk_size; j++){
      #pragma HLS PIPELINE II=1
      buf_i[j] = input[i+j];
    }
clip:
    for(int j = 0; j < chunk_size; j++){
      #pragma HLS PIPELINE 
      int tmp = buf_i[j];
      if(tmp > max) tmp = max;
      if(tmp < min) tmp = min;
      buf_o[j] = tmp ;//< min ? min : tmp;
    }
write:
    for(int j = 0; j < chunk_size; j++){
      #pragma HLS PIPELINE II=1
      output[i+j] = buf_o[j];
    }
  }

}
}
