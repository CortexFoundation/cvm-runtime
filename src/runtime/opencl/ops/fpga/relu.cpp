extern "C" {
void relu(const int* input, int*output, const int n){
#pragma HLS INTERFACE m_axi port=input offset=slave bundle=gmem
#pragma HLS INTERFACE m_axi port=output offset=slace bundle=gmem
#pragma HLS INTERFACE s_axilite port=input bundle=control
#pragma HLS INTERFACE s_axilite port=output bundle=control
#pragma HLS INTERFACE s_axilite port=n bundle=control
#pragma HLS INTERFACE s_axilite port = return bundle = control

  int buf_i[1024];
  int buf_o[1024];
  for(int i = 0; i < n; i+= 1024){
    int chunk_size = 1024;
    if(i + 1024 > n) chunk_size = n - i;

read1:
    for(int j = 0; j < chunk_size; j++){
      #pragma HLS PIPELINE II=1
      buf_i[j] = input[i+j];
    }
shift:
    for(int j = 0; j < chunk_size; j++){
      #pragma HLS PIPELINE II=1
      int tmp = buf_i[j];
      buf_o[j] = tmp < 0 ? 0 : tmp;
    }
write:
    for(int j = 0; j < chunk_size; j++){
      #pragma HLS PIPELINE II=1
      output[i+j] = buf_o[j];
    }
  }
}

}
