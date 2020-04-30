extern "C" {
void relu(const int* input, int*output, const int n){
#pragma HLS INTERFACE m_axi port=input offset=slave bundle=gmem
#pragma HLS INTERFACE m_axi port=output offset=slave bundle=gmem
#pragma HLS INTERFACE s_axilite port=input bundle=control
#pragma HLS INTERFACE s_axilite port=output bundle=control
#pragma HLS INTERFACE s_axilite port=n bundle=control
#pragma HLS INTERFACE s_axilite port = return bundle = control

  const int BS = 32;
  int buf_i[BS];
  int buf_o[BS];
  for(int i = 0; i < n; i+= BS){
    int chunk_size = BS;
    if(i + BS > n) chunk_size = n - i;

read1:
    for(int j = 0; j < chunk_size; j++){
      #pragma HLS PIPELINE II=1
      buf_i[j] = input[i+j];
    }
shift:
    for(int j = 0; j < chunk_size; j++){
      #pragma HLS PIPELINE 
      int tmp = buf_i[j];
      if(tmp < 0) tmp = 0;
      buf_o[j] = tmp;
    }
write:
    for(int j = 0; j < chunk_size; j++){
      #pragma HLS PIPELINE II=1
      output[i+j] = buf_o[j];
    }
  }
}

}
