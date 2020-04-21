extern "C"{
  void int32_to_int8(const int *input, char* output, const int n){
#pragma HLS INTERFACE m_axi port=input offset=slave bundle=gmem
#pragma HLS INTERFACE m_axi port=output offset=slave bundle=gmem
#pragma HLS INTERFACE s_axilite port=input bundle=control
#pragma HLS INTERFACE s_axilite port=output bundle=control
#pragma HLS INTERFACE s_axilite port=n bundle=control
#pragma HLS INTERFACE s_axilite port = return bundle = control
    const int BUF_SIZE = 64;
    int buf[BUF_SIZE];
    for(int i = 0; i < n; i+=BUF_SIZE){
      int chunk_size = BUF_SIZE;
      if(i + chunk_size > n) chunk_size = n - i;
load:
      for(int j = 0; j < chunk_size; j++){
#pragma HLS PIPELINE II=1
        buf[j] = input[i + j];
      }
write:
      for(int j = 0; j < chunk_size; j++){
#pragma HLS PIPELINE II=1
        output[i+j] = (char)buf[j];
      }
    }
  }
}
