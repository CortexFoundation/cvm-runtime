
extern "C" {
void cvm_shift(const int *input, int *output, const int shift_b, const int n, const int min, const int max, const int type){
#pragma HLS INTERFACE m_axi port=input offset=slave bundle=gmem
#pragma HLS INTERFACE m_axi port=output offset=slave bundle=gmem
#pragma HLS INTERFACE s_axilite port=input bundle=control
#pragma HLS INTERFACE s_axilite port=output bundle=control
#pragma HLS INTERFACE s_axilite port=shift_b bundle=control
#pragma HLS INTERFACE s_axilite port=n bundle=control
#pragma HLS INTERFACE s_axilite port=min bundle=control
#pragma HLS INTERFACE s_axilite port=max bundle=control
#pragma HLS INTERFACE s_axilite port=type bundle=control
#pragma HLS INTERFACE s_axilite port = return bundle = control

  const int BS = 64;
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
      if(type == 0){ // right_shift
        tmp = ((tmp >> (shift_b -1)) + 1) >> 1;
      }else if(type == 1){        // left_shift
        tmp = tmp << shift_b;
      }else{
        //clip   
      }
      if(tmp > max) tmp = max;
      if(tmp < min) tmp = min;
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
