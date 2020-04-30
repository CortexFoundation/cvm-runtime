extern "C"{
void where_same_shape(const int *x_data, const int *condition, int *result, const int n){
#pragma HLS INTERFACE m_axi port=x_data offset=slave bundle=gmem
#pragma HLS INTERFACE m_axi port=condition offset=slave bundle=gmem
#pragma HLS INTERFACE m_axi port=result offset=slave bundle=gmem
#pragma HLS INTERFACE s_axilite port=x_data bundle=control
#pragma HLS INTERFACE s_axilite port=condition bundle=control
#pragma HLS INTERFACE s_axilite port=result bundle=control
#pragma HLS INTERFACE s_axilite port=n bundle=control
#pragma HLS INTERFACE s_axilite port = return bundle = control

  const int BS = 32;
  int bufx[BS];
  int bufc[BS];
  for(int i = 0; i < n; i+=BS){
    int chunk_size = BS;
    if(i + BS > n) chunk_size = n - i;
    for(int j = 0; j < chunk_size; j++){
#pragma HLS PIPELINE
      bufx[j] = x_data[i+j]; 
      bufc[j] = condition[i+j]; 
    }
    for(int j = 0; j < chunk_size; j++){
#pragma HLS PIPELINE
      if(bufc[j] != 0){
        result[i+j] = bufx[j];
      } 
    }
  }
}
}
