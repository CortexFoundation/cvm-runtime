extern "C"{
void negative(const int *x_data, int *y_data, int n){
#pragma HLS INTERFACE m_axi port=x_data offset=slave bundle=gmem
#pragma HLS INTERFACE m_axi port=y_data offset=slave bundle=gmem
#pragma HLS INTERFACE s_axilite port=x_data bundle=control
#pragma HLS INTERFACE s_axilite port=y_data bundle=control
#pragma HLS INTERFACE s_axilite port=n bundle=control
#pragma HLS INTERFACE s_axilite port = return bundle = control
  for(int i = 0; i < n; i++){
#pragma HLS PIPELINE
    y_data[i] = -x_data[i];
  }
}
}
