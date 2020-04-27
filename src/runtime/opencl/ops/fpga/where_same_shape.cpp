extern "C"{
void where_same_shape(const int *x_data, const int *y_data, const int *condition_data, int *result, const int n){
#pragma HLS INTERFACE m_axi port=x_data offset=slave bundle=gmem
#pragma HLS INTERFACE m_axi port=y_data offset=slave bundle=gmem
#pragma HLS INTERFACE m_axi port=condition_data offset=slave bundle=gmem
#pragma HLS INTERFACE m_axi port=result offset=slave bundle=gmem
#pragma HLS INTERFACE s_axilite port=x_data bundle=control
#pragma HLS INTERFACE s_axilite port=y_data bundle=control
#pragma HLS INTERFACE s_axilite port=condition_data bundle=control
#pragma HLS INTERFACE s_axilite port=result bundle=control
#pragma HLS INTERFACE s_axilite port=n bundle=control
#pragma HLS INTERFACE s_axilite port = return bundle = control
  for(int i = 0; i < n; i++){
#pragma HLS PIPELINE
    result[i] = condition_data[i] == 0 ? y_data[i] : x_data[i];
  }
}
}
