extern "C"{
void where_shape0(const int *x_data, const int *y_data, const int *condition_data, int *result, const int shape0, const int n){
#pragma HLS INTERFACE m_axi port=x_data offset=slave bundle=gmem
#pragma HLS INTERFACE m_axi port=y_data offset=slave bundle=gmem
#pragma HLS INTERFACE m_axi port=condition_data offset=slave bundle=gmem
#pragma HLS INTERFACE m_axi port=result offset=slave bundle=gmem
#pragma HLS INTERFACE s_axilite port=x_data bundle=control
#pragma HLS INTERFACE s_axilite port=y_data bundle=control
#pragma HLS INTERFACE s_axilite port=condition_data bundle=control
#pragma HLS INTERFACE s_axilite port=result bundle=control
#pragma HLS INTERFACE s_axilite port=shape0 bundle=control
#pragma HLS INTERFACE s_axilite port=n bundle=control
#pragma HLS INTERFACE s_axilite port = return bundle = control
  for(int i = 0; i < shape0; i++){
    for(int j = 0; j < n; j++){
//#pragma HLS PIPELINE
      int condition = condition_data[i];
      int y = y_data[i*n+j];
      int x = x_data[i*n+j];
      //result[i * n + j] = (condition_data[i] == 0 ? y_data[i * n + j] : x_data[i * n + j]);
      if(condition == 0){
        result[i*n+j] = y;
      }else{
        result[i*n+j] = x;
      }
    }
  }
}
}
