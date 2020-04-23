extern "C"{
void get_valid_count(const int *inputs, int *valid_count, int *outputs,
    const int batchs, const int N, const int K, 
    const int score_threshold){
#pragma HLS INTERFACE m_axi port = inputs offset = slave bundle = gmem
#pragma HLS INTERFACE m_axi port = valid_count offset = slave bundle = gmem
#pragma HLS INTERFACE m_axi port = outputs offset = slave bundle = gmem
#pragma HLS INTERFACE s_axilite port = inputs bundle = control
#pragma HLS INTERFACE s_axilite port = valid_count bundle = control
#pragma HLS INTERFACE s_axilite port = outputs bundle = control
#pragma HLS INTERFACE s_axilite port = batchs bundle = control
#pragma HLS INTERFACE s_axilite port = N bundle = control
#pragma HLS INTERFACE s_axilite port = K bundle = control
#pragma HLS INTERFACE s_axilite port = score_threshold bundle = control
#pragma HLS INTERFACE s_axilite port = return bundle = control

  for(int i = 0; i < batchs; i++){
    int y_index = 0;
    const int *input = inputs + i * N * K;
    int *output = outputs + i * N * K;
    for(int j = 0; j < N; j++){
      const int *row = input + j * K;
      if(row[1] > score_threshold){
        for(int k = 0; k < K; k++){
          output[y_index * K + k] = row[k]; 
        }
        y_index += 1;
      }
    }
    valid_count[i] = y_index;
  }
}
}
