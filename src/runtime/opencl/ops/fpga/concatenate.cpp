
extern "C"{
void concatenate(const int *input, int *output, const int y_axis_batch, const int x_axis_batch, const int n, const int offset){
#pragma HLS INTERFACE m_axi port = input offset = slave bundle = gmem
#pragma HLS INTERFACE m_axi port = output offset = slave bundle = gmem
#pragma HLS INTERFACE s_axilite port = input bundle = control
#pragma HLS INTERFACE s_axilite port = output bundle = control
#pragma HLS INTERFACE s_axilite port = y_axis_batch bundle = control
#pragma HLS INTERFACE s_axilite port = x_axis_batch bundle = control
#pragma HLS INTERFACE s_axilite port = n bundle = control
#pragma HLS INTERFACE s_axilite port = offset  bundle = control
#pragma HLS INTERFACE s_axilite port = return bundle = control

  for(int i = 0; i < n; i++){
//#pragma HLS PIPELINE
    int y_iter = i / x_axis_batch;
    int x_iter = i % x_axis_batch;
    output[offset + y_iter * y_axis_batch + x_iter] = input[y_iter * x_axis_batch + x_iter];
  }
}

}
