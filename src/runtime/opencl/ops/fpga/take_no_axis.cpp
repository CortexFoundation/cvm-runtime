extern "C"{
void take_noaxis(const int *x_data, const int *indices_data, int *y_data, const int ysize, const int xsize){
#pragma HLS INTERFACE m_axi port=x_data offset=slave bundle=gmem
#pragma HLS INTERFACE m_axi port=indices_data offset=slave bundle=gmem
#pragma HLS INTERFACE m_axi port=y_data offset=slave bundle=gmem
#pragma HLS INTERFACE s_axilite port=x_data bundle=control
#pragma HLS INTERFACE s_axilite port=indices_data bundle=control
#pragma HLS INTERFACE s_axilite port=y_data bundle=control

#pragma HLS INTERFACE s_axilite port=ysize bundle=control
#pragma HLS INTERFACE s_axilite port=xsize bundle=control
#pragma HLS INTERFACE s_axilite port = return bundle = control
  //int tid = threadIdx.x + blockIdx.x * blockDim.x;
  //for(int i = tid; i < ysize; i+=gridDim.x*blockDim.x){
  for(int i = 0; i < ysize; i++){
#pragma HLS PIPELINE
    int in_i = indices_data[i];
    if(in_i < 0) in_i = 0;
    if(in_i > xsize-1) in_i = xsize-1;
    y_data[i] = x_data[in_i];
  }
}
}
