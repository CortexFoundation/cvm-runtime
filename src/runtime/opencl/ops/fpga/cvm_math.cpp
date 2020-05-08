extern "C"{
void cvm_math(const int *x, int *y, const int n, const int type){
#pragma HLS INTERFACE m_axi port=x offset=slave bundle=gmem
#pragma HLS INTERFACE m_axi port=y offset=slave bundle=gmem
#pragma HLS INTERFACE s_axilite port=x bundle=control
#pragma HLS INTERFACE s_axilite port=y bundle=control
#pragma HLS INTERFACE s_axilite port=n bundle=control
#pragma HLS INTERFACE s_axilite port=type bundle=control
#pragma HLS INTERFACE s_axilite port = return bundle = control
  //int tid = threadIdx.x + blockIdx.x * blockDim.x;
  //for(int j = tid; j < n; j += gridDim.x * blockDim.x){
  for(int i = 0; i < n; i++){
#pragma HLS PIPELINE
    int x_item = x[i];
    if(type == 0){ // abs
      if(x_item < 0) x_item = -x_item;
      y[i] = x_item;
    }else if(type == 1){ // log
      if(x_item < 0) x_item = -x_item;
      y[i] = 64;
      for(int j = 1; j < 64; j++){
        long long tmp = (long long)1 << j;
        if(x_item< tmp){
          y[i] = i;
          return;
        }
      }
    }else if(type == 2){  // negative
      y[i] = -x_item;
    }else if(type == 3){ // relu
      if(x_item < 0) x_item = 0;
      y[i] = x_item;
    }
  }
}
}

