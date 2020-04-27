extern "C"{
void transpose(const int *x_data, int *y_data, 
    const int ndim, const int ysize, 
    const int axes_ndim,
    const int xshp0, const int xshp1, const int xshp2, const int xshp3, const int xshp4, const int xshp5,
    const int yshp0, const int yshp1, const int yshp2, const int yshp3, const int yshp4, const int yshp5,
    const int axes0, const int axes1, const int axes2, const int axes3, const int axes4, const int axes5){
#pragma HLS INTERFACE m_axi port=x_data offset=slave bundle=gmem
#pragma HLS INTERFACE m_axi port=y_data offset=slave bundle=gmem
#pragma HLS INTERFACE s_axilite port=x_data bundle=control
#pragma HLS INTERFACE s_axilite port=y_data bundle=control

#pragma HLS INTERFACE s_axilite port=ndim bundle=control
#pragma HLS INTERFACE s_axilite port=ysize bundle=control
#pragma HLS INTERFACE s_axilite port=axes_ndim bundle=control

#pragma HLS INTERFACE s_axilite port=xshp0 bundle=control
#pragma HLS INTERFACE s_axilite port=xshp1 bundle=control
#pragma HLS INTERFACE s_axilite port=xshp2 bundle=control
#pragma HLS INTERFACE s_axilite port=xshp3 bundle=control
#pragma HLS INTERFACE s_axilite port=xshp4 bundle=control
#pragma HLS INTERFACE s_axilite port=xshp5 bundle=control

#pragma HLS INTERFACE s_axilite port=yshp0 bundle=control
#pragma HLS INTERFACE s_axilite port=yshp1 bundle=control
#pragma HLS INTERFACE s_axilite port=yshp2 bundle=control
#pragma HLS INTERFACE s_axilite port=yshp3 bundle=control
#pragma HLS INTERFACE s_axilite port=yshp4 bundle=control
#pragma HLS INTERFACE s_axilite port=yshp5 bundle=control

#pragma HLS INTERFACE s_axilite port=axes0 bundle=control
#pragma HLS INTERFACE s_axilite port=axes1 bundle=control
#pragma HLS INTERFACE s_axilite port=axes2 bundle=control
#pragma HLS INTERFACE s_axilite port=axes3 bundle=control
#pragma HLS INTERFACE s_axilite port=axes4 bundle=control
#pragma HLS INTERFACE s_axilite port=axes5 bundle=control

#pragma HLS INTERFACE s_axilite port = return bundle = control
  //int tid = threadIdx.x + blockDim.x * blockIdx.x;
  const int xshape[MAX_DIM] = {xshp0, xshp1, xshp2, xshp3, xshp4, xshp5};
  const int yshape[MAX_DIM] = {yshp0, yshp1, yshp2, yshp3, yshp4, yshp5};
  const int axes_data[MAX_DIM] = {axes0, axes1, axes2, axes3, axes4, axes5};
  //for(int i = tid; i < ysize; i+=gridDim.x*blockDim.x){
  for(int i = 0; i < ysize; i++){
#pragma HLS PIPELINE
    int in_i = 0, o_i = i;
    for(int j = ndim-1; j >= 0; j--){
      int col = o_i % yshape[j + MAX_DIM - ndim];
      o_i /= yshape[j + MAX_DIM - ndim];
      int xj = j;
      if(axes_ndim > 0){
        xj = axes_data[j + MAX_DIM - axes_ndim];
      }else{
        xj = ndim - 1 - j;
      }
      int xi = 1;
      for(int tx = ndim-1; tx > xj; tx--){
        xi *= xshape[tx + MAX_DIM - ndim];
      }
      in_i += col * xi;
    }
    y_data[i] = x_data[in_i];
  }
}
}
