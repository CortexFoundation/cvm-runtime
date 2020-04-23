const int MAX_DIM = 6;
extern "C"{
void reduce(const int *x, int *y, 
    const int axis_ndim, const int xndim, const int yndim, const int ysize, const int axis_size, const int type,
    const int xshp0, const int xshp1, const int xshp2, const int xshp3, const int xshp4, const int xshp5,
    const int yshp0, const int yshp1, const int yshp2, const int yshp3, const int yshp4, const int yshp5, 
    const int axshp0, const int axshp1, const int axshp2, const int axshp3, const int axshp4, const int axshp5,
    const int exshp0, const int exshp1, const int exshp2, const int exshp3, const int exshp4, const int exshp5,
    const int fshp0, const int fshp1, const int fshp2, const int fshp3, const int fshp4, const int fshp5
){
#pragma HLS INTERFACE m_axi port=x offset=slave bundle=gmem
#pragma HLS INTERFACE m_axi port=y offset=slave bundle=gmem

#pragma HLS INTERFACE s_axilite port=x bundle=control
#pragma HLS INTERFACE s_axilite port=y bundle=control

#pragma HLS INTERFACE s_axilite port=axis_ndim bundle=control
#pragma HLS INTERFACE s_axilite port=xndim bundle=control
#pragma HLS INTERFACE s_axilite port=yndim bundle=control
#pragma HLS INTERFACE s_axilite port=ysize bundle=control
#pragma HLS INTERFACE s_axilite port=axis_size bundle=control

#pragma HLS INTERFACE s_axilite port=type bundle=control
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

#pragma HLS INTERFACE s_axilite port=axshp0 bundle=control
#pragma HLS INTERFACE s_axilite port=axshp1 bundle=control
#pragma HLS INTERFACE s_axilite port=axshp2 bundle=control
#pragma HLS INTERFACE s_axilite port=axshp3 bundle=control
#pragma HLS INTERFACE s_axilite port=axshp4 bundle=control
#pragma HLS INTERFACE s_axilite port=axshp5 bundle=control

#pragma HLS INTERFACE s_axilite port=exshp0 bundle=control
#pragma HLS INTERFACE s_axilite port=exshp1 bundle=control
#pragma HLS INTERFACE s_axilite port=exshp2 bundle=control
#pragma HLS INTERFACE s_axilite port=exshp3 bundle=control
#pragma HLS INTERFACE s_axilite port=exshp4 bundle=control
#pragma HLS INTERFACE s_axilite port=exshp5 bundle=control

#pragma HLS INTERFACE s_axilite port=fshp0 bundle=control
#pragma HLS INTERFACE s_axilite port=fshp1 bundle=control
#pragma HLS INTERFACE s_axilite port=fshp2 bundle=control
#pragma HLS INTERFACE s_axilite port=fshp3 bundle=control
#pragma HLS INTERFACE s_axilite port=fshp4 bundle=control
#pragma HLS INTERFACE s_axilite port=fshp5 bundle=control
#pragma HLS INTERFACE s_axilite port = return bundle = control

  const int xshape[MAX_DIM] = {xshp0, xshp1, xshp2, xshp3, xshp4, xshp5};
  const int yshape[MAX_DIM] = {yshp0, yshp1, yshp2, yshp3, yshp4, yshp5};
  const int realAxis[MAX_DIM] = {axshp0, axshp1, axshp2, axshp3, axshp4, axshp5};
  const int every_xdim_size[MAX_DIM] = {exshp0, exshp1, exshp2, exshp3, exshp4, exshp5};
  const int flag[MAX_DIM] = {fshp0, fshp1, fshp2, fshp3, fshp4, fshp5};

  for(int i = 0; i < ysize; i++){
    int in_i = 0, o_i = i;
    for(int j = yndim-1, xj = xndim-1; j>=0; j--){
      int col = o_i % yshape[j + MAX_DIM - yndim];
      o_i /= yshape[j + MAX_DIM - yndim];
      while(xj >= 0 && flag[(xj--) + MAX_DIM - xndim] == 1);
      in_i += col * every_xdim_size[xj+1 + MAX_DIM - xndim];
    }
    int tmp = x[in_i];
    for(int xi = 1; xi < axis_size; xi++){
      int o_i = xi, tmp_in_i = 0;
      for(int j = axis_ndim - 1; j>=0; j--){
        int col = o_i % xshape[realAxis[j + MAX_DIM - axis_ndim] + MAX_DIM - xndim];
        o_i /= xshape[realAxis[j + MAX_DIM - axis_ndim] + MAX_DIM - xndim];
        tmp_in_i += col * every_xdim_size[realAxis[j + MAX_DIM - axis_ndim] + MAX_DIM - xndim];
      }
      //tmp = f(tmp, x[in_i + tmp_in_i]);
      int x_value = x[in_i + tmp_in_i];
      if(type == 0){//max
        if(tmp < x_value)
          tmp = x_value;
      }else{ // type
        tmp += x_value; 
      }
    }
    y[i] = tmp;
  }
}
}
