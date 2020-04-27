#define MAX_DIM 6
#define min(a, b) ((a) > (b) ? (b) : (a))
#define max(a, b) ((a) > (b) ? (a) : (b))
extern "C"{
void take(const int *x_data, const int *indices_data, int *y_data, const int yndim,
    const int xndim, const int indices_ndim, const int ysize, const int axis,
    const int xshp0, const int xshp1, const int xshp2, const int xshp3, const int xshp4, const int xshp5,
    const int yshp0, const int yshp1, const int yshp2, const int yshp3, const int yshp4, const int yshp5,
    const int idshp0, const int idshp1, const int idshp2, const int idshp3, const int idshp4, const int idshp5){
#pragma HLS INTERFACE m_axi port=x_data offset=slave bundle=gmem
#pragma HLS INTERFACE m_axi port=indices_data offset=slave bundle=gmem
#pragma HLS INTERFACE m_axi port=y_data offset=slave bundle=gmem
#pragma HLS INTERFACE s_axilite port=x_data bundle=control
#pragma HLS INTERFACE s_axilite port=indices_data bundle=control
#pragma HLS INTERFACE s_axilite port=y_data bundle=control

#pragma HLS INTERFACE s_axilite port=yndim bundle=control
#pragma HLS INTERFACE s_axilite port=xndim bundle=control
#pragma HLS INTERFACE s_axilite port=indices_ndim bundle=control
#pragma HLS INTERFACE s_axilite port=ysize bundle=control
#pragma HLS INTERFACE s_axilite port=axis bundle=control

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

#pragma HLS INTERFACE s_axilite port=idshp0 bundle=control
#pragma HLS INTERFACE s_axilite port=idshp1 bundle=control
#pragma HLS INTERFACE s_axilite port=idshp2 bundle=control
#pragma HLS INTERFACE s_axilite port=idshp3 bundle=control
#pragma HLS INTERFACE s_axilite port=idshp4 bundle=control
#pragma HLS INTERFACE s_axilite port=idshp5 bundle=control
#pragma HLS INTERFACE s_axilite port = return bundle = control
  const int xshape[MAX_DIM] = {xshp0, xshp1, xshp2, xshp3, xshp4, xshp5};
  const int yshape[MAX_DIM] = {yshp0, yshp1, yshp2, yshp3, yshp4, yshp5};
  const int indices_shape[MAX_DIM] = {idshp0, idshp1, idshp2, idshp3, idshp4, idshp5};
  for(int i = 0; i < ysize; i++){
    int o_i = i, x_i = 0, indices_i = 0, x_shape_size = 1, indices_shape_size = 1;
    for(int j = yndim - 1, k = indices_ndim-1; j>=axis; j--){
      int col = o_i % yshape[j + MAX_DIM - yndim];
      o_i /= yshape[j + MAX_DIM - yndim];
      if(j < axis + indices_ndim){
        indices_i += col * indices_shape_size;
        indices_shape_size = indices_shape_size * indices_shape[k + MAX_DIM - indices_ndim];
        --k;
      }
    }

    o_i = i;
    int k = xndim - 1;
    for(int j = yndim - 1; j >= axis + indices_ndim; j--, k--){
      int col = o_i % yshape[j + MAX_DIM - yndim];
      o_i /= yshape[j + MAX_DIM - yndim];
      x_i += col * x_shape_size;
      x_shape_size = x_shape_size * xshape[k + MAX_DIM - xndim];
    }

    int x_indices_i = min(max(indices_data[indices_i], 0), (int)xshape[k + MAX_DIM - xndim]-1);
    x_i += x_indices_i * x_shape_size;
    x_shape_size = x_shape_size * xshape[k + MAX_DIM - xndim];
    --k;

    o_i = i;
    for(int j = yndim - 1; j>=0 && k >= 0; j--){
      int col = o_i % yshape[j + MAX_DIM - yndim];
      o_i /= yshape[j + MAX_DIM - yndim];
      if(j < axis){
        x_i += col * x_shape_size;
        x_shape_size = x_shape_size * xshape[k + MAX_DIM - xndim];
        --k;
      }
    }
    y_data[i] = x_data[x_i];
  }
}
}
