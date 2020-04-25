#define MATRIX_PAD 64
extern "C"{
void im2col(const int * data_im,
    char* data_col,
    const int n,
    const int height, const int width, const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w,
    const int stride_h, const int stride_w,
    const int dilation_h, const int dilation_w,
    const int height_col, const int width_col) {
#pragma HLS INTERFACE m_axi port=data_im offset=slave bundle=gmem
#pragma HLS INTERFACE m_axi port=data_col offset=slave bundle=gmem
#pragma HLS INTERFACE s_axilite port=data_im  bundle=control
#pragma HLS INTERFACE s_axilite port=data_col bundle=control
#pragma HLS INTERFACE s_axilite port=n bundle=control
#pragma HLS INTERFACE s_axilite port=height bundle=control
#pragma HLS INTERFACE s_axilite port=width bundle=control
#pragma HLS INTERFACE s_axilite port=kernel_h bundle=control
#pragma HLS INTERFACE s_axilite port=kernel_w bundle=control
#pragma HLS INTERFACE s_axilite port=pad_h bundle=control
#pragma HLS INTERFACE s_axilite port=pad_w bundle=control
#pragma HLS INTERFACE s_axilite port=stride_h bundle=control
#pragma HLS INTERFACE s_axilite port=stride_w bundle=control
#pragma HLS INTERFACE s_axilite port=dilation_h bundle=control
#pragma HLS INTERFACE s_axilite port=dilation_w bundle=control
#pragma HLS INTERFACE s_axilite port=height_col bundle=control
#pragma HLS INTERFACE s_axilite port=width_col bundle=control
//#pragma HLS INTERFACE s_axilite port=offset bundle=control
#pragma HLS INTERFACE s_axilite port = return bundle = control

  const int BS = 32;
  const int MAX_KERNEL = 11;
  char buf[MAX_KERNEL*MAX_KERNEL][BS];
  const int cols = height_col * width_col;
  const int cols_offset = (cols + 63) / 64 * 64;
  for(int ri = 0; ri < n; ri+=BS){
    int chunk_n = BS;
    if(ri + BS > n) chunk_n = n - ri;
    for(int ti = 0; ti < chunk_n; ti++){
#pragma HLS PIPELINE
      int index = ri + ti; 
      const int h_index = index / width_col;
      const int h_col = h_index % height_col;
      const int w_col = index % width_col;
      const int c_im = h_index / height_col;
      const int c_col = c_im * kernel_h * kernel_w;
      const int h_offset = h_col * stride_h - pad_h;
      const int w_offset = w_col * stride_w - pad_w;
      int dst_index = c_col * cols_offset + h_col * width_col + w_col;
      int src_index = (c_im * height + h_offset) * width + w_offset;

      for (int i = 0; i < kernel_h; ++i) {
        for (int j = 0; j < kernel_w; ++j) {
          int h_im = h_offset + i * dilation_h;
          int w_im = w_offset + j * dilation_w;
          if(h_im < 0 || w_im < 0 || h_im >= height || w_im >= width){
            //data_col[dst_index] = 0;
            buf[i*kernel_w + j][ti] = 0;
          }else{
            //data_col[dst_index] = data_im[src_index + i*dilation_h * width + j * dilation_w];
            buf[i*kernel_w + j][ti] = data_im[src_index + i*dilation_h * width + j * dilation_w];
          }
          //dst_index += cols_offset;//height_col * width_col;
        }
      }
    }

    for (int i = 0; i < kernel_h; ++i) {
      for (int j = 0; j < kernel_w; ++j) {
        const int kernel_index = i*kernel_w + j;
        for(int ti = 0; ti < chunk_n; ti++){
#pragma HLS PIPELINE
          int index = ri + ti; 
          const int h_index = index / width_col;
          const int h_col = h_index % height_col;
          const int w_col = index % width_col;
          const int c_im = h_index / height_col;
          const int c_col = c_im * kernel_h * kernel_w;
          const int h_offset = h_col * stride_h - pad_h;
          const int w_offset = w_col * stride_w - pad_w;
          int dst_index = c_col * cols_offset + h_col * width_col + w_col + kernel_index * cols_offset;
          data_col[dst_index] = buf[kernel_index][ti];
        }
        //dst_index += cols_offset;//height_col * width_col;
      }
    }
  }

}
}
