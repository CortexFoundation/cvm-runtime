extern "C"{
void groupwise_conv2d_bias(
   const int *x_data, 
   const int *w_data, 
   const int *b_data,
   int *y_data, 
   const int n_batch, const int in_channels, const int x_h, const int x_w,
   const int filter_c, const int filter_h, const int filter_w,
   const int out_channels, const int o_h, const int o_w,
   const int pad_h, const int pad_w, 
   const int stride_h, const int stride_w, 
   const int dilation_h, const int dilation_w,
   const int groups){
#pragma HLS INTERFACE m_axi port=x_data offset=slave bundle=gmem
#pragma HLS INTERFACE m_axi port=w_data offset=slave bundle=gmem
#pragma HLS INTERFACE m_axi port=b_data offset=slave bundle=gmem
#pragma HLS INTERFACE m_axi port=y_data offset=slave bundle=gmem
#pragma HLS INTERFACE s_axilite port=x_data bundle=control
#pragma HLS INTERFACE s_axilite port=w_data bundle=control
#pragma HLS INTERFACE s_axilite port=b_data  bundle=control
#pragma HLS INTERFACE s_axilite port=y_data bundle=control
#pragma HLS INTERFACE s_axilite port=n_batch bundle=control
#pragma HLS INTERFACE s_axilite port=in_channels_h bundle=control
#pragma HLS INTERFACE s_axilite port=x_h bundle=control
#pragma HLS INTERFACE s_axilite port=x_w bundle=control
#pragma HLS INTERFACE s_axilite port=filter_c bundle=control
#pragma HLS INTERFACE s_axilite port=filter_h  bundle=control
#pragma HLS INTERFACE s_axilite port=filtwr_w bundle=control
#pragma HLS INTERFACE s_axilite port=out_channels bundle=control
#pragma HLS INTERFACE s_axilite port=o_h bundle=control
#pragma HLS INTERFACE s_axilite port=o_w bundle=control
#pragma HLS INTERFACE s_axilite port=pad_h bundle=control
#pragma HLS INTERFACE s_axilite port=pad_w bundle=control
#pragma HLS INTERFACE s_axilite port=stride_h bundle=control
#pragma HLS INTERFACE s_axilite port=stride_w bundle=control
#pragma HLS INTERFACE s_axilite port=dilation_h bundle=control
#pragma HLS INTERFACE s_axilite port=dilation_w bundle=control
#pragma HLS INTERFACE s_axilite port=groups bundle=control
#pragma HLS INTERFACE s_axilite port = return bundle = control

  int ochannels_per_group = out_channels / groups;
  int ichannels_per_group = in_channels / groups;
  for(int n = 0; n < n_batch; ++n){
    for(int oc = 0; oc < out_channels; ++oc){
      for(int oh = 0; oh < o_h; ++oh){
        for(int ow = 0; ow < o_w; ++ow){
          int oi = n * out_channels * o_h * o_w + oc * o_h * o_w + oh * o_w + ow;
//#pragma HLS PIPELINE 
          int sum = 0;
          int ic = oc / ochannels_per_group * ichannels_per_group;
          for(int tic = 0; tic < ichannels_per_group; ++tic){
            for(int fh = 0; fh < filter_h; ++fh){
              for(int fw = 0; fw < filter_w; ++fw){
                int th = oh * stride_h + fh*dilation_h - pad_h;
                int tw = ow * stride_w + fw*dilation_w - pad_w;
                if(th < 0 || tw < 0 || th >= x_h || tw >= x_w)
                  continue;
                sum += x_data[n * in_channels * x_h * x_w + (ic+tic) * x_h * x_w + th * x_w + tw]
                  * w_data[oc * filter_c * filter_h * filter_w + tic * filter_h * filter_w + fh * filter_w + fw];
              }
            }
          }
          y_data[oi] = sum + b_data[oc];
        }
      }
    }
  }
}

}
