extern "C"{
//only support nearest
void upsampling(const int *x_data, int *y_data, const int scale, const int ih, const int iw,
    const int oh, const int ow, const int channel){
#pragma HLS INTERFACE m_axi port=x_data offset=slave bundle=gmem
#pragma HLS INTERFACE m_axi port=y_data offset=slave bundle=gmem
#pragma HLS INTERFACE s_axilite port=x_data bundle=control
#pragma HLS INTERFACE s_axilite port=y_data bundle=control
#pragma HLS INTERFACE s_axilite port=scale bundle=control
#pragma HLS INTERFACE s_axilite port=ih bundle=control
#pragma HLS INTERFACE s_axilite port=iw bundle=control
#pragma HLS INTERFACE s_axilite port=oh bundle=control
#pragma HLS INTERFACE s_axilite port=ow bundle=control
#pragma HLS INTERFACE s_axilite port=channel bundle=control
#pragma HLS INTERFACE s_axilite port = return bundle = control
  for(int b = 0; b < channel; b++){
    for(int r = 0; r < oh; r++){
      for(int c = 0; c < ow; c++){
#pragma HLS PIPELINE
        y_data[b * oh * ow + r * ow + c] = x_data[b * ih * iw + r/scale * iw + c/scale];
      }
    }
  }
}
}
