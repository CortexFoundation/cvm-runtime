extern "C" {
void pool(const int* input, int* output,
	const int batch, const int c, const int h, const int w,
	const int kh, const int kw,   
	const int oh, const int ow,
	const int pad_h, const int pad_w,             
	const int stride_h, const int stride_w){      
#pragma HLS INTERFACE m_axi port=input offset=slave bundle=gmem0
#pragma HLS INTERFACE m_axi port=output offset=slave bundle=gmem0
#pragma HLS INTERFACE s_axilite port=input bundle=control
#pragma HLS INTERFACE s_axilite port=output bundle=control
#pragma HLS INTERFACE s_axilite port=batch bundle=control
#pragma HLS INTERFACE s_axilite port=c bundle=control
#pragma HLS INTERFACE s_axilite port=h bundle=control
#pragma HLS INTERFACE s_axilite port=w bundle=control
#pragma HLS INTERFACE s_axilite port=kh bundle=control
#pragma HLS INTERFACE s_axilite port=kw bundle=control
#pragma HLS INTERFACE s_axilite port=oh bundle=control
#pragma HLS INTERFACE s_axilite port=ow bundle=control
#pragma HLS INTERFACE s_axilite port=pad_h bundle=control
#pragma HLS INTERFACE s_axilite port=pad_w bundle=control
#pragma HLS INTERFACE s_axilite port=stride_h bundle=control
#pragma HLS INTERFACE s_axilite port=stride_w bundle=control
#pragma HLS INTERFACE s_axilite port = return bundle = control
  
	for(int n = 0; n < batch; n++){
	  for(int i = 0; i < c; i++){
	    for(int y = 0; y < oh; y+=8){
	      for(int x = 0; x < ow; x+=8){
          int chunk_size_y = 8;
          int chunk_size_x = 8;
          if(y + 8 > oh) chunk_size_y = oh - y;
          if(x + 8 > ow) chunk_size_x = ow - x;
          for(int ih = 0; ih < chunk_size_y; ih++){
            for(int iw = 0; iw < chunk_size_x; iw++){
              int min = 1 << 31;
              int max = min;
              for(int fy = 0; fy < kh; fy++){
                for(int fx = 0; fx < kw; fx++){
#pragma HLS PIPELINE
                  int index_y = (y+ih)*stride_h + fy - pad_h;
                  int index_x = (x+iw)*stride_w + fx - pad_w;
                  int tmp = min;
                  if(index_y >= 0 && index_y < h && index_x >= 0 && index_x < w){
                    tmp = input[n*c*h*w + i*h*w + index_y*w + index_x]; 
                  }
                  max = max < tmp ? tmp : max;
                }
              }    
              output[n*c*oh*ow + i*oh*ow + (y+ih)*ow + x+iw] = max;
            }
          } 
        }
      }
    }
  }

}
}

