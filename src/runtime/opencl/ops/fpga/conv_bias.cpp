extern "C" {
void conv_bias(const int* input, const int* weight, const int *bias, int* output,
    const int batch, const int c, const int h, const int w,
    const int oc, const int kh, const int kw,    //3*3 
    const int oh, const int ow){
  //const int pad_h, const int pad_w,            //0 
  //const int stride_h, const int stride_w,      //1 
  //const int dilation_h, const int dilation_w){ //1
#pragma HLS INTERFACE m_axi port=input offset=slave bundle=gmem0
#pragma HLS INTERFACE m_axi port=weight offset=slave bundle=gmem1
#pragma HLS INTERFACE m_axi port=bias offset=slave bundle=gmem2
#pragma HLS INTERFACE m_axi port=output offset=slave bundle=gmem0
#pragma HLS INTERFACE s_axilite port=input bundle=control
#pragma HLS INTERFACE s_axilite port=weight bundle=control
#pragma HLS INTERFACE s_axilite port=bias bundle=control
#pragma HLS INTERFACE s_axilite port=output bundle=control
#pragma HLS INTERFACE s_axilite port=batch bundle=control
#pragma HLS INTERFACE s_axilite port=c bundle=control
#pragma HLS INTERFACE s_axilite port=h bundle=control
#pragma HLS INTERFACE s_axilite port=w bundle=control
#pragma HLS INTERFACE s_axilite port=oc bundle=control
#pragma HLS INTERFACE s_axilite port=kh bundle=control
#pragma HLS INTERFACE s_axilite port=kw bundle=control
#pragma HLS INTERFACE s_axilite port=oh bundle=control
#pragma HLS INTERFACE s_axilite port=ow bundle=control
  //#pragma HLS INTERFACE s_axilite port=pad_h bundle=control
  //#pragma HLS INTERFACE s_axilite port=pad_w bundle=control
  //#pragma HLS INTERFACE s_axilite port=stride_h bundle=control
  //#pragma HLS INTERFACE s_axilite port=stride_w bundle=control
  //#pragma HLS INTERFACE s_axilite port=dilation_h bundle=control
  //#pragma HLS INTERFACE s_axilite port=dilation_w bundle=control
#pragma HLS INTERFACE s_axilite port = return bundle = control

  const int BLOCK_SIZE = 8;
  const int MAX_KERNEL_SIZE=3;
  //blocks: 8*8
  int bufw[MAX_KERNEL_SIZE][MAX_KERNEL_SIZE];
  int bufo[BLOCK_SIZE][BLOCK_SIZE];
  int bufi[BLOCK_SIZE+MAX_KERNEL_SIZE][BLOCK_SIZE+MAX_KERNEL_SIZE];

  for(int n = 0; n < batch; n++){
    for(int i = 0; i < oc; i++){
      for(int y = 0; y < oh; y+=BLOCK_SIZE){
        for(int x = 0; x < ow; x+=BLOCK_SIZE){
          int chunk_size_y = BLOCK_SIZE;
          int chunk_size_x = BLOCK_SIZE;
          if(y + BLOCK_SIZE > oh) chunk_size_y = oh - y;
          if(x + BLOCK_SIZE > ow) chunk_size_x = ow - x;

          for(int iy = 0; iy < BLOCK_SIZE; iy++){
            for(int ix = 0; ix < BLOCK_SIZE; ix++){
#pragma HLS PIPELINE II=1
              bufo[iy][ix] = 0;
            }
          }

          for(int ic = 0; ic < c; ic++){
read1:
            for(int fy = 0; fy < kh; fy++){
              for(int fx = 0; fx < kw; fx++){
#pragma HLS PIPELINE II=1
                bufw[fy][fx] = weight[i * c * kh*kw + ic * kh*kw + fy * kw + fx];
              }
            }
read2:
            for(int ih = 0; ih < chunk_size_y + kh; ih++){
              for(int iw = 0; iw < chunk_size_x + kw; iw++){
#pragma HLS PIPELINE II=1
                bufi[ih][iw] = input[n*c*h*w + ic*h*w + (y+ih)*w + x+iw]; 
              }
            }
madd:
            for(int iy = 0; iy < chunk_size_y; iy++){
              for(int ix = 0; ix < chunk_size_x; ix++){
#pragma HLS PIPELINE II=1
                int sum = 0;
                for(int fy = 0; fy < kh; fy++){
                  for(int fx = 0; fx < kw; fx++){
                    sum += bufi[iy + fy][ix + fx] * bufw[fy][fx];
                  }
                }
                bufo[iy][ix] += sum;
              }
            }
          }	

write:
          for(int oy = 0; oy < chunk_size_y; oy++){
            for(int ox = 0; ox < chunk_size_x; ox++){
#pragma HLS PIPELINE II=1
              output[n*oc*oh*ow + i*oh*ow + (y+oy)*ow + x+ox] = bufo[oy][ox] + bias[i];
            }
          }
        }
      }
    }
  } 
}
}

