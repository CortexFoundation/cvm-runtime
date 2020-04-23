int  reduce(int a, int b, const int type){
  if(type == 0){ // max
    return a < b ? b : a;
  }else { // sum
    return a + b;
  } 
}
extern "C"{
  void reduce_zero(const int *input, int *output, 
      const int n, const int type){
#pragma HLS INTERFACE m_axi port=input offset=slave bundle=gmem
#pragma HLS INTERFACE m_axi port=output offset=slave bundle=gmem
#pragma HLS INTERFACE s_axilite port=input bundle=control
#pragma HLS INTERFACE s_axilite port=output bundle=control
#pragma HLS INTERFACE s_axilite port=n bundle=control
#pragma HLS INTERFACE s_axilite port=type bundle=control
#pragma HLS INTERFACE s_axilite port = return bundle = control
  const int BSO = 256;
  const int BSI = 64;
  int bufI[BSI];
  int bufO[BSO];
  const int step = (n + 255) / 256;
  int s = 0;
  for(int i = 0; i < n; i += step, s++){
    int real_bso = step;
    if(i + step > n) real_bso = n - i;
    for(int j = 0; j < real_bso; j += BSI){ 
      int real_bsi = BSI;
      if(j + BSI > real_bso) real_bsi = real_bso - j;
read:
      for(int k = 0; k < real_bsi; k++){
        bufI[k] = input[i + j + k];
      }
reduce:
      int ret = bufI[0];
      if(j != 0) ret = reduce(ret, bufO[s], type);
      for(int k = 1; k < real_bsi; k++){
        ret = reduce(ret, bufI[k], type);
      }
      bufO[s] = ret;
    }
  }

  int ret = bufO[0];
  for(int i = 1; i < s; i++){
    ret = reduce(ret, bufO[i], type); 
  }
  output[0] = ret;
}
}
