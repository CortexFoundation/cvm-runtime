#define FORMAT_CORNER 1
#define FORMAT_CENTER 2

inline int max(int a, int b){
  if(a > b) return a;
  return b;
}
inline int min(int a, int b){
  if(a < b) return a;
  return b;
}
inline long long iou(const int *rect1, const int *rect2, const int format){
    int x1_min = format == FORMAT_CORNER ? rect1[0] : rect1[0] - rect1[2]/2;
    int y1_min = format == FORMAT_CORNER ? rect1[1] : rect1[1] - rect1[3]/2;
    int x1_max = format == FORMAT_CORNER ? rect1[2] : x1_min + rect1[2];
    int y1_max = format == FORMAT_CORNER ? rect1[3] : y1_min + rect1[3];

    int x2_min = format == FORMAT_CORNER ? rect2[0] : rect2[0] - rect2[2]/2;
    int y2_min = format == FORMAT_CORNER ? rect2[1] : rect2[1] - rect2[3]/2;
    int x2_max = format == FORMAT_CORNER ? rect2[2] : x2_min + rect2[2];
    int y2_max = format == FORMAT_CORNER ? rect2[3] : y2_min + rect2[3];

    //x1,x2,y1,y2 precision <= 30
    //sum_arrea precision<=63
    long long sum_area = (long long)(x1_max-x1_min) * (y1_max-y1_min) + 
        (long long)(x2_max-x2_min) * (y2_max-y2_min);
    if (sum_area <= 0) return 0;

    //w,h precision <= 31
    int w = max(0, min(x1_max, x2_max) - max(x1_min, x2_min));
    int h = max(0, min(y1_max, y2_max) - max(y1_min, y2_min));
    //overlap_area precision <= 62
    long long overlap_area = (long long)(h)*w;
    //tmp precision <= 63
    long long tmp = (sum_area - overlap_area);
    if (tmp <= 0) return 0;

    long long max64 = ((long long)1 << 63) - 1;
    if (max64 / 100 < overlap_area) { tmp /= 100; } 
    else { overlap_area *= 100; }

    return overlap_area / tmp;
}

extern "C"{
void non_max_suppression(const int*inputs, int *outputs, 
    const int n_max, const int p_max,
    const bool force_suppress, const int iou_threshold,
    const int N, const int K,
    const int i_offset, const int o_offset){
#pragma HLS INTERFACE m_axi port = inputs offset = slave bundle = gmem
#pragma HLS INTERFACE m_axi port = outputs offset = slave bundle = gmem
#pragma HLS INTERFACE s_axilite port = inputs bundle = control
#pragma HLS INTERFACE s_axilite port = outputs bundle = control
#pragma HLS INTERFACE s_axilite port = n_max bundle = control
#pragma HLS INTERFACE s_axilite port = p_max bundle = control
#pragma HLS INTERFACE s_axilite port = force_suppress bundle = control
#pragma HLS INTERFACE s_axilite port = iou_threshold bundle = control
#pragma HLS INTERFACE s_axilite port = N bundle = control
#pragma HLS INTERFACE s_axilite port = K bundle = control
#pragma HLS INTERFACE s_axilite port = i_offset bundle = control
#pragma HLS INTERFACE s_axilite port = o_offset bundle = control
#pragma HLS INTERFACE s_axilite port = return bundle = control
  for (int p = 0, n = 0; n < n_max && p < p_max; ++p) { // p \in [0, p_max)
    //if (R[p][0] < 0) continue; // R[b, p, 0] >= 0
    if(inputs[i_offset + p*K] < 0) continue;

    bool ignored = false; // iou(p, q) <= iou_threshold, \forall q in U.
    for (int i = 0; i < n; ++i) {
      if (force_suppress || outputs[o_offset + i*K+0] == inputs[i_offset + p*K]) {
        long long iou_ret = iou(outputs + o_offset +i*K+2, inputs + i_offset + p*K + 2, FORMAT_CORNER);
        if (iou_ret >= iou_threshold) {
          ignored = true;
          break;
        }
      }
    }

    if (!ignored) { // append U: copy corresponding element to Y.
      for(int k = 0; k < K; k++){
        outputs[o_offset + n*K + k] = inputs[i_offset + p*K + k];
      }
      ++n;
    }
  }
}
}
