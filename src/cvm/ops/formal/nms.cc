#include <iostream>
#include <vector>
#include <algorithm>
#include <stdint.h>
#include <string.h>

#include "../common.h"
#include "ops.h"

namespace cvm {
namespace runtime{

int64_t iou(const int32_t *rect1, const int32_t *rect2, const int32_t format){
    int32_t x1_min = format == FORMAT_CORNER ? rect1[0] : rect1[0] - rect1[2]/2;
    int32_t y1_min = format == FORMAT_CORNER ? rect1[1] : rect1[1] - rect1[3]/2;
    int32_t x1_max = format == FORMAT_CORNER ? rect1[2] : x1_min + rect1[2];
    int32_t y1_max = format == FORMAT_CORNER ? rect1[3] : y1_min + rect1[3];

    int32_t x2_min = format == FORMAT_CORNER ? rect2[0] : rect2[0] - rect2[2]/2;
    int32_t y2_min = format == FORMAT_CORNER ? rect2[1] : rect2[1] - rect2[3]/2;
    int32_t x2_max = format == FORMAT_CORNER ? rect2[2] : x2_min + rect2[2];
    int32_t y2_max = format == FORMAT_CORNER ? rect2[3] : y2_min + rect2[3];

    //x1,x2,y1,y2 precision <= 30
    //sum_arrea precision<=63
    int64_t sum_area = static_cast<int64_t>(x1_max-x1_min) * (y1_max-y1_min) + static_cast<int64_t>(x2_max-x2_min) * (y2_max-y2_min);
    if(sum_area <= 0){
        return 0;
    }

    //w,h precision <= 31
    int32_t w = std::max(0, std::min(x1_max, x2_max) - std::max(x1_min, x2_min));
    int32_t h = std::max(0, std::min(y1_max, y2_max) - std::max(y1_min, y2_min));
    //overlap_area precision <= 62
    int64_t overlap_area = static_cast<int64_t>(h)*w;
    //tmp precision <= 63
    int64_t tmp = (sum_area - overlap_area);
    if(tmp <= 0){
        return 0;
    }
    int64_t max64 = ((uint64_t)1 << 63) - 1;
    if(max64 / 100 < overlap_area){
        tmp /= 100;
    }else{
        overlap_area *= 100;
    }
    int64_t ret = (overlap_area / tmp);
    return ret;
}

void get_valid_count(const int32_t *x_data, int32_t *y_data, int32_t *valid_count_data, const int32_t batchs, const int32_t n, const int32_t k, const int32_t score_threshold){
  for(int32_t i = 0; i < batchs; i++){
      int32_t y_index = 0;
      const int32_t *input = x_data + i * n * k;
      int32_t *output = y_data + i * n * k;
      for(int32_t j = 0; j < n; j++){
          const int32_t *row = input + j * k;
          if(row[1] > score_threshold){
              memcpy(&output[y_index * k], row, k * sizeof(int32_t));
              y_index += 1;
          }
      }
      valid_count_data[i] = y_index;
      if(y_index < n){
          memset(&output[y_index * k], -1, (n-y_index) * k * sizeof(int32_t));
      }
  }
}

CVM_REGISTER_GLOBAL("cvm.runtime.cvm.get_valid_counts")
.set_body([](cvm::runtime::CVMArgs args, cvm::runtime::CVMRetValue *rv){
    DLTensor *x = args[0];
    DLTensor *valid_count = args[1];
    DLTensor *y = args[2];
    void* _attr = args[3];
    auto *attr = static_cast<cvm::NodeAttrs*>(_attr);
    auto &param = cvm::get<cvm::top::GetValidCountsParam>(attr->parsed);

    int32_t score_threshold = param.score_threshold; 

    int32_t batches = x->shape[0];
    int32_t n = x->shape[1];
    int32_t k = x->shape[2];

    int32_t *x_data = static_cast<int32_t*>(x->data);
    int32_t *valid_count_data = static_cast<int32_t*>(valid_count->data);
    int32_t *y_data = static_cast<int32_t*>(y->data);

    get_valid_count(x_data, y_data, valid_count_data, batches, n, k, score_threshold);
});

int non_max_suppression(int32_t *x_data, const int32_t *valid_count_data, int32_t *y_data, const int32_t batchs, const int32_t n, const int32_t k,
        const int32_t max_output_size, const int32_t iou_threshold, const int32_t topk,
        const int32_t coord_start, const int32_t score_index, const int32_t id_index, const bool force_suppress){
  for(int32_t b = 0; b < batchs; b++){
      int32_t vc = valid_count_data[b];
      int32_t *x_batch = x_data + b * n * k;
      int32_t *y_batch = y_data + b * n * k;
      // TODO(zkh): avoid throw exception.
      if(vc > n){
          return -1;
      }
      if(vc <= 0){
          memset(y_batch, -1, n * k * sizeof(int32_t));
          return 0;
      }

      if(iou_threshold <= 0){
          memcpy(y_batch, x_batch, vc * k * sizeof(int32_t));
          memset(y_batch + vc * n * k, -1, (n-vc)*k * sizeof(int32_t));
      }else{
        std::vector<int32_t*> rows(vc);
        for (int i = 0; i < vc; i++) {
          rows[i] = x_batch + i * k;
        }
        std::stable_sort(rows.begin(), rows.end(), [score_index](const int32_t *a, const int32_t *b) -> bool{
              return a[score_index] > b[score_index];
            });
        if(topk > 0 && topk < vc){
          for(int i = 0; i < vc - topk; i++){
            memset(rows[i+topk], -1, k * sizeof(int32_t));
          }
        }

        std::vector<bool> removed(n, false);
        int need_keep = ((topk >= 0 && topk < vc) ? topk : vc);
        for(int i = need_keep; i < vc; i++){
          removed[i] = true;
        }

        int32_t y_index = 0;
        for(int i = 0; i < need_keep; i++){
          int32_t *row1 = rows[i];

          if(removed[i] == false && row1[0] >= 0){
            memcpy(&y_batch[y_index*k], row1, k*sizeof(int32_t));
            y_index += 1;
          }
          for(int j = i+1; j < need_keep && !removed[i] && rows[j][0] >= 0; j++){
            int32_t* row2 = rows[j];
            if(force_suppress || (id_index < 0 || row1[id_index] == row2[id_index])){
              int64_t iou_ret = iou(row1+coord_start, row2+coord_start, FORMAT_CORNER);
              if(iou_ret >= iou_threshold){
                removed[j] = true;
              }
            }
          }
        }
        if(y_index < n){
          memset(&y_batch[y_index*k], -1, (n - y_index) * k * sizeof(int32_t));
        }
      }
      if(max_output_size > 0){
        int j = 0;
        for(int i = 0; i < vc; i++){
          if(y_batch[i*k] >= 0){
            if(j == max_output_size){
              memset(y_batch + i * k, -1, k * sizeof(int32_t));
            }else{
              j += 1;
            }
          }
        }
      }
  }
  return 0;
}

CVM_REGISTER_GLOBAL("cvm.runtime.cvm.non_max_suppression")
.set_body([](cvm::runtime::CVMArgs args, cvm::runtime::CVMRetValue *rv){
#if true
    // parse the input, output and params
    auto X = CVMArg2Data<int32_t>(args[0]);
    auto valid_count = CVMArg2Data<int32_t>(args[1]);
    auto Y = CVMArg2Data<int32_t>(args[2]);
    auto params = CVMArg2Attr<top::NonMaximumSuppressionParam>(args[3]);

    // X's shape must be (B, N, K), K = 6
    auto x_shape = CVMArgShape(args[0]);
    int32_t B = x_shape[0];
    int32_t N = x_shape[1];
    int32_t K = x_shape[2];

    for (int32_t b = 0; b < B; ++b) {
      int32_t T = std::max(std::min(B, valid_count[b]), 0);
      std::vector<int32_t*> R(T);
      for (int i = 0; i < T; i++) {
        R[i] = X + b * B * K + i * K;
      }
      std::stable_sort(R.begin(), R.end(), 
        [](const int32_t* a, const int32_t* b) -> bool {
            return a[1] > b[1];
        });

      int32_t n = 0; // current y index, stand for dynamic union U's size
      int32_t idx = 0; // current x index

      // n \in [0, min{T, MOS, card{U}})
      int32_t n_max = T;
      if (params.max_output_size >= 0)
        n_max = std::min(n_max, params.max_output_size);
      int32_t idx_max = T;
      if (params.top_k >= 0)
        idx_max = std::min(idx_max, params.top_k);

      int32_t *y_batch = Y + b * B * K; // temporary variable
      for (; n < n_max && idx < idx_max; ++idx) {
        // check current bbox whether satisfies U's condition: bbox non-overlap
        bool ignored = false;
        for (int32_t i = 0; i < n; ++i) {
          int64_t iou_ret = 0;
          if (params.force_suppress || y_batch[i * K + 0] == R[idx][0])
            iou_ret = iou(y_batch + i * K + 2, R[idx]+2, FORMAT_CORNER);
          if (iou_ret >= params.iou_threshold) {
            ignored = true;
            break;
          }
        }

        if (!ignored) { // satisfied: copy element into U and add U's size
          memcpy(y_batch + n * K, R[idx], K * sizeof(int32_t));
          n += 1;
        }
      }

      // others, memset -1
      memset(y_batch + n * K, -1, (N - n) * K * sizeof(int32_t));
    }

#else

    DLTensor *x = args[0];
    int32_t *x_data = static_cast<int32_t*>(x->data);
    DLTensor *vc = args[1];
    int32_t *vc_data = static_cast<int32_t*>(vc->data);
    DLTensor *y = args[2];
    int32_t *y_data = static_cast<int32_t*>(y->data);

    void* _attr = args[3];
    auto *attr = static_cast<cvm::NodeAttrs*>(_attr);
    auto &param = cvm::get<cvm::top::NonMaximumSuppressionParam>(attr->parsed);

    int32_t max_output_size = param.max_output_size;
    int32_t iou_threshold = param.iou_threshold;
    int32_t topk = param.top_k;
    int32_t coord_start = param.coord_start;
    int32_t score_index = param.score_index;
    int32_t id_index = param.id_index;
    bool force_suppress = param.force_suppress;

    // X's shape (B, N, K)
    int32_t batch = x->shape[0];
    int32_t n = x->shape[1];
    int32_t k = x->shape[2];

    int ret = non_max_suppression(
            x_data, vc_data, y_data, batch, n, k,
            max_output_size, iou_threshold, topk, coord_start, score_index, id_index, force_suppress);
    VERIFY(ret >= 0);

    print_to_file(x, "nms_x.txt", true);
    print_to_file(vc, "nms_valid_count.txt", true);
    print_to_file(y, "nms.txt");
#endif

});

}
}
