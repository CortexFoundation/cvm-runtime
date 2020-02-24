#ifndef PROFILING_H
#define PROFILING_H

namespace cvm{
namespace runtime {
#define CVM_PROFILING
extern double transpose_int8_avx256_transpose_cnt;
extern double transpose_int8_avx256_gemm_cnt;
extern double im2col_cnt;
extern double cvm_op_dense_cnt;
extern double cvm_op_maxpool_cnt;
extern double cvm_op_concat_cnt;
extern double cvm_op_upsampling_cnt;
extern double cvm_op_inline_matmul_cnt;
extern double cvm_op_elemwise_cnt;
extern double cvm_op_chnwise_conv_cnt;
extern double cvm_op_chnwise_conv1x1_cnt;
extern double cvm_op_depthwise_conv_cnt;
extern double cvm_op_depthwise_conv1x1_cnt;
extern double cvm_op_clip_cnt;
extern double cvm_op_cvm_shift_cnt;
extern double cvm_op_broadcast_cnt;
extern double cvm_op_relu_cnt;

}
}
#endif
