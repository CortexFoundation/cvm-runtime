#ifndef OPENCL_OPS_H
#define OPENCL_OPS_H

#include <string>


const std::string kernel_str = R"(
  __kernel void elemwise_add(__global const int* a, __global const int* b, __global int *c, int n){
    int gid = get_global_id(0);
    if(gid < n){
      c[gid] = a[gid] + b[gid];
    }
  } 
)";

//static cl_program program;

const char* opencl_elemwise_add(int32_t *a, int32_t *b, int32_t *c, uint64_t n, int& error_code){
  
  printf("call opencl elemwise add...\n");
  return "";
}
#endif
