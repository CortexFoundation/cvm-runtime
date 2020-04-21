#include <CL/opencl.h>
#include <iostream>
#include <string>
#include <vector>
#include <iostream>
#include <fstream>
#include "util.hpp"
using namespace std;

void gemm_cpu(const int* ext_space, const int *bias, int* C,
	const int M, const int K, const int N){
  const int *A = ext_space;
  const int *B = ext_space + M*K;
  for(int i = 0; i < M; i++){
    for(int j = 0; j < N; j++){
      int sum = 0;
      for(int k = 0; k < K; k++){
        sum += A[i*K + k] * B[k * N + j]; 
      }
      C[i*N+j] = sum + bias[i];
    }
  }
}

void gemm_fpga(cl_context context, cl_command_queue queue, cl_program program){
  cl_int code;
  const int M = 64;
  const int K = 64;
  const int N = 64;
  int *ext_space = new int[M*K+K*N];
  int *C = new int[M*N];
  int *bias = new int[M];

  for(int i = 0; i < M*K+K*N; i++){
    ext_space[i] = i % 127;
  }
  for(int i = 0; i < M; i++){
    bias[i] = i;
  }

  gemm_cpu(ext_space, bias, C, M, K, N);

  cl_mem bufA = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(int)*M*K, NULL, &code);
  cl_mem bufB = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(int)*K*N, NULL, &code);
  cl_mem bufb = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(int)*M, NULL, &code);

  clEnqueueWriteBuffer(queue, bufA, CL_TRUE, 0, sizeof(int)*M*K, ext_space, 0, nullptr, nullptr);
  clEnqueueWriteBuffer(queue, bufB, CL_TRUE, 0, sizeof(int)*N*K, ext_space + M*K, 0, nullptr, nullptr);
  clEnqueueWriteBuffer(queue, bufb, CL_TRUE, 0, sizeof(int)*M, bias, 0, nullptr, nullptr);

  cl_mem bufC = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(int)*M*N, NULL, &code);
  cl_kernel kernel = bias != NULL ? clCreateKernel(program, "gemm_bias", &code) : clCreateKernel(program, "gemm", &code);
  int index = 0;
  clSetKernelArg(kernel, index++, sizeof(cl_mem), (void*)&bufA);
  clSetKernelArg(kernel, index++, sizeof(cl_mem), (void*)&bufB);
  if(bias != NULL)
    clSetKernelArg(kernel, index++, sizeof(cl_mem), (void*)&bufb);
  clSetKernelArg(kernel, index++, sizeof(cl_mem), (void*)&bufC);
  clSetKernelArg(kernel, index++, sizeof(int), (void*)&M);
  clSetKernelArg(kernel, index++, sizeof(int), (void*)&K);
  clSetKernelArg(kernel, index++, sizeof(int), (void*)&N);
  clEnqueueTask(queue, kernel, 0, NULL, NULL);

  unsigned int C2[M*N];
  clEnqueueReadBuffer(queue, bufC, CL_TRUE, 0, sizeof(int)*M*N, C2, 0, nullptr, nullptr); 
  for(int i = 0; i< M*N; i++){
    if(C2[i] != C[i]){
      cout << "failed: " << i << ":" << C[i] << " " << C2[i]<< endl;
      return;
    }
  }
  cout << "success..." << endl;

  //release
}

int main(){
  cl_uint ret_size;
  cl_int code = clGetPlatformIDs(0, nullptr, &ret_size);
  std::vector<cl_platform_id> platform_ids;
  platform_ids.resize(ret_size);
  clGetPlatformIDs(ret_size, &platform_ids[0], nullptr);
  for(int i = 0; i < ret_size; i++){
    size_t name_size;
    clGetPlatformInfo(platform_ids[i], CL_PLATFORM_NAME, 0, nullptr, &name_size);
    std::string name;
    name.resize(name_size);
    clGetPlatformInfo(platform_ids[i], CL_PLATFORM_NAME, name_size, &name[0], nullptr);
    cout << "find platform : " << name << endl;
    if(name.find("Xilinx") != string::npos){
      std::vector<cl_device_id> devices;
      cl_uint device_num = 0;
      cl_int code = clGetDeviceIDs(platform_ids[i], CL_DEVICE_TYPE_ACCELERATOR, 0, nullptr, &device_num);
      devices.resize(device_num);
      clGetDeviceIDs(platform_ids[i], CL_DEVICE_TYPE_ACCELERATOR, device_num, &devices[0], nullptr);
      cl_context context = clCreateContext(nullptr, devices.size(), &devices[0], nullptr, nullptr, &code); 
      cl_command_queue queue = clCreateCommandQueue(context, devices[0], 0, &code);
      vector<unsigned char> binary = read_binary_file("ops.sw_emu.xclbin");
      const unsigned char *binary_data = binary.data();
      size_t size = binary.size();
      int binary_status;
      cl_program program = clCreateProgramWithBinary(context, 1, &devices[0],&size, (const unsigned char**)&binary_data, &binary_status, &code); 

      gemm_fpga(context, queue, program);
      return 0;
    }
  }
}
