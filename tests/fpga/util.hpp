#ifndef TEST_UTIL_H
#define TEST_UTIL_H

#include <vector>
#include <string>
#include <fstream>
#include <sys/stat.h>
#include <unistd.h>
#include <CL/opencl.h>
#include <assert.h>

struct DLTensor {
  int* data;
  int *shape;
  int ndim;
  DLTensor(int ndim){
    this->ndim = ndim;
    shape = new int[ndim];
  }
};

int getSize(DLTensor *dl){
  int size = 1;
  for(int i = 0; i < dl->ndim; i++){
    size *= dl->shape[i]; 
  }
  return size;
}

#define MAX_DIM 6
template<typename T>
inline void get_cuda_shape(const T *ishape, const int dim, T *oshape){
  int shift = MAX_DIM - dim;
  for(int i = 0; i < MAX_DIM; i++){
    oshape[i] = 1;
    if(i >= shift){
      oshape[i] = ishape[i - shift];
    }
  }
}

inline std::vector<unsigned char> read_binary_file(const std::string& xclbin_file_name) {
    std::cout << "INFO: Reading " << xclbin_file_name << std::endl;

    if (access(xclbin_file_name.c_str(), R_OK) != 0) {
        printf("ERROR: %s xclbin not available please build\n",
               xclbin_file_name.c_str());
        exit(EXIT_FAILURE);
    }
    //Loading XCL Bin into char buffer
    //std::cout << "Loading: '" << xclbin_file_name.c_str() << "'\n";
    std::ifstream bin_file(xclbin_file_name.c_str(), std::ifstream::binary);
    bin_file.seekg(0, bin_file.end);
    auto nb = bin_file.tellg();
    bin_file.seekg(0, bin_file.beg);
    std::vector<unsigned char> buf;
    buf.resize(nb);
    bin_file.read(reinterpret_cast<char *>(buf.data()), nb);
    return buf;
}

cl_context context;
cl_command_queue queue;
cl_program program;

void init_opencl(const std::string& bin_file_name){
  cl_uint ret_size;
  cl_int code = clGetPlatformIDs(0, nullptr, &ret_size);
  std::vector<cl_platform_id> platform_ids;
  platform_ids.resize(ret_size);
  clGetPlatformIDs(ret_size, &platform_ids[0], nullptr);
  bool init_success = false;

  for(int i = 0; i < ret_size; i++){
    size_t name_size;
    clGetPlatformInfo(platform_ids[i], CL_PLATFORM_NAME, 0, nullptr, &name_size);
    std::string name;
    name.resize(name_size);
    clGetPlatformInfo(platform_ids[i], CL_PLATFORM_NAME, name_size, &name[0], nullptr);
    if(name.find("Xilinx") != std::string::npos){
      std::vector<cl_device_id> devices;
      cl_uint device_num = 0;
      cl_int code = clGetDeviceIDs(platform_ids[i], CL_DEVICE_TYPE_ACCELERATOR, 0, nullptr, &device_num);
      assert(code == CL_SUCCESS);
      devices.resize(device_num);
      clGetDeviceIDs(platform_ids[i], CL_DEVICE_TYPE_ACCELERATOR, device_num, &devices[0], nullptr);
      context = clCreateContext(nullptr, devices.size(), &devices[0], nullptr, nullptr, &code); 
      assert(code == CL_SUCCESS);
      queue = clCreateCommandQueue(context, devices[0], 0, &code);
      assert(code == CL_SUCCESS);
      std::vector<unsigned char> binary = read_binary_file(bin_file_name);
      const unsigned char *binary_data = binary.data();
      size_t size = binary.size();
      int binary_status;
      program = clCreateProgramWithBinary(context, 1, &devices[0],&size, (const unsigned char**)&binary_data, &binary_status, &code); 
      assert(code == CL_SUCCESS);
      init_success = true;
      break;
    }
  }
  assert(init_success);
}

template<typename T>
void verify(const T *a, const T *b, const int n){
  for(int i = 0; i < n; i++){
    if(a[i] != b[i]){
      std::cout << "verify failed: " << i << " : " << a[i] << ", " << b[i] << std::endl;
      return;
    }
  }

  std::cout << "success" << std::endl;
}
#endif
