#include <cvm/c_api.h>
#include <cvm/model.h>
#include <cvm/node.h>
#include <cvm/op.h>
#include <cvm/op_attr_types.h>
#include <cvm/runtime/c_runtime_api.h>
#include <cvm/runtime/device_api.h>
#include <cvm/runtime/ndarray.h>
#include <cvm/runtime/packed_func.h>
#include <cvm/runtime/registry.h>
#include <cvm/runtime/serializer.h>
#include <dirent.h>
#include <omp.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#include <fstream>
#include <iostream>
#include <numeric>
#include <thread>
#include <map>

#include "npy.hpp"

using std::string;
using std::vector;
using cvm::runtime::NDArray;

typedef std::map<string, NDArray> ParamDict;

CVMByteArray save_param_dict(const std::map<string, NDArray>& data) {
  vector<void*> passed;
  for (auto it = data.begin(); it != data.end(); ++it) {
    passed.push_back(const_cast<char*>(it->first.c_str()));
    passed.push_back(const_cast<DLTensor*>(it->second.operator->()));
  }
  CVMByteArray ret;
  CHECK(CVMSaveParamsDict(&(passed[0]), passed.size(), &ret) == 0)
      << "CVMSaveParamDict failed.\n";
  return ret;
}

std::map<string, NDArray> load_param_dict(CVMByteArray data) {
  std::map<string, NDArray> ret;
  int retNum;
  char** retName;
  void** retVal;
  int rv = CVMLoadParamsDict(data.data, data.size, &retNum, &retName, &retVal);
  CHECK(rv == 0) << "CVMLoadParamsDict failed\n";
  for (int i = 0; i < retNum; i++) {
    string name = retName[i];
    NDArray::Container* tensor = (NDArray::Container*)retVal[i];
    std::cout << "returned name[" << i << "] is " << name << retName[i] << std::endl
              << "tensor is ";
    cvm::runtime::printTensor(&tensor->dl_tensor);
    ret[name] = NDArray(tensor);
    tensor->DecRef();
  }
  CHECK(CVMDeleteLDPointer(retNum, retName, retVal) == 0)
      << "CVMDeleteLDPointer failed\n";
  return ret;
}

void test_SL_param_dict() {
  std::map<string, NDArray> data;
  NDArray a = NDArray::Empty(vector<int64_t>{4}, DLDataType{0, 32, 1},
                             DLContext{kDLCPU, 0});
  NDArray b = NDArray::Empty(vector<int64_t>{4, 2}, DLDataType{0, 32, 1},
                             DLContext{kDLCPU, 0});
  for (int i = 0; i < 4; i++) {
    static_cast<int32_t*>(const_cast<DLTensor*>(a.operator->())->data)[i] =
        i + 1;
  }
  for (int i = 0; i < 8; i++) {
    static_cast<int32_t*>(const_cast<DLTensor*>(b.operator->())->data)[i] =
        i + 5;
  }
  data["a"] = a;
  data["b"] = b;
  CVMByteArray saved = save_param_dict(data);
  std::map<string, NDArray> loaded = load_param_dict(saved);
}

int main() {
  test_SL_param_dict();
	return 0; 
}