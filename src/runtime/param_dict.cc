#include <cvm/runtime/c_runtime_api.h>
#include <cvm/dlpack.h>
#include <cvm/runtime/param_dict.h>
#include <cvm/runtime/ndarray.h>

#include <utils/memory_io.h>

namespace cvm {
namespace runtime {

/*! \brief Magic number for NDArray list file  */
constexpr uint64_t kCVMNDArrayListMagic = 0xF7E58D4F05049CB7;

void load_param_dict(
    utils::Stream* strm,
    std::vector<std::string>& names,
    std::vector<NDArray>& values) {
  uint64_t header, reserved;
  VERIFY(strm->Read(&header))
      << "Invalid parameters file format";
  VERIFY(header == kCVMNDArrayListMagic)
      << "Invalid parameters file format";
  VERIFY(strm->Read(&reserved))
      << "Invalid parameters file format";

  VERIFY(strm->Read(&names))
      << "Invalid parameters file format";
  uint64_t sz;
  strm->Read(&sz);
  size_t size = static_cast<size_t>(sz);
  VERIFY(size == names.size())
      << "Invalid parameters file format";

  values.resize(size);
  for (size_t i = 0; i < size; ++i) {
    values[i].Load(strm);
  }
}

void save_param_dict(
    utils::Stream* strm,
    std::vector<std::string> const& names,
    std::vector<DLTensor*> const& values) {
  uint64_t header = kCVMNDArrayListMagic, reserved = 0;
  strm->Write(header);
  strm->Write(reserved);
  strm->Write(names);
  {
    uint64_t sz = static_cast<uint64_t>(values.size());
    strm->Write(sz);
    for (size_t i = 0; i < sz; ++i) {
      SaveDLTensor(strm, values[i]);
    }
  }
}

}
}

typedef utils::ThreadLocalStore<std::vector<std::string>>
NamesLocalStore;
typedef utils::ThreadLocalStore<
  std::vector<cvm::runtime::NDArray>>
ArraysLocalStore;
typedef utils::ThreadLocalStore<std::vector<void*>>
RetLocalStore;

int CVMLoadParamsDict(
    const char* data, int datalen,
    int* retNum,
    void*** retNames, void*** retValues) {
  auto names = NamesLocalStore::Get();
  auto values = ArraysLocalStore::Get();
  auto retStore = RetLocalStore::Get();

  API_BEGIN();
  std::string dataBuffer(data, datalen);
  utils::MemoryStringStream strm(&dataBuffer);
  utils::Stream* fi = &strm;

  names->clear();
  values->clear();
  retStore->clear();

  load_param_dict(fi, *names, *values);

  for (auto &name: *names) {
    retStore->push_back(const_cast<char*>(name.c_str()));
  }
  for (auto &nd: *values) {
    retStore->push_back(nd.MoveAsDLTensor());
  }

  uint64_t sz = names->size();
  *retNum = sz;
  *retNames = retStore->data();
  *retValues = retStore->data() + sz;

  API_END();
}

int CVMSaveParamsDict(
    const void** params, int params_size,
    CVMByteArray* ret){
  API_BEGIN();

  CHECK_EQ(params_size % 2, 0u);
  size_t num_params = params_size / 2;

  std::vector<std::string> names;
  names.reserve(num_params);
  std::vector<DLTensor*> arrays;
  arrays.reserve(num_params);
  for (size_t i = 0; i < num_params * 2; i += 2) {
    names.emplace_back(std::string((char*)params[i]));
    arrays.emplace_back((DLTensor*)params[i+1]);
  }

  CVMRuntimeEntry* e = CVMAPIRuntimeStore::Get();
  utils::MemoryStringStream strm(&e->ret_str);
  cvm::runtime::save_param_dict(&strm, names, arrays);

  ret->data = e->ret_str.c_str();
  ret->size = e->ret_str.size();

  API_END();
}

