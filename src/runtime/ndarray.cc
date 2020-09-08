/*!
 *  Copyright (c) 2017 by Contributors
 * \file ndarray.cc
 * \brief NDArray container infratructure.
 */
#include <utils/logging.h>
#include <cvm/errors.h>
#include <cvm/runtime/ndarray.h>
#include <cvm/runtime/c_runtime_api.h>
#include <cvm/runtime/device_api.h>
#include <cvm/runtime/forward.h>
#include <cvm/tuple.h>
#include <cvm/runtime/base.h>
#include <cvm/dlpack.h>

// deleter for arrays used by DLPack exporter
extern "C" void NDArrayDLPackDeleter(DLManagedTensor* tensor);

namespace cvm {
namespace runtime {

inline void VerifyDataType(DLDataType dtype) {
  CHECK_GE(dtype.lanes, 1);
  if (dtype.code == kDLFloat) {
    CHECK_EQ(dtype.bits % 8, 0);
  } else {
    // allow uint1 as a special flag for bool.
    if (dtype.bits == 1 && dtype.code == kDLUInt) return;
    CHECK_EQ(dtype.bits % 8, 0);
  }
  CHECK_EQ(dtype.bits & (dtype.bits - 1), 0);
}

inline size_t GetDataAlignment(const DLTensor& arr) {
  size_t align = (arr.dtype.bits / 8) * arr.dtype.lanes;
  if (align < kAllocAlignment) return kAllocAlignment;
  return align;
}

struct NDArray::Internal {
  // Default deleter for the container
  static void DefaultDeleter(NDArray::Container* ptr) {
    using cvm::runtime::NDArray;
    if (ptr->manager_ctx != nullptr) {
      static_cast<NDArray::Container*>(ptr->manager_ctx)->DecRef();
    } else if (ptr->dl_tensor.data != nullptr) {
      cvm::runtime::DeviceAPI::Get(ptr->dl_tensor.ctx)->FreeDataSpace(
          ptr->dl_tensor.ctx, ptr->dl_tensor.data);
    }
    delete ptr;
  }
  // Deleter for NDArray converted from DLPack
  // This is used from data which is passed from external DLPack(DLManagedTensor)
  // that are not allocated inside of CVM.
  // This enables us to create NDArray from memory allocated by other
  // frameworks that are DLPack compatible
  static void DLPackDeleter(NDArray::Container* ptr) {
    DLManagedTensor* tensor = static_cast<DLManagedTensor*>(ptr->manager_ctx);
    if (tensor->deleter != nullptr) {
      (*tensor->deleter)(tensor);
    }
    delete ptr;
  }
  // Local create function which allocates tensor metadata
  // but does not allocate space for the data.
  static NDArray Create(std::vector<int64_t> shape,
                        DLDataType dtype,
                        DLContext ctx) {
    VerifyDataType(dtype);
    // critical zone
    NDArray::Container* data = new NDArray::Container();
    data->deleter = DefaultDeleter;
    NDArray ret(data);
    ret.data_ = data;
    // RAII now in effect
    // setup shape
    data->shape_ = std::move(shape);
    data->dl_tensor.shape = utils::BeginPtr(data->shape_);
    data->dl_tensor.ndim = static_cast<int>(data->shape_.size());
    // setup dtype
    data->dl_tensor.dtype = dtype;
    // setup ctx
    data->dl_tensor.ctx = ctx;
    return ret;
  }
  // Container to DLManagedTensor
  static DLManagedTensor* ToDLPack(NDArray::Container* from) {
    CHECK(from != nullptr);
    DLManagedTensor* ret = new DLManagedTensor();
    ret->dl_tensor = from->dl_tensor;
    ret->manager_ctx = from;
    from->IncRef();
    ret->deleter = NDArrayDLPackDeleter;
    return ret;
  }
};

NDArray NDArray::CreateView(std::vector<int64_t> shape,
                            DLDataType dtype) {
  CHECK(data_ != nullptr);
  CHECK(data_->dl_tensor.strides == nullptr)
      << "Can only create view for compact tensor";
  NDArray ret = Internal::Create(shape, dtype, data_->dl_tensor.ctx);
  ret.data_->dl_tensor.byte_offset =
      this->data_->dl_tensor.byte_offset;
  size_t curr_size = GetDataSize(this->data_->dl_tensor);
  size_t view_size = GetDataSize(ret.data_->dl_tensor);
  CHECK_LE(view_size, curr_size)
      << "Tries to create a view that has bigger memory than current one";
  // increase ref count
  this->data_->IncRef();
  ret.data_->manager_ctx = this->data_;
  ret.data_->dl_tensor.data = this->data_->dl_tensor.data;
  return ret;
}

DLManagedTensor* NDArray::ToDLPack() const {
  return Internal::ToDLPack(data_);
}

DLTensor* NDArray::MoveAsDLTensor() {
  DLTensor* tensor = const_cast<DLTensor*>(operator->());
  CHECK(reinterpret_cast<DLTensor*>(data_) == tensor);
  data_ = nullptr;
  return tensor;
}

NDArray NDArray::Empty(std::vector<int64_t> shape,
                       DLDataType dtype,
                       DLContext ctx) {
  NDArray ret = Internal::Create(shape, dtype, ctx);
  // setup memory content
  size_t size = GetDataSize(ret.data_->dl_tensor);
  size_t alignment = GetDataAlignment(ret.data_->dl_tensor);
  ret.data_->dl_tensor.data =
      DeviceAPI::Get(ret->ctx)->AllocDataSpace(
          ret->ctx, size, alignment, ret->dtype);
  return ret;
}

NDArray NDArray::FromDLPack(DLManagedTensor* tensor) {
  NDArray::Container* data = new NDArray::Container();
  data->deleter = Internal::DLPackDeleter;
  data->manager_ctx = tensor;
  data->dl_tensor = tensor->dl_tensor;
  return NDArray(data);
}

void NDArray::CopyFromTo(DLTensor* from,
                         DLTensor* to,
                         CVMStreamHandle stream) {
  size_t from_size = GetDataSize(*from);
  size_t to_size = GetDataSize(*to);
  CHECK_EQ(from_size, to_size)
    << "CVMArrayCopyFromTo: The size must exactly match";

  CHECK(from->ctx.device_type == to->ctx.device_type
        || from->ctx.device_type == kDLCPU
        || to->ctx.device_type == kDLCPU)
    << "Can not copy across different ctx types directly";

  // Use the context that is *not* a cpu context to get the correct device
  // api manager.
  CVMContext ctx = from->ctx.device_type != kDLCPU ? from->ctx : to->ctx;

  DeviceAPI::Get(ctx)->CopyDataFromTo(
    from->data, static_cast<size_t>(from->byte_offset),
    to->data, static_cast<size_t>(to->byte_offset),
    from_size, from->ctx, to->ctx, from->dtype, stream);
}

template <typename T>
void NDArray::CPUFill(T value) {
  VERIFY(operator->()->ctx.device_type == kDLCPU)
      << "CPUFill() can only be called for a CPU tensor, but got "
      << operator->()->ctx.device_id << "\n";
  T* data = static_cast<T*>(operator->()->data);
  for (Indices idx(TShape(data_->shape_)); !idx.End(); idx++) {
    data[idx.Index()] = value;
  }
}

}  // namespace runtime
}  // namespace cvm

using namespace cvm::runtime;

void NDArrayDLPackDeleter(DLManagedTensor* tensor) {
  static_cast<NDArray::Container*>(tensor->manager_ctx)->DecRef();
  delete tensor;
}

int CVMArrayAlloc(const cvm_index_t* shape,
                  int ndim,
                  int dtype_code,
                  int dtype_bits,
                  int dtype_lanes,
                  int device_type,
                  int device_id,
                  CVMArrayHandle* out) {
  API_BEGIN();
  DLDataType dtype;
  dtype.code = static_cast<uint8_t>(dtype_code);
  dtype.bits = static_cast<uint8_t>(dtype_bits);
  dtype.lanes = static_cast<uint16_t>(dtype_lanes);
  DLContext ctx;
  ctx.device_type = static_cast<DLDeviceType>(device_type);
  ctx.device_id = device_id;
  *out = NDArray::Empty(
      std::vector<int64_t>(shape, shape+ndim),
      dtype, ctx)
    .MoveAsDLTensor();
  API_END();
}

int CVMArrayFree(CVMArrayHandle handle) {
  API_BEGIN();
  reinterpret_cast<NDArray::Container*>(handle)->DecRef();
  API_END();
}

int CVMArrayCopyFromTo(CVMArrayHandle from,
                       CVMArrayHandle to,
                       CVMStreamHandle stream) {
  API_BEGIN();
  NDArray::CopyFromTo(from, to, stream);
  API_END();
}

int CVMArrayCopyFromBytes(CVMArrayHandle handle,
                          void* data,
                          size_t nbytes) {
  API_BEGIN();
  CVMContext cpu_ctx;
  cpu_ctx.device_type = kDLCPU;
  cpu_ctx.device_id = 0;
  size_t arr_size = GetDataSize(*handle);
  CHECK_EQ(arr_size, nbytes)
      << "CVMArrayCopyFromBytes: size mismatch";
  DeviceAPI::Get(handle->ctx)->CopyDataFromTo(
      data, 0,
      handle->data, static_cast<size_t>(handle->byte_offset),
      nbytes, cpu_ctx, handle->ctx, handle->dtype, nullptr);
  API_END();
}

int CVMArrayCopyToBytes(CVMArrayHandle handle,
                        void* data,
                        size_t nbytes) {
  API_BEGIN();
  CVMContext cpu_ctx;
  cpu_ctx.device_type = kDLCPU;
  cpu_ctx.device_id = 0;
  size_t arr_size = GetDataSize(*handle);
  CHECK_EQ(arr_size, nbytes)
      << "CVMArrayCopyToBytes: size mismatch";
  DeviceAPI::Get(handle->ctx)->CopyDataFromTo(
      handle->data, static_cast<size_t>(handle->byte_offset),
      data, 0,
      nbytes, handle->ctx, cpu_ctx, handle->dtype, nullptr);
  API_END();
}

int CVMAssignNDScalar(CVMArrayHandle target, int* indices, double value) {
  API_BEGIN();
  // TODO: is it ok to modify without checking reference count?
  auto container =
      reinterpret_cast<cvm::runtime::NDArray::Container*>(target);
  int ndim = target->ndim;
  int* starts = indices;
  int* ends = indices + ndim;
  int* steps = indices + ndim * 2;
  int* sizes = indices + ndim * 3;
  cvm::TShape targetShape(target->shape, target->shape + ndim);
  //std::cout << "assigning data to:" << std::endl;
  //for (int i = 0; i < ndim; i++) {
  //  std::cout << starts[i] << ' ' << ends[i] << " " << steps[i] << " "
  //            << sizes[i] << std::endl;
  //}
  Indices assignIdx(cvm::Tuple<cvm::dim_t>(sizes, sizes + ndim));
  Indices targetIdx(targetShape);
  targetIdx.CopyIndicesFrom(std::vector<int64_t>(starts, starts + ndim));

  CVM_TYPE_SWITCH(target->dtype, DType, {
    DType typeValue = static_cast<DType>(value);
    for (; !assignIdx.End(); ++assignIdx) {
      Indices tmpIdx(targetIdx);
      for (int i = 0; i < ndim; i++) {
        tmpIdx.Ref(i) += assignIdx[i] * steps[i];
      }
      DeviceAPI::Get(target->ctx)
          ->CopyDataFromTo(&typeValue, 0, target->data,
                           tmpIdx.Index() * sizeof(DType), sizeof(DType),
                           DLContext{kDLCPU, 0}, target->ctx,
                           CVMType{kDLFloat, 64, 1}, nullptr);
    }
  })
  API_END();
}

int CVMFullND(CVMArrayHandle target, double value) {
  API_BEGIN();
  
  // TODO: if target is a CPU tensor, call CPUFill directly.
  //if (target->ctx.device_id == kDLCPU) {
  //  NDArray tmp(target);
  //}

  std::vector<int64_t> targetShape(target->shape, target->shape + target->ndim);
  Indices idx(cvm::TShape(targetShape));
  CVM_TYPE_SWITCH(target->dtype, DType, {
    NDArray tmp = NDArray::Empty(targetShape, target->dtype, DLContext{kDLCPU, 0});
    DType typeValue = static_cast<DType>(value);
    tmp.CPUFill<DType>(typeValue);
    NDArray::CopyFromTo(tmp.operator->(), target);
  })

  API_END();
}