/*!
 *  Copyright (c) 2017 by Contributors
 * \file cvm/runtime/ndarray.h
 * \brief Abstract device memory management API
 */
#ifndef CVM_RUNTIME_NDARRAY_H_
#define CVM_RUNTIME_NDARRAY_H_

#include <atomic>
#include <vector>
#include <utility>
#include "c_runtime_api.h"
#include "serializer.h"
#include "../errors.h"

namespace cvm {
namespace runtime {
/*!
 * \brief Managed NDArray.
 *  The array is backed by reference counted blocks.
 */
class NDArray {
 public:
  // internal container type
  class Container;
  /*! \brief default constructor */
  NDArray() {}
  /*!
   * \brief cosntruct a NDArray that refers to data
   * \param data The data this NDArray refers to
   */
  explicit inline NDArray(Container* data);
  /*!
   * \brief copy constructor.
   *
   * It does not make a copy, but the reference count of the input NDArray is incremented
   *
   * \param other NDArray that shares internal data with the input NDArray.
   */
  inline NDArray(const NDArray& other);  // NOLINT(*)
  /*!
   * \brief move constructor
   * \param other The value to be moved
   */
  NDArray(NDArray&& other) // NOLINT(*)
      : data_(other.data_) {
    other.data_ = nullptr;
  }
  /*! \brief destructor */
  ~NDArray() {
    this->reset();
  }
  /*!
   * \brief Swap this array with another NDArray
   * \param other The other NDArray
   */
  void swap(NDArray& other) {  // NOLINT(*)
    std::swap(data_, other.data_);
  }
  /*!
   * \brief copy assignmemt
   * \param other The value to be assigned.
   * \return reference to self.
   */
  NDArray& operator=(const NDArray& other) {  // NOLINT(*)
    // copy-and-swap idiom
    NDArray(other).swap(*this);  // NOLINT(*)
    return *this;
  }
  /*!
   * \brief move assignmemt
   * \param other The value to be assigned.
   * \return reference to self.
   */
  NDArray& operator=(NDArray&& other) {  // NOLINT(*)
    // copy-and-swap idiom
    NDArray(std::move(other)).swap(*this); // NOLINT(*)
    return *this;
  }
  /*! \return If NDArray is defined */
  bool defined() const {
    return data_ != nullptr;
  }
  /*! \return If both NDArray reference the same container */
  bool same_as(const NDArray& other) const {
    return data_ == other.data_;
  }
  /*! \brief reset the content of NDArray to be nullptr */
  inline void reset();
  /*!
   * \return the reference counter
   * \note this number is approximate in multi-threaded setting.
   */
  inline int use_count() const;
  /*! \return Pointer to content of DLTensor */
  inline const DLTensor* operator->() const;
  inline DLTensor* operator->();
  /*!
   * \brief Copy data content from another array.
   * \param other The source array to be copied from.
   * \note The copy may happen asynchrously if it involves a GPU context.
   *       CVMSynchronize is necessary.
   */
  inline void CopyFrom(DLTensor* other);
  inline void CopyFrom(const NDArray& other);
  /*!
   * \brief Copy data content into another array.
   * \param other The source array to be copied from.
   * \note The copy may happen asynchrously if it involves a GPU context.
   *       CVMSynchronize is necessary.
   */
  inline void CopyTo(DLTensor* other) const;
  inline void CopyTo(const NDArray& other) const;
  /*!
   * \brief Copy the data to another context.
   * \param ctx The target context.
   * \return The array under another context.
   */
  inline NDArray CopyTo(const DLContext& ctx) const;
  /*!
   * \brief Can ONLY be called for a cpu tensor! Fill the tensor with 
   *        a scalar value
   * \param value The value to assign to the tensor
  */
  template<typename T>
  void CPUFill(T value);
  /*!
   * \brief Load NDArray from stream
   * \param stream The input data stream
   * \return Whether load is successful
   */
  inline bool Load(utils::Stream* stream);
  /*!
   * \brief Save NDArray to stream
   * \param stream The output data stream
   */
  inline void Save(utils::Stream* stream) const;
  /*!
   * \brief Create a NDArray that shares the data memory with the current one.
   * \param shape The shape of the new array.
   * \param dtype The data type of the new array.
   * \note The memory size of new array must be smaller than the current one.
   */
  CVM_DLL NDArray CreateView(
      std::vector<int64_t> shape, DLDataType dtype);
  /*!
   * \brief Create a reference view of NDArray that
   *  represents as DLManagedTensor.
   * \return A DLManagedTensor
   */
  CVM_DLL DLManagedTensor* ToDLPack() const;
  /*!
   * \brief Move the container back to front-end via C API.
   *  This marks the current container as null.
   *  This managed resource is moved to fron-end and
   *  the front end should take charge in managing them.
   *
   * \param ret_value The return container pointer.
   **/
  CVM_DLL DLTensor* MoveAsDLTensor();
  /*!
   * \brief Create an empty NDArray.
   * \param shape The shape of the new array.
   * \param dtype The data type of the new array.
   * \param ctx The context of the Array.
   * \return The created Array
   */
  CVM_DLL static NDArray Empty(std::vector<int64_t> shape,
                               DLDataType dtype,
                               DLContext ctx);
  /*!
   * \brief Create a NDArray backed by a dlpack tensor.
   *
   * This allows us to create a NDArray using the memory
   * allocated by an external deep learning framework
   * that is DLPack compatible.
   *
   * The memory is retained until the NDArray went out of scope.
   * \param tensor The DLPack tensor to copy from.
   * \return The created NDArray view.
   */
  CVM_DLL static NDArray FromDLPack(DLManagedTensor* tensor);
  /*!
   * \brief Function to copy data from one array to another.
   * \param from The source array.
   * \param to The target array.
   * \param stream The stream used in copy.
   */
  CVM_DLL static void CopyFromTo(
      DLTensor* from, DLTensor* to, CVMStreamHandle stream = nullptr);

  // internal namespace
  struct Internal;
 protected:
  /*! \brief Internal Data content */
  Container* data_{nullptr};
  // enable internal functions
  friend struct Internal;
  friend class CVMPODValue_;
  friend class CVMArgValue;
  friend class CVMRetValue;
  friend class CVMArgsSetter;
};

/*!
 * \brief The type trait indicates subclass of CVM's NDArray.
 *  For irrelavant classes, code = -1.
 *  For CVM NDArray itself, code = 0.
 *  All subclasses of NDArray should override code > 0.
 */
template<typename T>
struct array_type_info {
  /*! \brief the value of the traits */
  static const int code = -1;
};

// Overrides the type trait for cvm's NDArray.
template<>
struct array_type_info<NDArray> {
  static const int code = 0;
};

/*!
 * \brief Save a DLTensor to stream
 * \param strm The outpu stream
 * \param tensor The tensor to be saved.
 */
inline bool SaveDLTensor(utils::Stream* strm, const DLTensor* tensor);

/*!
 * \brief Reference counted Container object used to back NDArray.
 *
 *  This object is DLTensor compatible:
 *    the pointer to the NDArrayContainer can be directly
 *    interpreted as a DLTensor*
 *
 * \note do not use this function directly, use NDArray.
 */
class NDArray::Container {
 public:
  // NOTE: the first part of this structure is the same as
  // DLManagedTensor, note that, however, the deleter
  // is only called when the reference counter goes to 0
  /*!
   * \brief The corresponding dl_tensor field.
   * \note it is important that the first field is DLTensor
   *  So that this data structure is DLTensor compatible.
   *  The head ptr of this struct can be viewed as DLTensor*.
   */
  DLTensor dl_tensor;
  /*!
   * \brief addtional context, reserved for recycling
   * \note We can attach additional content here
   *  which the current container depend on
   *  (e.g. reference to original memory when creating views).
   */
  void* manager_ctx{nullptr};
  /*!
   * \brief Customized deleter
   *
   * \note The customized deleter is helpful to enable
   *  different ways of memory allocator that are not
   *  currently defined by the system.
   */
  void (*deleter)(Container* self) = nullptr;

 protected:
  friend class NDArray;
  friend class CVMPODValue_;
  friend class CVMArgValue;
  friend class CVMRetValue;
  friend class RPCWrappedFunc;
  /*!
   * \brief Type flag used to indicate subclass.
   *  Default value 0 means normal NDArray::Conatainer.
   *
   *  We can extend a more specialized NDArray::Container
   *  and use the array_type_code_ to indicate
   *  the specific array subclass.
   */
  int32_t array_type_code_{0};
  /*! \brief The internal reference counter */
  std::atomic<int> ref_counter_{0};
  /*!
   * \brief The shape container,
   *  can be used used for shape data.
   */
  std::vector<int64_t> shape_;

 public:
  /*! \brief default constructor */
  Container() {
    dl_tensor.data = nullptr;
    dl_tensor.ndim = 0;
    dl_tensor.shape = nullptr;
    dl_tensor.strides = nullptr;
    dl_tensor.byte_offset = 0;
  }
  /*! \brief developer function, increases reference counter */
  void IncRef() {
    ref_counter_.fetch_add(1, std::memory_order_relaxed);
  }
  /*! \brief developer function, decrease reference counter */
  void DecRef() {
    if (ref_counter_.fetch_sub(1, std::memory_order_release) == 1) {
      std::atomic_thread_fence(std::memory_order_acquire);
      if (this->deleter != nullptr) {
        (*this->deleter)(this);
      }
    }
  }
};

// implementations of inline functions
// the usages of functions are documented in place.
inline NDArray::NDArray(Container* data)
  : data_(data) {
  if (data != nullptr) {
    data_->IncRef();
  }
}

inline NDArray::NDArray(const NDArray& other)
  : data_(other.data_) {
  if (data_ != nullptr) {
    data_->IncRef();
  }
}

inline void NDArray::reset() {
  if (data_ != nullptr) {
    data_->DecRef();
    data_ = nullptr;
  }
}

/*! \brief return the size of data the DLTensor hold, in term of number of bytes
 *
 *  \param arr the input DLTensor
 *
 *  \return number of  bytes of data in the DLTensor.
 */
inline size_t GetDataSize(const DLTensor& arr) {
  size_t size = 1;
  for (cvm_index_t i = 0; i < arr.ndim; ++i) {
    size *= static_cast<size_t>(arr.shape[i]);
  }
  size *= (arr.dtype.bits * arr.dtype.lanes + 7) / 8;
  return size;
}

inline void NDArray::CopyFrom(DLTensor* other) {
  CHECK(data_ != nullptr);
  CopyFromTo(other, &(data_->dl_tensor));
}

inline void NDArray::CopyFrom(const NDArray& other) {
  CHECK(data_ != nullptr);
  CHECK(other.data_ != nullptr);
  CopyFromTo(&(other.data_->dl_tensor), &(data_->dl_tensor));
}

inline void NDArray::CopyTo(DLTensor* other) const {
  CHECK(data_ != nullptr);
  CopyFromTo(&(data_->dl_tensor), other);
}

inline void NDArray::CopyTo(const NDArray& other) const {
  CHECK(data_ != nullptr);
  CHECK(other.data_ != nullptr);
  CopyFromTo(&(data_->dl_tensor), &(other.data_->dl_tensor));
}

inline NDArray NDArray::CopyTo(const DLContext& ctx) const {
  CHECK(data_ != nullptr);
  const DLTensor* dptr = operator->();
  NDArray ret = Empty(std::vector<int64_t>(dptr->shape, dptr->shape + dptr->ndim),
                      dptr->dtype, ctx);
  this->CopyTo(ret);
  return ret;
}

inline int NDArray::use_count() const {
  if (data_ == nullptr) return 0;
  return data_->ref_counter_.load(std::memory_order_relaxed);
}

inline const DLTensor* NDArray::operator->() const {
  return &(data_->dl_tensor);
}

inline DLTensor* NDArray::operator->() { return &(data_->dl_tensor); }

inline void printDType(DLDataType dtype, std::string message) {
  std::cout << message << "code: " << (int)dtype.code
            << "\tbits: " << (int)dtype.bits
            << "\tlanes: " << (int)dtype.lanes << std::endl;
}

inline void printTensor(const DLTensor* tmp) {
  std::cout << "the tensor content: ndim: " << tmp->ndim << ", shape: ";
  int tmpsize = 1;
  for (int j = 0; j < tmp->ndim; j++) {
    std::cout << tmp->shape[j] << " ";
    tmpsize *= tmp->shape[j];
  }
  std::cout << std::endl;
  for (int j = 0; j < tmpsize; j++) {
    int64_t out_data = 123456789;
    switch (tmp->dtype.bits) {
      case 8:
        out_data = ((int8_t*)tmp->data)[j];
        break;
      case 32:
        out_data = ((int32_t*)tmp->data)[j];
        break;
      case 64:
        out_data = ((int64_t*)tmp->data)[j];
        break;
      default:
        break;
    }
    std::cout << out_data << " ";
  }
  std::cout << std::endl;
}


/*! \brief Magic number for NDArray file */
constexpr uint64_t kCVMNDArrayMagic = 0xDD5E40F096B4A13F;

inline bool SaveDLTensor(utils::Stream* strm,
                         DLTensor* tensor) {
  uint64_t header = kCVMNDArrayMagic, reserved = 0;
  strm->Write(header);
  strm->Write(reserved);
  // Always save data as CPU context
  //
  // Parameters that get serialized should be in CPU by default.
  // So even the array's context is GPU, it will be stored as CPU array.
  // This is used to prevent case when another user loads the parameters
  // back on machine that do not have GPU or related context.
  //
  // We can always do array.CopyTo(target_ctx) to get a corresponding
  // array in the target context.
  DLContext cpu_ctx;
  cpu_ctx.device_type = kDLCPU;
  cpu_ctx.device_id = 0;
  strm->Write(cpu_ctx);
  strm->Write(tensor->ndim);
  strm->Write(tensor->dtype);
  int ndim = tensor->ndim;
  strm->WriteArray(tensor->shape, ndim);
  int type_bytes = tensor->dtype.bits / 8;
  int64_t num_elems = 1;
  for (int i = 0; i < ndim; ++i) {
    num_elems *= tensor->shape[i];
  }
  int64_t data_byte_size = type_bytes * num_elems;
  strm->Write(data_byte_size);

  if (CVMUTIL_IO_NO_ENDIAN_SWAP &&
      tensor->ctx.device_type == kDLCPU &&
      tensor->strides == nullptr &&
      tensor->byte_offset == 0) {
    // quick path
    strm->Write(tensor->data, data_byte_size);
  } else {
    std::vector<uint8_t> bytes(data_byte_size);
    CHECK_EQ(CVMArrayCopyToBytes(
        tensor, utils::BeginPtr(bytes), data_byte_size), 0)
        << CVMGetLastError();
    if (!CVMUTIL_IO_NO_ENDIAN_SWAP) {
      utils::ByteSwap(utils::BeginPtr(bytes), type_bytes, num_elems);
    }
    strm->Write(utils::BeginPtr(bytes), data_byte_size);
  }
  return true;
}

inline void NDArray::Save(utils::Stream* strm) const {
  SaveDLTensor(strm, const_cast<DLTensor*>(operator->()));
}

inline bool NDArray::Load(utils::Stream* strm) {
  uint64_t header, reserved;
  CHECK(strm->Read(&header))
      << "Invalid DLTensor file format";
  CHECK(strm->Read(&reserved))
      << "Invalid DLTensor file format";
  CHECK(header == kCVMNDArrayMagic)
      << "Invalid DLTensor file format";

  DLContext ctx;
  int ndim;
  DLDataType dtype;
  CHECK(strm->Read(&ctx))
      << "Invalid DLTensor file format";
  CHECK(strm->Read(&ndim))
      << "Invalid DLTensor file format";
  CHECK(strm->Read(&dtype))
      << "Invalid DLTensor file format";
  // what the following code did is now done in CvmRuntime::SetInput
  //CHECK_EQ(ctx.device_type, kDLCPU)
  //    << "Invalid DLTensor context: can only save as CPU tensor";
  //std::cout << (int)dtype.code << "\t" << (int)dtype.bits << "\t" << (int)dtype.lanes
  //          << std::endl;
  //VERIFY((dtype.code == kDLInt) &&
  //    (dtype.bits == 8 || dtype.bits == 32) &&
  //    (dtype.lanes == 1))
  //  << "cvm runtime only supported INT8 or INT32 NDArray vs. ("
  //  << dtype.code << ", " << dtype.bits << ", " << dtype.lanes << ")";

  std::vector<int64_t> shape(ndim);
  if (ndim != 0) {
    CHECK(strm->ReadArray(&shape[0], ndim))
        << "Invalid DLTensor file format";
  }
  NDArray ret = NDArray::Empty(shape, dtype, ctx);
  int64_t num_elems = 1;
  int elem_bytes = (ret->dtype.bits + 7) / 8;
  for (int i = 0; i < ret->ndim; ++i) {
    num_elems *= ret->shape[i];
  }
  int64_t data_byte_size;
  CHECK(strm->Read(&data_byte_size))
      << "Invalid DLTensor file format";
  CHECK(data_byte_size == num_elems * elem_bytes)
      << "Invalid DLTensor file format";
  CHECK(strm->Read(ret->data, data_byte_size))
      << "Invalid DLTensor file format";
  if (!CVMUTIL_IO_NO_ENDIAN_SWAP) {
    utils::ByteSwap(ret->data, elem_bytes, num_elems);
  }

  // what the following code did is now done in CvmRuntime::SetInput
  //if(dtype.bits == 8){
  //    DLDataType dtype32 = dtype;
  //    dtype32.bits = 32;
  //    NDArray ret32 = NDArray::Empty(shape, dtype32, ctx);
  //    int8_t *data8 = static_cast<int8_t*>(ret->data);
  //    int32_t *data32 = static_cast<int32_t*>(ret32->data);
  //    for(int i = 0; i < num_elems; i++){
  //      data32[i] = static_cast<int32_t>(data8[i]);
  //    }
  //    *this = ret32;
  //}else{
  //    *this = ret;
  //}
  *this = ret;
  return true;
}

}  // namespace runtime
}  // namespace cvm
#endif  // CVM_RUNTIME_NDARRAY_H_
