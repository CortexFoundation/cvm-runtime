#ifndef CVM_FORWARD_H
#define CVM_FORWARD_H

#include <cvm/tuple.h>
#include <cvm/runtime/packed_func.h>

namespace cvm {
namespace runtime {

// Helper functions used in operators' forward
template<typename DType>
inline DType* CVMArg2Data(CVMArgValue const& av) {
  DLTensor *tensor = av.operator DLTensor *();
  return static_cast<DType*>(tensor->data);
}

template<typename PType>
inline PType& CVMArg2Attr(CVMArgValue const& av) {
  void *ptr = av.operator void *();
  auto attr = static_cast<cvm::NodeAttrs*>(ptr);
  return cvm::get<PType>(attr->parsed);
}

inline TShape CVMArgShape(CVMArgValue const& av) {
  DLTensor *tensor = av.operator DLTensor *();
  return TShape(tensor->shape, tensor->shape + tensor->ndim);
}

inline size_t CVMArgSize(CVMArgValue const& av) {
  TShape const& shape = CVMArgShape(av);
  return shape.Size();
}

inline uint32_t CVMArgNdim(CVMArgValue const& av) {
  DLTensor* tensor = av.operator DLTensor*();
  return tensor->ndim;
}

/**
 * \brief TShape Iterator Class: Indices
 *
 * The indices class hold two vector-like member: shape and
 *  indices, whose ndim is the same as each other. The indices
 *  variable could iterate all over the range from 
 *  [0, 0, 0 ...] into the maximum shape.
 **/
class Indices {
 public:
  Indices(TShape const& src)
    : shape_(src), max_size_(src.Size()),
      indices_(src.ndim()), index_cache_(0) {}
  Indices(TShape && src)
    : shape_(src), max_size_(src.Size()),
      indices_(src.ndim()), index_cache_(0) {}

  Indices(Indices && other) {
    this->swap(other);
  }
  Indices& operator=(Indices && other) {
    this->swap(other);
    return *this;
  }

  inline uint32_t ndim() const { return indices_.size(); }
  /**
   * \brief Returns the flatten index corresponding with the
   *    indices_ value and source shape.
   **/
  size_t Index() const { 
    index_correct();
    return index_cache_; 
  }
  bool End() const { return Index() == max_size_; }
  void CopyIndicesFrom(Indices const& indices) {
    index_cache_ = -1;
    indices_ = indices.indices_;
  }
  void CopyIndicesFrom(const std::vector<dim_t>& vecIdx) {
    index_cache_ = -1;
    indices_ = vecIdx;
  }
  void CopyIndicesFrom(std::vector<dim_t>&& vecIdx) {
    std::swap(vecIdx, indices_);
    index_cache_ = -1;
  }
  dim_t& Ref(size_t i) {
    index_cache_ = -1;
    return indices_[i];
  }

  void operator++();
  void operator++(int) { return this->operator++(); }

  dim_t const& operator[](size_t i) const { return indices_[i]; }

  void swap(Indices &other);
  std::string to_string() const;
  void reset() {
    std::fill(indices_.begin(), indices_.end(), 0);
    index_cache_ = -1;
  }

 private:
  void index_correct() const;

 private:
  /**
   * \brief initial maximum shape
   *
   * The shape and corresponding max size should not change
   *  after setup.
   **/
  TShape shape_;
  size_t max_size_;
  /**
   * \brief real indices stands for
   *
   * Not support negative index.
   **/
  std::vector<dim_t> indices_;
  /**
   * \brief flatten index cache
   *
   * Delcare the index_cache_ type as int to set -1 for invalid
   *  state. However, the user API:`index()` to get Indices'
   *  flatten index will be size_t type, obeying the unify
   *  Indices or Shape Size interface.
   *
   * TShape Size function refers to file `include/cvm/tuple.h`, 
   *  line 388.
   **/
  mutable int64_t index_cache_; 
  /**
   * \brief shape flatten level cache
   **/
  mutable std::vector<size_t> shape_level_;
};

inline int32_t CVMShapeBegin(CVMArgValue const& av){
  return 0;
}

inline int32_t CVMShapeEnd(CVMArgValue const& av){
  return CVMArgSize(av);
}

//  Convert an index (id_1, id_2,,, id_n) into a number using shape (s_1, s_2,,, s_n) as its base.
inline int64_t Index2Number(const TShape& shape,
                            const std::vector<int64_t>& index){
  auto number = index[0];
  for (u_int32_t i = 1; i < shape.ndim(); i++){
    number = number * shape[i] + index[i];
  }
  return number;
}
inline int64_t Index2Number(const std::vector<int64_t>& shape,
                            const std::vector<int64_t>& index){
  return Index2Number(TShape(shape), index);
}

//  Add index (id_1, id_2,,, id_n) with 1 using shape (s_1, s_2,,, s_n) as its shape
inline void IndexBaseShapeAddOne(const TShape& shape,
                                 std::vector<int64_t>& index){
  auto cnt = shape.ndim() - 1;
  index[cnt]++;
  while (cnt > 0 && index[cnt] == shape[cnt]){
    index[cnt--] = 0;
    index[cnt]++;
  }
}
inline void IndexBaseShapeAddOne(const std::vector<int64_t>& shape,
                                 std::vector<int64_t>& index) {
  IndexBaseShapeAddOne(TShape(shape), index);
}

const std::string DIR = "/tmp/zkh/ssd/";
inline void print_to_file(
    DLTensor *y, std::string filename, bool all=false){
#if defined(CVM_PRINT_OP_RESULT)
  FILE *fp = fopen((DIR + filename).c_str(), "a+");
  int32_t *y_data = static_cast<int32_t*>(y->data);

  int32_t min = y_data[0], max= y_data[0];
  for(uint64_t i = 0; i < getSize(y); i++){
      min = min > y_data[i] ? y_data[i] : min;
      max = max < y_data[i] ? y_data[i] : max;
  }
  fprintf(fp, "%d %d\n", min, max);
  if(all){
    for(uint64_t i = 0; i < getSize(y); i++){
      fprintf(fp, "%d ", y_data[i]);
    }
  }else{
    for(uint64_t i = 0; i < 100 && i < getSize(y); i++){
      fprintf(fp, "%d ", y_data[i]);
    }
  }
  fprintf(fp, "\n");
  fclose(fp);
#endif
}

}
}

#endif // CVM_FORWARD_H
