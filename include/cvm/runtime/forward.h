#ifndef CVM_FORWARD_H
#define CVM_FORWARD_H

#include <cvm/tuple.h>
#include <cvm/runtime/packed_func.h>
#include <utils/logging.h>

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

class Indices {
 public:
  Indices(TShape const& src)
    : shape_(src), max_size_(src.Size()),
      indices_(src.ndim()), index_cache_(0) {}
  Indices(TShape && src)
    : shape_(src), max_size_(src.Size()),
      indices_(src.ndim()), index_cache_(0) {}
  Indices(TShape const& src, Indices const& indices) 
    : shape_(src), max_size_(src.Size()),
      indices_(indices.indices_), index_cache_(-1) {}

  inline uint32_t ndim() const { return indices_.size(); }
  /**
   * \brief Returns the flatten index corresponding with the
   *    indices_ value and source shape.
   **/
  size_t Index() const { 
    index_correct();
    return index_cache_; 
  }

  inline bool End() const {
    return Index() + 1 == max_size_;
  }

  void operator++() {
    CHECK(shape_.ndim()) << "Unknown shape of dim equals zero";
    CHECK(!End()) << "Iterator has been the end";

    index_cache_ = Index() + 1;

    int cnt = shape_.ndim() - 1;
    while (true) {
      indices_[cnt] ++;

      if (indices_[cnt] < shape_[cnt]) return ;

      indices_[cnt] = 0;
      cnt --;
    }
  }

  inline dim_t& operator[](size_t i) { 
    // The value may be changed outside the class, so set
    //  the cache to be invalid state.
    index_cache_ = -1;
    return indices_[i]; 
  }
  inline const dim_t& operator[](size_t i) const { 
    return indices_[i]; 
  }

 private:
  TShape shape_;
  size_t max_size_;
  std::vector<dim_t> indices_;
  /*
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

  void index_correct() const {
    if (index_cache_ != -1) return ;

    size_t real_index = 0;
    size_t current_level = 1;
    for (dim_t i = 0; i < shape_.ndim(); ++i) {
      dim_t real_i = shape_.ndim() - 1 - i;
      // Calculate the current indices' level number
      real_index += indices_[real_i] * current_level;
      // Update the level number of next indices stands for.
      current_level *= shape_[real_i];
    }
    index_cache_ = real_index;
  }
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
