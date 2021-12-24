#include <cvm/runtime/forward.h>
#include <utils/logging.h>

namespace cvm {
namespace runtime {

void Indices::operator++() {
  CHECK(!End()) << "Indices has been the end of Shape";

  index_cache_ = Index() + 1;
  int cnt = shape_.ndim() - 1;
  while (cnt >= 0) {
    indices_[cnt] ++;
    if (indices_[cnt] < shape_[cnt]) return ;

    indices_[cnt] = 0;
    cnt --;
  }
}

void Indices::index_correct() const {
  if (index_cache_ != -1) return ;

  uint32_t ndim = shape_.ndim();
  // lazy initialize the shape level cache
  if (shape_level_.size() != ndim) {
    shape_level_.resize(ndim, 1);
    for (int i = int(ndim)-2; i >= 0; --i) {
      shape_level_[i] = shape_[i+1] * shape_level_[i+1];
    }
  }

  size_t real_index = 0;
  for (int i = int(ndim)-1; i >= 0; --i) {
    // indices should not be negative
    // CHECK(indices_[i] >= 0) << "indices has negative";
    // calculate the current indices' level number
    real_index += indices_[i] * shape_level_[i];
  }
  CHECK(real_index <= max_size_) << "indices out of bound";

  index_cache_ = real_index;
}

std::string Indices::to_string() const {
  std::ostringstream oss;
  oss << "<Indices[";
  for (auto it = indices_.begin(); it != indices_.end(); ++it) {
    if (it != indices_.begin()) oss << ",";
    oss << *it;
  }
  oss << "] over TShape";
  oss << shape_;
  oss << ">";
  return oss.str();
}

void Indices::swap(Indices &other) {
  std::swap(shape_, other.shape_);
  std::swap(max_size_, other.max_size_);
  std::swap(shape_level_, other.shape_level_);

  std::swap(indices_, other.indices_);
  std::swap(index_cache_, other.index_cache_);
}

}
}
