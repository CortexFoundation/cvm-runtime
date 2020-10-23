#ifndef CVM_PARAM_DICT_H
#define CVM_PARAM_DICT_H

#include <vector>

#include <utils/io.h>
#include <cvm/runtime/ndarray.h>

namespace cvm {
namespace runtime {

void load_param_dict(
    utils::Stream* strm,
    std::vector<std::string>& names,
    std::vector<cvm::runtime::NDArray>& values);

void save_param_dict(
    utils::Stream* strm,
    std::vector<std::string> const& names,
    std::vector<DLTensor*> const& values);

}
}

#endif // CVM_PARAM_DICT_H
