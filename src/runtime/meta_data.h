/*!
 *  Copyright (c) 2017 by Contributors
 * \file meta_data.h
 * \brief Meta data related utilities
 */
#ifndef CVM_RUNTIME_META_DATA_H_
#define CVM_RUNTIME_META_DATA_H_

#include <dmlc/json.h>
#include <dmlc/io.h>
#include <cvm/runtime/packed_func.h>
#include <string>
#include <vector>
#include "runtime_base.h"

namespace cvm {
namespace runtime {

/*! \brief function information needed by device */
struct FunctionInfo {
  std::string name;
  std::vector<CVMType> arg_types;
  std::vector<std::string> thread_axis_tags;

  void Save(dmlc::JSONWriter *writer) const;
  void Load(dmlc::JSONReader *reader);
  void Save(dmlc::Stream *writer) const;
  bool Load(dmlc::Stream *reader);
};
}  // namespace runtime
}  // namespace cvm

namespace dmlc {
DMLC_DECLARE_TRAITS(has_saveload, ::cvm::runtime::FunctionInfo, true);
}  // namespace dmlc
#endif  // CVM_RUNTIME_META_DATA_H_
