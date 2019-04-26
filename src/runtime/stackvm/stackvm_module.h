/*!
 *  Copyright (c) 2018 by Contributors
 * \file stackvm_module.h
 * \brief StackVM module
 */
#ifndef CVM_RUNTIME_STACKVM_STACKVM_MODULE_H_
#define CVM_RUNTIME_STACKVM_STACKVM_MODULE_H_

#include <cvm/runtime/packed_func.h>
#include <string>
#include <unordered_map>
#include "stackvm.h"

namespace cvm {
namespace runtime {
/*!
 * \brief create a stackvm module
 *
 * \param fmap The map from name to function
 * \param entry_func The entry function name.
 * \return The created module
 */
Module StackVMModuleCreate(std::unordered_map<std::string, StackVM> fmap,
                           std::string entry_func);

}  // namespace runtime
}  // namespace cvm
#endif  // CVM_RUNTIME_STACKVM_STACKVM_MODULE_H_
