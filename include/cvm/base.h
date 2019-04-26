/*!
 *  Copyright (c) 2016 by Contributors
 * \file cvm/base.h
 * \brief Configuration of cvm as well as basic data structure.
 */
#ifndef CVM_BASE_H_
#define CVM_BASE_H_

#include <dmlc/base.h>
#include <dmlc/common.h>
#include <dmlc/any.h>
#include <dmlc/memory.h>
#include <dmlc/logging.h>
#include <dmlc/registry.h>
#include <dmlc/array_view.h>

namespace cvm {

/*! \brief any type */
using dmlc::any;

/*! \brief array_veiw type  */
using dmlc::array_view;

/*!\brief getter function of any type */
using dmlc::get;

/*!\brief "unsafe" getter function of any type */
using dmlc::unsafe_get;

}  // namespace cvm

// describe op registration point
#define CVM_STRINGIZE_DETAIL(x) #x
#define CVM_STRINGIZE(x) CVM_STRINGIZE_DETAIL(x)
#define CVM_DESCRIBE(...) describe(__VA_ARGS__ "\n\nFrom:" __FILE__ ":" CVM_STRINGIZE(__LINE__))
#define CVM_ADD_FILELINE "\n\nDefined in " __FILE__ ":L" CVM_STRINGIZE(__LINE__)
#endif  // CVM_BASE_H_
