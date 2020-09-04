/*!
 *  Copyright (c) 2016 by Contributors
 * \file cvm/runtime/c_runtime_api.h
 * \brief CVM runtime library.
 *
 *  The philosophy of CVM project is to customize the compilation
 *  stage to generate code that can used by other projects transparently.
 *  So this is a minimum runtime code gluing, and some limited
 *  memory management code to enable quick testing.
 *
 *  The runtime API is independent from CVM compilation stack and can
 *  be linked via libcvm_runtime.
 *
 *  The common flow is:
 *   - Use CVMFuncListGlobalNames to get global function name
 *   - Use CVMFuncCall to call these functions.
 */
#ifndef CVM_RUNTIME_BASE_H
#define CVM_RUNTIME_BASE_H

#ifndef CVM_DLL
#define CVM_DLL __attribute__((visibility("default")))
#endif

#include <cvm/dlpack.h>
#include <utils/logging.h>

#define CVM_TYPE_SWITCH(dtype, DType, ...)                  \  
  if (dtype.code == kDLInt) {                                              \
    switch (dtype.bits) {                                                  \
      case 8: {                                                          \
        typedef int8_t DType;                                            \
        { __VA_ARGS__ }                                                  \
      } break;                                                           \
      case 16: {                                                         \
        typedef int16_t DType;                                           \
        { __VA_ARGS__ }                                                  \
      } break;                                                           \
      case 32: {                                                         \
        typedef int32_t DType;                                           \
        { __VA_ARGS__ }                                                  \
      } break;                                                           \
      case 64: {                                                         \
        typedef int64_t DType;                                           \
        { __VA_ARGS__ }                                                  \
      } break;                                                           \
      default:                                                           \
        LOG(FATAL) << "unknown size for int type: " << dtype.bits          \
                   << ". only accept 8, 16, 32 or 64 bits int\n";        \
    }                                                                    \
  } else if (dtype.code == kDLUInt) {                                      \
    switch (dtype.bits) {                                                  \
      case 8: {                                                          \
        typedef uint8_t DType;                                           \
        { __VA_ARGS__ }                                                  \
      } break;                                                           \
      case 16: {                                                         \
        typedef uint16_t DType;                                          \
        { __VA_ARGS__ }                                                  \
      } break;                                                           \
      case 32: {                                                         \
        typedef uint32_t DType;                                          \
        { __VA_ARGS__ }                                                  \
      } break;                                                           \
      case 64: {                                                         \
        typedef uint64_t DType;                                          \
        { __VA_ARGS__ }                                                  \
      } break;                                                           \
      default:                                                           \
        LOG(FATAL) << "unknown size for uint type: " << dtype.bits         \
                   << ". only accept 8, 16, 32 or 64 bits int\n";        \
    }                                                                    \
  } else if (dtype.code == kDLFloat) {                                     \
    switch (dtype.bits) {                                                  \
      case 32: {                                                         \
        typedef float DType;                                             \
        { __VA_ARGS__ }                                                  \
      } break;                                                           \
      case 64: {                                                         \
        typedef double DType;                                            \
        { __VA_ARGS__ }                                                  \
      } break;                                                           \
      default:                                                           \
        LOG(FATAL) << "unknown size for float type: " << dtype.bits        \
                   << ". only accept 32 bits float or 64 bits double\n"; \
    }                                                                    \
  }

#endif  // CVM_RUNTIME_BASE_H