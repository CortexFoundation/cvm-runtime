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
#ifndef CVM_RUNTIME_C_RUNTIME_API_H_
#define CVM_RUNTIME_C_RUNTIME_API_H_

#ifndef CVM_DLL
#define CVM_DLL __attribute__((visibility("default")))
#endif

// CVM version
#define CVM_VERSION "0.6.dev"


// CVM Runtime is DLPack compatible.
#include <cvm/dlpack.h>

#ifdef __cplusplus
extern "C" {
#endif
#include <stdint.h>
#include <stddef.h>

/*! \brief type of array index. */
typedef int64_t cvm_index_t;


/*!
 * \brief The type code in CVMType
 * \note CVMType is used in two places.
 */
typedef enum {
  // The type code of other types are compatible with DLPack.
  // The next few fields are extension types
  // that is used by CVM API calls.
  kHandle = 3U,
  kNull = 4U,
  kCVMType = 5U,
  kCVMContext = 6U,
  kArrayHandle = 7U,
  kNodeHandle = 8U,
  kModuleHandle = 9U,
  kFuncHandle = 10U,
  kStr = 11U,
  kBytes = 12U,
  kNDArrayContainer = 13U,
  // Extension codes for other frameworks to integrate CVM PackedFunc.
  // To make sure each framework's id do not conflict, use first and
  // last sections to mark ranges.
  // Open an issue at the repo if you need a section of code.
  kExtBegin = 15U,
  kCVMFirst = 16U,
  kCVMLast = 20U,
  // The following section of code is used for non-reserved types.
  kExtReserveEnd = 64U,
  kExtEnd = 128U
} CVMTypeCode;

/*!
 * \brief The data type used in CVM Runtime.
 *
 *  Examples
 *   - float: type_code = 2, bits = 32, lanes=1
 *   - float4(vectorized 4 float): type_code = 2, bits = 32, lanes=4
 *   - int8: type_code = 0, bits = 8, lanes=1
 *
 * \note Arguments CVM API function always takes bits=64 and lanes=1
 */
typedef DLDataType CVMType;

/*!
 * \brief The Device information, abstract away common device types.
 */
typedef DLContext CVMContext;

/*!
 * \brief The tensor array stucture to CVM API.
 */
typedef DLTensor CVMArray;

/*! \brief the array handle */
typedef CVMArray* CVMArrayHandle;

/*!
 * \brief Union type of values
 *  being passed through API and function calls.
 */
typedef union {
  int64_t v_int64;
  double v_float64;
  void* v_handle;
  const char* v_str;
  CVMType v_type;
  CVMContext v_ctx;
} CVMValue;

/*!
 * \brief Byte array type used to pass in byte array
 *  When kBytes is used as data type.
 */
typedef struct {
  const char* data;
  size_t size;
} CVMByteArray;

/*! \brief Handle to CVM runtime modules. */
typedef void* CVMModuleHandle;
/*! \brief Handle to packed function handle. */
typedef void* CVMFunctionHandle;
/*! \brief Handle to hold return value. */
typedef void* CVMRetValueHandle;
/*!
 * \brief The stream that is specific to device
 * can be NULL, which indicates the default one.
 */
typedef void* CVMStreamHandle;

/*!
 * \brief Load module from file.
 * \param file_name The file name to load the module from.
 * \param format The format of the module.
 * \param out The result module
 *
 * \return 0 when success, -1 when failure happens
 * \note The resulting module do not contain import relation.
 *  It can be reconstructed by CVMModImport.
 */
CVM_DLL int CVMModLoadFromFile(const char* file_name,
                               const char* format,
                               CVMModuleHandle* out);

/*!
 * \brief Add dep to mod's dependency.
 *  This allows functions in this module to use modules.
 *
 * \param mod The module handle.
 * \param dep The dependent module to be imported.
 * \return 0 when success, -1 when failure happens
 */
CVM_DLL int CVMModImport(CVMModuleHandle mod,
                         CVMModuleHandle dep);

/*!
 * \brief Get function from the module.
 * \param mod The module handle.
 * \param func_name The name of the function.
 * \param query_imports Whether to query imported modules
 * \param out The result function, can be NULL if it is not available.
 * \return 0 when no error is thrown, -1 when failure happens
 */
CVM_DLL int CVMModGetFunction(CVMModuleHandle mod,
                              const char* func_name,
                              int query_imports,
                              CVMFunctionHandle *out);

/*!
 * \brief Free front-end extension type resource.
 * \param handle The extension handle.
 * \param type_code The type of of the extension type.
 * \return 0 when success, -1 when failure happens
 */
CVM_DLL int CVMExtTypeFree(void* handle, int type_code);

/*!
 * \brief Free the Module
 * \param mod The module to be freed.
 *
 * \note This may not free up the module's resources.
 *  If there is active CVMFunctionHandle uses the module
 *  Or if this module is imported by another active module.
 *
 *  The all functions remains valid until CVMFuncFree is called.
 * \return 0 when success, -1 when failure happens
 */
CVM_DLL int CVMModFree(CVMModuleHandle mod);

/*!
 * \brief Free the function when it is no longer needed.
 * \param func The function handle
 * \return 0 when success, -1 when failure happens
 */
CVM_DLL int CVMFuncFree(CVMFunctionHandle func);

/*!
 * \brief Call a Packed CVM Function.
 *
 * \param func node handle of the function.
 * \param arg_values The arguments
 * \param type_codes The type codes of the arguments
 * \param num_args Number of arguments.
 *
 * \param ret_val The return value.
 * \param ret_type_code the type code of return value.
 *
 * \return 0 when success, -1 when failure happens
 * \note CVM calls always exchanges with type bits=64, lanes=1
 *
 * \note API calls always exchanges with type bits=64, lanes=1
 *   If API call returns container handles (e.g. FunctionHandle)
 *   these handles should be managed by the front-end.
 *   The front-end need to call free function (e.g. CVMFuncFree)
 *   to free these handles.
 */
CVM_DLL int CVMFuncCall(CVMFunctionHandle func,
                        CVMValue* arg_values,
                        int* type_codes,
                        int num_args,
                        CVMValue* ret_val,
                        int* ret_type_code);

/*!
 * \brief Set the return value of CVMPackedCFunc.
 *
 *  This function is called by CVMPackedCFunc to set the return value.
 *  When this function is not called, the function returns null by default.
 *
 * \param ret The return value handle, pass by ret in CVMPackedCFunc
 * \param value The value to be returned.
 * \param type_code The type of the value to be returned.
 * \param num_ret Number of return values, for now only 1 is supported.
 */
CVM_DLL int CVMCFuncSetReturn(CVMRetValueHandle ret,
                              CVMValue* value,
                              int* type_code,
                              int num_ret);

/*!
 * \brief Inplace translate callback argument value to return value.
 *  This is only needed for non-POD arguments.
 *
 * \param value The value to be translated.
 * \param code The type code to be translated.
 * \note This function will do a shallow copy when necessary.
 *
 * \return 0 when success, -1 when failure happens.
 */
CVM_DLL int CVMCbArgToReturn(CVMValue* value, int code);

/*!
 * \brief C type of packed function.
 *
 * \param args The arguments
 * \param type_codes The type codes of the arguments
 * \param num_args Number of arguments.
 * \param ret The return value handle.
 * \param resource_handle The handle additional resouce handle from fron-end.
 * \return 0 if success, -1 if failure happens, set error via CVMAPISetLastError.
 * \sa CVMCFuncSetReturn
 */
typedef int (*CVMPackedCFunc)(
    CVMValue* args,
    int* type_codes,
    int num_args,
    CVMRetValueHandle ret,
    void* resource_handle);

/*!
 * \brief C callback to free the resource handle in C packed function.
 * \param resource_handle The handle additional resouce handle from fron-end.
 */
typedef void (*CVMPackedCFuncFinalizer)(void* resource_handle);

/*!
 * \brief Signature for extension function declarer.
 *
 *  CVM call this function to get the extension functions
 *  The declarer will call register_func to register function and their name.
 *
 * \param register_func_handle The register function
 * \return 0 if success, -1 if failure happens
 */
typedef int (*CVMExtensionFuncDeclarer)(CVMFunctionHandle register_func_handle);

/*!
 * \brief Wrap a CVMPackedCFunc to become a FunctionHandle.
 *
 * The resource_handle will be managed by CVM API, until the function is no longer used.
 *
 * \param func The packed C function.
 * \param resource_handle The resource handle from front-end, can be NULL.
 * \param fin The finalizer on resource handle when the FunctionHandle get freed, can be NULL
 * \param out the result function handle.
 * \return 0 when success, -1 when failure happens
 */
CVM_DLL int CVMFuncCreateFromCFunc(CVMPackedCFunc func,
                                   void* resource_handle,
                                   CVMPackedCFuncFinalizer fin,
                                   CVMFunctionHandle *out);

/*!
 * \brief Register the function to runtime's global table.
 *
 * The registered function then can be pulled by the backend by the name.
 *
 * \param name The name of the function.
 * \param f The function to be registered.
 * \param override Whether allow override already registered function.
 */
CVM_DLL int CVMFuncRegisterGlobal(
    const char* name, CVMFunctionHandle f, int override);

/*!
 * \brief Get a global function.
 *
 * \param name The name of the function.
 * \param out the result function pointer, NULL if it does not exist.
 *
 * \note The function handle of global function is managed by CVM runtime,
 *  So CVMFuncFree is should not be called when it get deleted.
 */
CVM_DLL int CVMFuncGetGlobal(const char* name, CVMFunctionHandle* out);

/*!
 * \brief List all the globally registered function name
 * \param out_size The number of functions
 * \param out_array The array of function names.
 * \return 0 when success, -1 when failure happens
 */
CVM_DLL int CVMFuncListGlobalNames(int* out_size,
                                   const char*** out_array);

// Array related apis for quick proptyping
/*!
 * \brief Allocate a nd-array's memory,
 *  including space of shape, of given spec.
 *
 * \param shape The shape of the array, the data content will be copied to out
 * \param ndim The number of dimension of the array.
 * \param dtype_code The type code of the dtype
 * \param dtype_bits The number of bits of dtype
 * \param dtype_lanes The number of lanes in the dtype.
 * \param device_type The device type of context
 * \param device_id The device id of context.
 * \param out The output handle.
 * \return 0 when success, -1 when failure happens
 */
CVM_DLL int CVMArrayAlloc(const cvm_index_t* shape,
                          int ndim,
                          int dtype_code,
                          int dtype_bits,
                          int dtype_lanes,
                          int device_type,
                          int device_id,
                          CVMArrayHandle* out);

/*!
 * \brief Free the CVM Array.
 * \param handle The array handle to be freed.
 * \return 0 when success, -1 when failure happens
 */
CVM_DLL int CVMArrayFree(CVMArrayHandle handle);

/*!
 * \brief Copy array data from CPU byte array.
 * \param handle The array handle.
 * \param data the data pointer
 * \param nbytes The number of bytes to copy.
 * \return 0 when success, -1 when failure happens
 */
CVM_DLL int CVMArrayCopyFromBytes(CVMArrayHandle handle,
                                  void* data,
                                  size_t nbytes);

/*!
 * \brief Copy array data to CPU byte array.
 * \param handle The array handle.
 * \param data the data pointer
 * \param nbytes The number of bytes to copy.
 * \return 0 when success, -1 when failure happens
 */
CVM_DLL int CVMArrayCopyToBytes(CVMArrayHandle handle,
                                void* data,
                                size_t nbytes);

/*!
 * \brief Copy the array, both from and to must be valid during the copy.
 * \param from The array to be copied from.
 * \param to The target space.
 * \param stream The stream where the copy happens, can be NULL.
 * \return 0 when success, -1 when failure happens
 */
CVM_DLL int CVMArrayCopyFromTo(CVMArrayHandle from,
                               CVMArrayHandle to,
                               CVMStreamHandle stream);

CVM_DLL int CVMSaveParamsDict(
    const void** params,
    int params_size,
    CVMByteArray* re);

CVM_DLL int CVMLoadParamsDict(
    const char* data,
    int datalen,
    int* retNum,
    void*** retNames,
    void*** retValues);

CVM_DLL int CVMAssignSliceScalar(
    CVMArrayHandle target,
    int* indices,
    double value
);

CVM_DLL int CVMAssignSliceND(
    CVMArrayHandle target,
    int* indices,
    CVMArrayHandle source
);

CVM_DLL int CVMAssignAllScalar(CVMArrayHandle target, double value);

CVM_DLL int CVMAssignAllND(CVMArrayHandle target, CVMArrayHandle source);

#ifdef __cplusplus
}  // CVM_EXTERN_C
#endif
#endif  // CVM_RUNTIME_C_RUNTIME_API_H_
