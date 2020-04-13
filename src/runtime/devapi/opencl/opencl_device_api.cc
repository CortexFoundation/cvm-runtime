/*!
 *  Copyright (c) 2017 by Contributors
 * \file cuda_device_api.cc
 * \brief GPU specific API
 */
#include <cvm/runtime/device_api.h>

#include <utils/thread_local.h>
#include <cvm/runtime/registry.h>

#define CL_USE_DEPRECATED_OPENCL_1_2_APIS

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/opencl.h>
#endif

namespace cvm {
namespace runtime {

inline const char* CLGetErrorString(cl_int error) {
  switch (error) {
    case CL_SUCCESS: return "CL_SUCCESS";
    case CL_DEVICE_NOT_FOUND: return "CL_DEVICE_NOT_FOUND";
    case CL_DEVICE_NOT_AVAILABLE: return "CL_DEVICE_NOT_AVAILABLE";
    case CL_COMPILER_NOT_AVAILABLE: return "CL_COMPILER_NOT_AVAILABLE";
    case CL_MEM_OBJECT_ALLOCATION_FAILURE: return "CL_MEM_OBJECT_ALLOCATION_FAILURE";
    case CL_OUT_OF_RESOURCES: return "CL_OUT_OF_RESOURCES";
    case CL_OUT_OF_HOST_MEMORY: return "CL_OUT_OF_HOST_MEMORY";
    case CL_PROFILING_INFO_NOT_AVAILABLE: return "CL_PROFILING_INFO_NOT_AVAILABLE";
    case CL_MEM_COPY_OVERLAP: return "CL_MEM_COPY_OVERLAP";
    case CL_IMAGE_FORMAT_MISMATCH: return "CL_IMAGE_FORMAT_MISMATCH";
    case CL_IMAGE_FORMAT_NOT_SUPPORTED: return "CL_IMAGE_FORMAT_NOT_SUPPORTED";
    case CL_BUILD_PROGRAM_FAILURE: return "CL_BUILD_PROGRAM_FAILURE";
    case CL_MAP_FAILURE: return "CL_MAP_FAILURE";
    case CL_INVALID_VALUE: return "CL_INVALID_VALUE";
    case CL_INVALID_DEVICE_TYPE: return "CL_INVALID_DEVICE_TYPE";
    case CL_INVALID_PLATFORM: return "CL_INVALID_PLATFORM";
    case CL_INVALID_DEVICE: return "CL_INVALID_DEVICE";
    case CL_INVALID_CONTEXT: return "CL_INVALID_CONTEXT";
    case CL_INVALID_QUEUE_PROPERTIES: return "CL_INVALID_QUEUE_PROPERTIES";
    case CL_INVALID_COMMAND_QUEUE: return "CL_INVALID_COMMAND_QUEUE";
    case CL_INVALID_HOST_PTR: return "CL_INVALID_HOST_PTR";
    case CL_INVALID_MEM_OBJECT: return "CL_INVALID_MEM_OBJECT";
    case CL_INVALID_IMAGE_FORMAT_DESCRIPTOR: return "CL_INVALID_IMAGE_FORMAT_DESCRIPTOR";
    case CL_INVALID_IMAGE_SIZE: return "CL_INVALID_IMAGE_SIZE";
    case CL_INVALID_SAMPLER: return "CL_INVALID_SAMPLER";
    case CL_INVALID_BINARY: return "CL_INVALID_BINARY";
    case CL_INVALID_BUILD_OPTIONS: return "CL_INVALID_BUILD_OPTIONS";
    case CL_INVALID_PROGRAM: return "CL_INVALID_PROGRAM";
    case CL_INVALID_PROGRAM_EXECUTABLE: return "CL_INVALID_PROGRAM_EXECUTABLE";
    case CL_INVALID_KERNEL_NAME: return "CL_INVALID_KERNEL_NAME";
    case CL_INVALID_KERNEL_DEFINITION: return "CL_INVALID_KERNEL_DEFINITION";
    case CL_INVALID_KERNEL: return "CL_INVALID_KERNEL";
    case CL_INVALID_ARG_INDEX: return "CL_INVALID_ARG_INDEX";
    case CL_INVALID_ARG_VALUE: return "CL_INVALID_ARG_VALUE";
    case CL_INVALID_ARG_SIZE: return "CL_INVALID_ARG_SIZE";
    case CL_INVALID_KERNEL_ARGS: return "CL_INVALID_KERNEL_ARGS";
    case CL_INVALID_WORK_DIMENSION: return "CL_INVALID_WORK_DIMENSION";
    case CL_INVALID_WORK_GROUP_SIZE: return "CL_INVALID_WORK_GROUP_SIZE";
    case CL_INVALID_WORK_ITEM_SIZE: return "CL_INVALID_WORK_ITEM_SIZE";
    case CL_INVALID_GLOBAL_OFFSET: return "CL_INVALID_GLOBAL_OFFSET";
    case CL_INVALID_EVENT_WAIT_LIST: return "CL_INVALID_EVENT_WAIT_LIST";
    case CL_INVALID_EVENT: return "CL_INVALID_EVENT";
    case CL_INVALID_OPERATION: return "CL_INVALID_OPERATION";
    case CL_INVALID_GL_OBJECT: return "CL_INVALID_GL_OBJECT";
    case CL_INVALID_BUFFER_SIZE: return "CL_INVALID_BUFFER_SIZE";
    case CL_INVALID_MIP_LEVEL: return "CL_INVALID_MIP_LEVEL";
    default: return "Unknown OpenCL error code";
  }
}
#define OPENCL_CHECK_ERROR(e)                                           \
  {                                                                     \
    CHECK(e == CL_SUCCESS)                                              \
        << "OpenCL Error, code=" << e << ": " << CLGetErrorString(e); \
  }

#define OPENCL_CALL(func)                                             \
  {                                                                   \
    cl_int e = (func);                                                \
    OPENCL_CHECK_ERROR(e);                                            \
  }

class OpenCLDeviceAPI final : public DeviceAPI {
  public:
  // type key
  std::string type_key;
  // global platform id
  cl_platform_id platform_id;
  // global platform name
  std::string platform_name;
  // global context of this process
  cl_context context{nullptr};
  // whether the workspace it initialized.
  bool initialized_{false};
  // the device type
  std::string device_type;
  // the devices
  std::vector<cl_device_id> devices;
  // the queues
  std::vector<cl_command_queue> queues;
  // Number of registered kernels
  // Used to register kernel into the workspace.
  size_t num_registered_kernels{0};
  // The version counter, used
  size_t timestamp{0};
  // Ids that are freed by kernels.
  std::vector<size_t> free_kernel_ids;
  // the mutex for initialization
  std::mutex mu;

  std::string GetPlatformInfo(
      cl_platform_id pid, cl_platform_info param_name) {
    size_t ret_size;
    OPENCL_CALL(clGetPlatformInfo(pid, param_name, 0, nullptr, &ret_size));
    std::string ret;
    ret.resize(ret_size);
    OPENCL_CALL(clGetPlatformInfo(pid, param_name, ret_size, &ret[0], nullptr));
    return ret;
  }

  bool MatchPlatformInfo(
      cl_platform_id pid,
      cl_platform_info param_name,
      std::string value) {
    if (value.length() == 0) return true;
    std::string param_value = GetPlatformInfo(pid, param_name);
    return param_value.find(value) != std::string::npos;
  }
  std::vector<cl_platform_id> GetPlatformIDs() {
    cl_uint ret_size;
    cl_int code = clGetPlatformIDs(0, nullptr, &ret_size);
    std::vector<cl_platform_id> ret;
    if (code != CL_SUCCESS) return ret;
    ret.resize(ret_size);
    OPENCL_CALL(clGetPlatformIDs(ret_size, &ret[0], nullptr));
    return ret;
  }
  std::vector<cl_device_id> GetDeviceIDs(
      cl_platform_id pid, std::string device_type) {
    cl_device_type dtype = CL_DEVICE_TYPE_ALL;
    if (device_type == "cpu") dtype = CL_DEVICE_TYPE_CPU;
    if (device_type == "gpu") dtype = CL_DEVICE_TYPE_GPU;
    if (device_type == "accelerator") dtype = CL_DEVICE_TYPE_ACCELERATOR;
    cl_uint ret_size;
    cl_int code = clGetDeviceIDs(pid, dtype, 0, nullptr, &ret_size);
    std::vector<cl_device_id> ret;
    if (code != CL_SUCCESS) return ret;
    ret.resize(ret_size);
    OPENCL_CALL(clGetDeviceIDs(pid, dtype, ret_size, &ret[0], nullptr));
    return ret;
  }

  bool IsOpenCLDevice(CVMContext ctx) {
    return ctx.device_type == kDLOpenCL;
  }

  cl_command_queue GetQueue(CVMContext ctx) {
    CHECK(IsOpenCLDevice(ctx));
    this->Init();
    CHECK(ctx.device_id >= 0  && static_cast<size_t>(ctx.device_id) < queues.size())
        << "Invalid OpenCL device_id=" << ctx.device_id;
    return queues[ctx.device_id];
  }
  void Init(const std::string& type_key, const std::string& device_type,
                           const std::string& platform_name = "") {
    if (initialized_) return;
    std::lock_guard<std::mutex> lock(this->mu);
    if (initialized_) return;
    if (context != nullptr) return;
    this->type_key = type_key;
    // matched platforms
    std::vector<cl_platform_id> platform_ids = GetPlatformIDs();
    if (platform_ids.size() == 0) {
      LOG(WARNING) << "No OpenCL platform matched given existing options ...";
      return;
    }
    this->platform_id = nullptr;
    for (auto platform_id : platform_ids) {
      if (!MatchPlatformInfo(platform_id, CL_PLATFORM_NAME, platform_name)) {
        continue;
      }
      std::vector<cl_device_id> devices_matched = GetDeviceIDs(platform_id, device_type);
      if ((devices_matched.size() == 0) && (device_type == "gpu")) {
        LOG(WARNING) << "Using CPU OpenCL device";
        devices_matched = GetDeviceIDs(platform_id, "cpu");
      }
      if (devices_matched.size() > 0) {
        this->platform_id = platform_id;
        this->platform_name = GetPlatformInfo(platform_id, CL_PLATFORM_NAME);
        this->device_type = device_type;
        this->devices = devices_matched;
        break;
      }
    }
    if (this->platform_id == nullptr) {
      LOG(WARNING) << "No OpenCL device";
      return;
    }
    cl_int err_code;
    this->context = clCreateContext(
        nullptr, this->devices.size(), &(this->devices[0]),
        nullptr, nullptr, &err_code);
    OPENCL_CHECK_ERROR(err_code);
    CHECK_EQ(this->queues.size(), 0U);
    for (size_t i = 0; i < this->devices.size(); ++i) {
      cl_device_id did = this->devices[i];
      this->queues.push_back(
          clCreateCommandQueue(this->context, did, 0, &err_code));
      OPENCL_CHECK_ERROR(err_code);
    }
    initialized_ = true;
  }

  virtual void Init() {
    Init("opencl", "gpu");
  }

  void SetDevice(CVMContext ctx) final {
    //context.device_id = ctx.device_id;
  }

  void GetAttr(CVMContext ctx, DeviceAttrKind kind, CVMRetValue* rv) final {
    this->Init();
    size_t index = static_cast<size_t>(ctx.device_id);
    if (kind == kExist) {
      *rv = static_cast<int>(index< devices.size());
      return;
    }
    CHECK_LT(index, devices.size())
      << "Invalid device id " << index;
    switch (kind) {
      case kExist: break;
      case kMaxThreadsPerBlock: {
                                  size_t value;
                                  OPENCL_CALL(clGetDeviceInfo(
                                        devices[index],  CL_DEVICE_MAX_WORK_GROUP_SIZE,
                                        sizeof(size_t), &value, nullptr));
                                  *rv = static_cast<int64_t>(value);
                                  break;
                                }
      case kWarpSize: {
                        /* TODO: the warp size of OpenCL device is not always 1
                           e.g. Intel Graphics has a sub group concept which contains 8 - 32 work items,
                           corresponding to the number of SIMD entries the heardware configures.
                           We need to figure out a way to query this information from the hardware.
                           */
                        *rv = 1;
                        break;
                      }
      case kMaxSharedMemoryPerBlock: {
                                       cl_ulong value;
                                       OPENCL_CALL(clGetDeviceInfo(
                                             devices[index], CL_DEVICE_LOCAL_MEM_SIZE,
                                             sizeof(cl_ulong), &value, nullptr));
                                       *rv = static_cast<int64_t>(value);
                                       break;
                                     }
      case kComputeVersion: return;
      case kDeviceName: {
                          char value[128] = {0};
                          OPENCL_CALL(clGetDeviceInfo(
                                devices[index], CL_DEVICE_NAME,
                                sizeof(value) - 1, value, nullptr));
                          *rv = std::string(value);
                          break;
                        }
      case kMaxClockRate: {
                            cl_uint value;
                            OPENCL_CALL(clGetDeviceInfo(
                                  devices[index], CL_DEVICE_MAX_CLOCK_FREQUENCY,
                                  sizeof(cl_uint), &value, nullptr));
                            *rv = static_cast<int32_t>(value);
                            break;
                          }
      case kMultiProcessorCount: {
                                   cl_uint value;
                                   OPENCL_CALL(clGetDeviceInfo(
                                         devices[index], CL_DEVICE_MAX_COMPUTE_UNITS,
                                         sizeof(cl_uint), &value, nullptr));
                                   *rv = static_cast<int32_t>(value);
                                   break;
                                 }
      case kMaxThreadDimensions: {
                                   size_t dims[3];
                                   OPENCL_CALL(clGetDeviceInfo(
                                         devices[index], CL_DEVICE_MAX_WORK_ITEM_SIZES, sizeof(dims), dims, nullptr));

                                   std::stringstream ss;  // use json string to return multiple int values;
                                   ss << "[" << dims[0] <<", " << dims[1] << ", " << dims[2] << "]";
                                   *rv = ss.str();
                                   break;
                                 }
    }
  }
  void* AllocDataSpace(
      CVMContext ctx, size_t size, size_t alignment, CVMType type_hint) {
    this->Init();
    CHECK(context != nullptr) << "No OpenCL device";
    if(size == 0) return nullptr;
    cl_int err_code;
    cl_mem mptr = clCreateBuffer(
        this->context, CL_MEM_READ_WRITE, size, nullptr, &err_code);
    OPENCL_CHECK_ERROR(err_code);
    return mptr;
  }

  void FreeDataSpace(CVMContext ctx, void* ptr) {
    // We have to make sure that the memory object is not in the command queue
    // for some OpenCL platforms.
    OPENCL_CALL(clFinish(this->GetQueue(ctx)));

    cl_mem mptr = static_cast<cl_mem>(ptr);
    OPENCL_CALL(clReleaseMemObject(mptr));
  }

  void CopyDataFromTo(const void* from,
      size_t from_offset,
      void* to,
      size_t to_offset,
      size_t size,
      CVMContext ctx_from,
      CVMContext ctx_to,
      CVMType type_hint,
      CVMStreamHandle stream) {
    this->Init();
    CHECK(stream == nullptr);
    if (IsOpenCLDevice(ctx_from) && IsOpenCLDevice(ctx_to)) {
      OPENCL_CALL(clEnqueueCopyBuffer(
            this->GetQueue(ctx_to),
            static_cast<cl_mem>((void*)from),  // NOLINT(*)
            static_cast<cl_mem>(to),
            from_offset, to_offset, size, 0, nullptr, nullptr));
    } else if (IsOpenCLDevice(ctx_from) && ctx_to.device_type == kDLCPU) {
      OPENCL_CALL(clEnqueueReadBuffer(
            this->GetQueue(ctx_from),
            static_cast<cl_mem>((void*)from),  // NOLINT(*)
            CL_FALSE, from_offset, size,
            static_cast<char*>(to) + to_offset,
            0, nullptr, nullptr));
      OPENCL_CALL(clFinish(this->GetQueue(ctx_from)));
    } else if (ctx_from.device_type == kDLCPU && IsOpenCLDevice(ctx_to)) {
      OPENCL_CALL(clEnqueueWriteBuffer(
            this->GetQueue(ctx_to),
            static_cast<cl_mem>(to),
            CL_FALSE, to_offset, size,
            static_cast<const char*>(from) + from_offset,
            0, nullptr, nullptr));
      OPENCL_CALL(clFinish(this->GetQueue(ctx_to)));
    } else {
      LOG(FATAL) << "Expect copy from/to OpenCL or between OpenCL";
    }
  }
  static const std::shared_ptr<OpenCLDeviceAPI>& Global() {
    static std::shared_ptr<OpenCLDeviceAPI> inst = std::make_shared<OpenCLDeviceAPI>();
    return inst;
  }

};

CVM_REGISTER_GLOBAL("device_api.opencl")
.set_body([](CVMArgs args, CVMRetValue* rv) {
    DeviceAPI* ptr = OpenCLDeviceAPI::Global().get();
    *rv = static_cast<void*>(ptr);
  });

}
}
