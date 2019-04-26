#include <dlpack/dlpack.h>
#include <tvm/runtime/module.h>
#include <tvm/runtime/registry.h>
#include <tvm/runtime/packed_func.h>

#include <fstream>
#include <iterator>
#include <algorithm>

using namespace tvm::runtime;

int main()
{
    // tvm module for compiled functions
    //tvm::runtime::Module mod_syslib = tvm::runtime::Module::LoadFromFile("/tmp/imagenet.so");
	tvm::runtime::Module mod_syslib = (*tvm::runtime::Registry::Get("module._GetSystemLib"))();

    // json graph
    std::ifstream json_in("/tmp/imagenet_cuda.json", std::ios::in);
    std::string json_data((std::istreambuf_iterator<char>(json_in)), std::istreambuf_iterator<char>());
    json_in.close();

    // parameters in binary
    std::ifstream params_in("/tmp/imagenet.params", std::ios::binary);
    std::string params_data((std::istreambuf_iterator<char>(params_in)), std::istreambuf_iterator<char>());
    params_in.close();

    // parameters need to be TVMByteArray type to indicate the binary data
    TVMByteArray params_arr;
    params_arr.data = params_data.c_str();
    params_arr.size = params_data.length();

    int dtype_code = kDLInt;
    int dtype_bits = 32;
    int dtype_lanes = 1;
    int device_type = kDLCPU;
    int device_id = 0;
		int64_t ops = (*tvm::runtime::Registry::Get("tvm.cvm_runtime.estimate_ops"))(json_data);
		std::cout << "ops " << ops << std::endl;
		// get global function module for graph runtime
    tvm::runtime::Module mod = (*tvm::runtime::Registry::Get("tvm.cvm_runtime.create"))(json_data, mod_syslib, device_type, device_id);

    DLTensor* x;
    int in_ndim = 2;
    int64_t in_shape[2] = {1, 28};
    TVMArrayAlloc(in_shape, in_ndim, dtype_code, dtype_bits, dtype_lanes, device_type, device_id, &x);
    auto x_iter = static_cast<int*>(x->data);
    for (auto i = 0; i < 28; i++) {
        x_iter[i] = i;
    }
    std::cout << "\n";
    // load image data saved in binary
    // std::ifstream data_fin("cat.bin", std::ios::binary);
    // data_fin.read(static_cast<char*>(x->data), 3 * 224 * 224 * 4);

    // get the function from the module(set input data)
    tvm::runtime::PackedFunc set_input = mod.GetFunction("set_input");
    set_input("data", x);
    {
        DLTensor* w;
        int win_ndim = 2;
        int64_t win_shape[2] = {16, 28};
        TVMArrayAlloc(win_shape, win_ndim, dtype_code, dtype_bits, dtype_lanes, device_type, device_id, &w);
        for (auto i = 0; i < 16 * 28; i++) {
            static_cast<int*>(w->data)[i] = i / 10;
        }
        set_input("dense0_weight", w);
    }
    {
        DLTensor* w1;
        int w1in_ndim = 2;
        int64_t w1in_shape[2] = {10, 16};
        TVMArrayAlloc(w1in_shape, w1in_ndim, dtype_code, dtype_bits, dtype_lanes, device_type, device_id, &w1);
        for (auto i = 0; i < 10 * 16; i++) {
            static_cast<int*>(w1->data)[i] = i / 10;
        }
        set_input("dense1_weight", w1);
    }

    // get the function from the module(run it)
    tvm::runtime::PackedFunc run = mod.GetFunction("run");
    run();

    DLTensor* y;
    int out_ndim = 2;
    int64_t out_shape[2] = {1, 10, };
    TVMArrayAlloc(out_shape, out_ndim, dtype_code, dtype_bits, dtype_lanes, device_type, device_id, &y);

    // get the function from the module(get output data)
    tvm::runtime::PackedFunc get_output = mod.GetFunction("get_output");
    get_output(0, y);

    // DLTensor* y_cpu;
    // TVMArrayAlloc(y->shape, y->ndim, y->dtype.code, y->dtype.bits, y->dtype.lanes, kDLCPU, 0, &y_cpu);
    // TVMArrayCopyFromTo(y_cpu, y, nullptr);

    // get the maximum position in output vector
    auto y_iter = static_cast<int*>(y->data);
    for (auto i = 0; i < 10; i++) {
        std::cout << y_iter[i] << " ";
    }
    std::cout << "\n";

    TVMArrayFree(x);
    TVMArrayFree(y);
    // TVMArrayFree(y_cpu);

    return 0;
}
