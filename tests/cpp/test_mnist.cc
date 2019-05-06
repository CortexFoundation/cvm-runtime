#include <cvm/dlpack.h>
#include <cvm/runtime/module.h>
#include <cvm/runtime/registry.h>
#include <cvm/runtime/packed_func.h>

#include <fstream>
#include <iterator>
#include <algorithm>
#include <stdio.h>
#include <string.h>

#include <time.h>
#include <npy.hpp>
#include "omp.h"

int dtype_code = kDLInt;
int dtype_bits = 32;
int dtype_lanes = 1;
int device_type = kDLCPU;
int device_id = 0;

long RunCVM(DLTensor* x, CVMByteArray& params_arr, std::string json_data,
        cvm::runtime::Module &mod_syslib  ,  std::string runtime_name, DLTensor *y, int devicetype) {
		auto t1 = clock();
	// get global function module for graph runtime
    auto mf =  (*cvm::runtime::Registry::Get("cvm." + runtime_name + ".create"));
    cvm::runtime::Module mod = mf(json_data, mod_syslib, static_cast<int>(x->ctx.device_type), device_id);

    // load image data saved in binary
    // std::ifstream data_fin("cat.bin", std::ios::binary);
    // data_fin.read(static_cast<char*>(x->data), 3 * 224 * 224 * 4);

    // get the function from the module(set input data)
    cvm::runtime::PackedFunc set_input = mod.GetFunction("set_input");
    set_input("data", x);


    // get the function from the module(load patameters)
    cvm::runtime::PackedFunc load_params = mod.GetFunction("load_params");
    load_params(params_arr);


		auto t2 = clock();
    // get the function from the module(run it)
    cvm::runtime::PackedFunc run = mod.GetFunction("run");
    run();
    // get the function from the module(get output data)
    cvm::runtime::PackedFunc get_output = mod.GetFunction("get_output");
    get_output(0, y);

//    auto y_iter = static_cast<int*>(y->data);
//    // get the maximum position in output vector
//    auto max_iter = std::max_element(y_iter, y_iter + out_shape[1]);
//    auto max_index = std::distance(y_iter, max_iter);
//    std::cout << "The maximum position in output vector is: " << max_index << std::endl;

//    for (auto i = 0; i < out_shape[1]; i++) {
//        if (i < 1000)
//            std::cout << y_iter[i] << " ";
//    }
//    std::cout << "\n";
//
//
//    CVMArrayFree(y);
}

int main()
{
    cvm::runtime::Module mod_syslib = (*cvm::runtime::Registry::Get("module._GetSystemLib"))();

    for(int in = 0; in < 1; in++){

        std::cout << "start load data and params\n";
        clock_t load_start = clock();
        std::vector<unsigned long> tshape;
        std::vector<unsigned char> tdata;
        npy::LoadArrayFromNumpy("/tmp/data.npy", tshape, tdata);

        DLTensor* x;
        int in_ndim = 4;
        int64_t in_shape[4] = {1, 1, 28, 28};
        CVMArrayAlloc(in_shape, in_ndim, dtype_code, dtype_bits, dtype_lanes, device_type, device_id, &x);
        auto x_iter = static_cast<int*>(x->data);
        for (auto i = 0; i < 1*28*28; i++) {
        //    std::cout << (int)((int8_t)tdata[i]) << " ";
        //    if(i != 0 && i %28 == 0) std::cout << std::endl;
            x_iter[i] = (int)((int8_t)tdata[i]);
        }

        // parameters in binary
        std::ifstream params_in("/tmp/mnist_yxnet_deploy.params", std::ios::binary);
        std::string params_data((std::istreambuf_iterator<char>(params_in)), std::istreambuf_iterator<char>());
        params_in.close();

        // parameters need to be CVMByteArray type to indicate the binary data
        CVMByteArray params_arr;
        params_arr.data = params_data.c_str();
        params_arr.size = params_data.length();


        int out_ndim = 2;
        int64_t out_shape[2] = {1, 10, };

        std::ifstream json_in("/tmp/mnist_yxnet_deploy.symbol", std::ios::in);
        std::string json_data((std::istreambuf_iterator<char>(json_in)), std::istreambuf_iterator<char>());
        json_in.close();

        DLTensor* gpu_x, *gpu_y;
        CVMArrayAlloc(in_shape, in_ndim, dtype_code, dtype_bits, dtype_lanes, kDLGPU, device_id, &gpu_x);
        CVMArrayAlloc(out_shape, out_ndim, dtype_code, dtype_bits, dtype_lanes, kDLGPU, device_id, &gpu_y);
        CVMStreamHandle stream1;
        CVMStreamCreate(kDLGPU, device_id, &stream1);
        CVMArrayCopyFromTo(x, gpu_x, stream1);

        DLTensor* y2;
        CVMArrayAlloc(out_shape, out_ndim, dtype_code, dtype_bits, dtype_lanes, device_type, device_id, &y2);
        clock_t load_end = clock();
        std::cout << "load data and params use: " << (load_end - load_start)*1.0/CLOCKS_PER_SEC << "s" << std::endl;
        std::cout << "start run..\n";
        double cvm_start = omp_get_wtime();
        for (int i = 0; i < 1; i++) {
            RunCVM(x, params_arr, json_data, mod_syslib, "cvm_runtime", y2, (int)kDLCPU);
        }
        double cvm_end = omp_get_wtime();
        std::cout << (cvm_end - cvm_start)/10 << "s" << std::endl;
//        CVMArrayCopyFromTo(gpu_y, y2, stream1);
        //CVMArrayFree(y_cpu);
        for(int i = 0; i < 10; i++){
            std::cout << static_cast<int32_t*>(y2->data)[i] << " ";
        }
        std::cout << std::endl;
        int32_t ret[] = {-47, -6, -28, 95, -34, 66, -54, -8, -83, -35};
        std::cout << (memcmp(ret, y2->data, 10*sizeof(int32_t)) == 0 ? "pass" : "failed") << std::endl;
//        CVMArrayFree(x);
//        CVMArrayFree(gpu_x);
//        CVMArrayFree(gpu_y);
        //    CVMArrayFree(t_gpu_x);
        //    CVMArrayFree(t_gpu_y);
//        CVMArrayFree(y2);

    }
    return 0;
}
