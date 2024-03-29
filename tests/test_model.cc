#include <iostream>
#include <thread>
#include <omp.h>
#include <vector>
#include <fstream>

#include <cvm/c_api.h>
#include <cvm/model.h>
#include <cvm/runtime/registry.h>
#include <cvm/op.h>
#include "npy.hpp"
using namespace std;

using cvm::runtime::PackedFunc;
using cvm::runtime::Registry;

#ifndef DEVICE
#define DEVICE  0
#endif

#define CHECK_STATUS(x, msg) \
  if (x != SUCCEED) { \
    cout << "STATUS ERROR: " << x << " " << msg << "\n"; \
    return -1; \
  }

void read_data(const char *filename, vector<unsigned long> &shape, vector<int32_t>& data){
    FILE *fp = fopen(filename, "r");
    if(fp == NULL){
        return;
    }
    int32_t shape_dim = 0;
    fscanf(fp, "%d ", &shape_dim);
    printf("shape_dim = %d\n", shape_dim);
    shape.resize(shape_dim);
    uint64_t size = 1;
    for(int i = 0; i < shape_dim; i++){
        int64_t value = 0;
        fscanf(fp, "%ld ", &value);
        shape[i] = value;
        size *= shape[i];
    }
    data.resize(size);
    for(int i = 0; i < size; i++){
        int32_t value = 0;
        fscanf(fp, "%d ", &value);
        data[i] = value;
    }
    fclose(fp);
}

void write_result(const char *filename, vector<char>& data){
    FILE* fp = fopen(filename, "w");
    if(fp == NULL){
        printf("open file %s failed\n", filename);
        return;
    }
    fprintf(fp, "%lu\n", data.size());
    for(int i = 0; i < data.size(); i++){
        fprintf(fp, "%d ", data[i]);
    }
    fprintf(fp, "\n");
    fclose(fp);
}

void compare_result(const char *filename, vector<char>& data){
    FILE* fp = fopen(filename, "r");
    if(fp == NULL){
      printf("save result..\n");
        write_result(filename, data);
        // printf("open file %s failed\n", filename);
        return;
    }
    int n = 0;
    fscanf(fp, "%d", &n);
    assert(n == data.size());

    for(int i = 0; i < data.size(); i++){
      int value;
      fscanf(fp, "%d ", &value);
      assert((int32_t)data[i] == value);
    }
    fclose(fp);
    printf("compare result: success\n\n");
}

struct OpArgs {
  std::vector<DLTensor> args;
  std::vector<CVMValue> arg_values;
  std::vector<int> arg_tcodes;
  std::vector<int64_t> shape_data;
};

int run_LIF(string model_root, int device_type = 0) {

  //printf("the elewise cnt = %.4f\n", cvm::runtime::cvm_op_elemwise_cnt);
  string json_path = model_root + "/symbol";
  string params_path = model_root + "/params";
  cout << "load " << json_path << "\n";
  cout << "load " << params_path << "\n";
  std::string params, json;
  {
    std::ifstream input_stream(json_path, std::ios::binary);
    json = string((std::istreambuf_iterator<char>(input_stream)), std::istreambuf_iterator<char>());
    input_stream.close();
  }
  {
    std::ifstream input_stream(params_path, std::ios::binary);
    params  = string((std::istreambuf_iterator<char>(input_stream)), std::istreambuf_iterator<char>());
    input_stream.close();
  }
  void *net;
  auto status = CVMAPILoadModel(json.c_str(), json.size(),
                                params.c_str(), params.size(),
                                &net,
                                device_type, 0);
  cout<< "model loaded\n";
  CHECK_STATUS(status, "model loaded failed");

  unsigned long long gas = 0;
  status = CVMAPIGetGasFromModel(net, &gas);
  CHECK_STATUS(status, "gas invalid");
  cout << "ops " << gas / 1024 / 1024 << "\n";
  // API only accepts byte array
  vector<char> input, output;
  unsigned long long input_size, output_size;
  CVMAPIGetInputLength(net, &input_size);
  CVMAPIGetOutputLength(net, &output_size);
  input.resize(input_size, 0); // 1 * 1 * 28 * 28);
  output.resize(output_size, 0); //1 * 10);
  if (model_root.find("trec") != string::npos)
  {
    vector<int32_t> input_int32_t;
    std::vector<unsigned long> tshape;
    npy::LoadArrayFromNumpy(model_root + "/data.npy", tshape, input_int32_t);
    std::cout << "Loading a int32 data and cast to byte array: "
              << input.size() << " " << input_int32_t.size() << "\n";
    memcpy(input.data(), input_int32_t.data(), input.size());
  }
  else if (model_root.find("yolo") != string::npos)
  {
    std::vector<unsigned long> tshape;
    npy::LoadArrayFromNumpy(model_root + "/data.npy", tshape, input);
    std::cerr << tshape.size() << "\n";
    for (auto x : tshape) {
      std::cout << x << " ";
    }
    std::cout << "\n";
  }
  else if (model_root.find("std_out") != string::npos)
  {
    string data_file = model_root + "/data.npy";
    std::vector<unsigned long> tshape;
    npy::LoadArrayFromNumpy(data_file, tshape, input);
    std::cout << tshape.size() << "\n";
    for (auto x : tshape) {
      std::cout << x << " ";
    }
    std::cout << "\n";
  }
  else if (model_root.find("3145ad19228c1cd2d051314e72f26c1ce77b7f02") != string::npos)
  {
    string data_file =  model_root + "/cpu.txt";
    std::vector<unsigned long> tshape;
    std::vector<int32_t> data;
    //npy::LoadArrayFromNumpy(data_file, tshape, input);
    read_data(data_file.c_str(), tshape, data);
    std::cout << tshape.size() << "\n";
    for (int i = 0; i < data.size(); i++) {
      input[i]= (int8_t)data[i];
      if(i < 10){
        printf("%d ", input[i]);
      }
    }
    printf("\n");
  }

  status = CVMAPIInference(net, input.data(), input.size(), output.data());
  CHECK_STATUS(status, "inference failed");
  status = CVMAPIFreeModel(net);
  CHECK_STATUS(status, "free model failed");

  if (json_path.find("yolo") != string::npos || json_path.find("ssd") != string::npos) {
    uint64_t n_bytes = 4;
    uint64_t ns =  output.size() / n_bytes;
    std::cout << "yolo output size = " << ns << " n_bytes = " << n_bytes << "\n";
    int32_t* int32_output = static_cast<int32_t*>((void*)output.data());
    for (auto i = 0; i < std::min(60UL, ns); i++) {
      std::cout << (int32_t)int32_output[i] << " ";
      if ((i + 1) % 6 == 0)
        std::cout << "\n";
    }
    // last 60 rows of results
    if (ns > 60) {
      for (auto i = (size_t)(std::max(0, ((int)(ns) - 60))); i < ns; i++) {
        std::cout << (int32_t)int32_output[i] << " ";
        if ((i + 1) % 6 == 0)
          std::cout << "\n";
      }
    }
    std::cout << "\n";
  } else {
    std::cout << "output size = " << output.size() << "\n";
    for (auto i = 0; i < std::min(6UL * 10, output.size()); i++) {
      std::cout << (int32_t)output[i] << " ";
    }
    std::cout << "\n";
    if (output.size() > 60) {
      for (auto i = (size_t)(std::max(0, ((int)(output.size()) - 6 * 10))); i < output.size(); i++) {
        std::cout << (int32_t)output[i] << " ";
      }
      std::cout << "\n";
    }

   // string data_file = model_root + "/result_0.npy";
   // vector<unsigned long> tshape;
   // vector<int32_t> tout;
   // npy::LoadArrayFromNumpy(data_file, tshape, tout);
   // cout << tout.size() << " " << output.size() << endl;
   // for(int i = 0; i < tout.size() && i < 60; i++){
   //   cout << tout[i] << " ";
   // }
   // cout << endl;
   // for(int i = 0; i < tout.size(); i++){
   //     if((int32_t)output[i] != tout[i]){
   //        cout << "failed!!!!! : " << i << " " << (int32_t)output[i] << " " << (int32_t)tout[i] << endl;
   //     }
   //     assert((int32_t)output[i] == tout[i]);
   // }
  }
  string out_file = model_root + "/result_0.txt";
  // write_result(out_file.c_str(), output);
  compare_result(out_file.c_str(), output);
  return 0;
}
void test_thread() {
  vector<std::thread> threads;
  for (int t = 0; t < 1; ++t) {
    cout << "threads t = " << t << "\n";
    threads.push_back(thread([&]() {
          string model_root = "/home/tian/model_storage/resnet50_v1/data/";
          // model_root = "/home/kaihuo/cortex_fullnode_storage/cifar_resnet20_v2/data";
          // model_root = "/home/tian/storage/mnist/data/";
          // model_root = "/home/tian/storage/animal10/data";
          // model_root = "/home/kaihuo/cortex_fullnode_storage/imagenet_inceptionV3/data";
          run_LIF(model_root);
          //run_LIF(model_root);
          }));
  }
  for (auto& thread : threads) {
    thread.join();
  }
}

int test_models(int device_type = 0) {
  std::cout << device_type << " DDDDDD" << std::endl;
  std::string model_root = "/data1/";
  auto model_dirs = {
    "std_out/yolo_tfm",
    "std_out/null",
    "std_out/resnet50_mxg",
    "std_out/ssd_512_mobilenet1.0_voc_tfm",
    "std_out/resnet18_v1_tfm",
    "std_out/resnet50_v2",
    "std_out/qd10_resnet20_v2",
    "std_out/trec",
    // "new_cvm/yolo3_darknet53_voc/data",
    "lz_model_storage/dcnet_mnist_v1/data",
    "lz_model_storage/mobilenetv1.0_imagenet/data",
    "lz_model_storage/resnet50_v1_imagenet/data",
    "lz_model_storage/animal10/data",
    "lz_model_storage/resnet50_v2/data",
    "lz_model_storage/vgg16_gcv/data",
    "lz_model_storage/sentiment_trec/data",
    "lz_model_storage/vgg19_gcv/data",
    "lz_model_storage/squeezenet_gcv1.1/data",
    "lz_model_storage/squeezenet_gcv1.0/data",
    // invalid has strange attribute in operator elemwise_add.
    // "lz_model_storage/octconv_resnet26_0.250/data",
    "std_out/resnet50_mxg/",
    "std_out/resnet50_v2",
    "std_out/qd10_resnet20_v2",
    "std_out/random_3_0/",
    "std_out/random_3_1/",
    "std_out/random_3_2/",
    "std_out/random_3_3/",
    "std_out/random_3_4/",
    "std_out/random_3_5/",
    "std_out/random_4_0/",
    "std_out/random_4_1/",
    // "std_out/random_4_2/",
    // "std_out/random_4_3/",
    // "std_out/random_4_4/",
    "std_out/random_4_5/",
    "std_out/random_4_6/",
    "std_out/random_4_7/",
    "std_out/random_4_8/",
    "std_out/random_4_9/",
    "std_out/log2",
    //"./tests/3145ad19228c1cd2d051314e72f26c1ce77b7f02/",
    "std_out/lr_attr",
    // "std_out/non_in",
    "std_out/shufflenet",
    "std_out/ssd",
    "std_out/ssd_512_mobilenet1.0_coco_tfm/",
  };
  for (auto dir : model_dirs) {
    auto ret = run_LIF(model_root + dir, device_type);
    if (ret == -1) return -1;
  }
  return 0;
}
int main() {
 if (test_models(DEVICE) != 0)
   return -1;
  return 0;
}
