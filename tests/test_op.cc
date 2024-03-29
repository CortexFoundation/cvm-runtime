#include <cvm/c_api.h>
#include <cvm/model.h>
#include <iostream>
#include <numeric>
#include <thread>
#include <omp.h>
#include <cvm/runtime/registry.h>
#include <cvm/op.h>
#include <cvm/op_attr_types.h>
#include <cvm/runtime/device_api.h>
#include <cvm/runtime/ndarray.h>
#include <cvm/runtime/packed_func.h>
#include <cvm/runtime/registry.h>
#include <cvm/runtime/serializer.h>
#include <cvm/runtime/param_dict.h>
#include <cvm/node.h>
#include <cvm/runtime/c_runtime_api.h>
#include "npy.hpp"
#include <string.h>
#include <fstream>
#include <dirent.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>

using namespace std;

using cvm::runtime::PackedFunc;
using cvm::runtime::Registry;
using namespace cvm;
using namespace cvm::runtime;

static 
std::unordered_map<int, DLDeviceType> const APIDevTypeMap = {
  {0, kDLCPU},
  {1, kDLGPU},
  {2, kDLFORMAL},
  {3, kDLOpenCL},
};

int dtype_code{kDLInt};
int dtype_bits{32};
int dtype_lanes{1};

struct CVMOpParam {
  std::string func_name;
  uint32_t num_inputs;
  uint32_t num_outputs;
  uint32_t flatten_data = false;
  std::string attrs;
};

#ifndef DEVICE
#define DEVICE 0
#endif 

int ctx = APIDevTypeMap.at(DEVICE);

int device_id = 0;
/*
30 52 -68 75
40 110 123 -100
9 -13 31 13

45 91 -93 -97
80 114 55 -68
68 -100 30 -82
*/
void LoadOp(string op_type, NodeAttrs& attrs) {
  if (op_type == "null") return;
  attrs.name = op_type;
  std::cerr << "op_type " << op_type << "\n";
  attrs.op = cvm::Op::Get(op_type);
  std::cerr << "op_type =====" << op_type << "\n";
}
void LoadOpAttr(std::string json_, NodeAttrs& attrs) {
  std::istringstream is(json_);
  utils::JSONReader reader(&is);
  reader.Read(&attrs.dict);
  if (attrs.op->attr_parser) {
    attrs.op->attr_parser(&attrs);
  }
}

struct OpArgs {
    std::vector<DLTensor> args;
    std::vector<CVMValue> arg_values;
    std::vector<int> arg_tcodes;
    std::vector<int64_t> shape_data;
};

std::function<void()> get_func(
    const CVMOpParam& param, NodeAttrs* attr,
    const std::vector<DLTensor>& args,
    size_t num_inputs,
    DLTensor* extra_space = nullptr)
{

  struct OpArgs {
    std::vector<DLTensor> args;
    std::vector<CVMValue> arg_values;
    std::vector<int> arg_tcodes;
    std::vector<int64_t> shape_data;
  };

  std::shared_ptr<OpArgs> arg_ptr = std::make_shared<OpArgs>();
  // setup address.
  arg_ptr->args = args;
  if (param.flatten_data) {
    arg_ptr->shape_data.resize(arg_ptr->args.size());
  }
  for (size_t i = 0; i < arg_ptr->args.size(); ++i) {
    CVMValue v;
    DLTensor* t = &(arg_ptr->args[i]);
    v.v_handle = t;
    arg_ptr->arg_values.push_back(v);
    arg_ptr->arg_tcodes.push_back(kArrayHandle);
    if (param.flatten_data) {
      arg_ptr->shape_data[i] = std::accumulate(
          t->shape, t->shape + t->ndim, 1, std::multiplies<int64_t>());
      t->ndim = 1;
      t->shape = &(arg_ptr->shape_data[i]);
    }
  }
  CVMValue t_attr;
  t_attr.v_handle = (void*)attr;
  arg_ptr->arg_values.push_back(t_attr);
  arg_ptr->arg_tcodes.push_back(kHandle);


  auto op = param.func_name;
  int device_type = static_cast<int>(ctx);
  std::string module_name = "cvm.runtime.";
  module_name += DeviceName(device_type);
  module_name += ".";
  auto func = cvm::runtime::Registry::Get(module_name + op);
  VERIFY(func != nullptr) << "function undefined " << module_name + op;
  return [arg_ptr, op, func, extra_space](){
    CVMRetValue rv;
    CVMArgs targs(
      arg_ptr->arg_values.data(),
      arg_ptr->arg_tcodes.data(),
      static_cast<int>(arg_ptr->arg_values.size()),
      extra_space
    );
    func->CallPacked(targs, &rv);
  };

  return [](){};
}
void test_depthwise_conv () {
    string attr_str = " {\"layout\": \"NCHW\", \"kernel_layout\": \"OIHW\", \"kernel_size\": \"[3, 3]\", \"padding\": \"(1, 1)\", \"use_bias\": \"True\", \"strides\": \"(1, 1)\", \"channels\": \"10\", \"dilation\": \"(1, 1)\", \"groups\": \"1024\"} ";
    std::vector<int> dims_ = {4, 4, 4};
    vector<std::vector<int64_t>> shapes_ = {{1, 1024, 7, 7}, {1024, 1, 3, 3}, {1, 1024, 7, 7}};
    CVMOpParam params;
    params.num_inputs = 2;
    params.num_outputs= 1;
    params.func_name = "conv2d";
    std::vector<DLTensor> args(params.num_inputs + params.num_outputs);
    for (uint32_t i = 0; i < args.size(); i++) {
      DLTensor* dl;
      CVMArrayAlloc(shapes_[i].data(), dims_[i], dtype_code, dtype_bits, dtype_lanes, ctx, device_id, &dl);
      args[i] = *dl;
    }

    std::vector<unsigned long> tshape;
    std::vector<int32_t> tdata;
    npy::LoadArrayFromNumpy("/tmp/conv2d_depthwise/in.x.npy", tshape, tdata);
    int32_t *dldata = static_cast<int32_t*>(args[0].data);
    memcpy(dldata, tdata.data(), sizeof(int32_t) * tdata.size());
    int n_c = shapes_[0][1];
    int i_h = shapes_[0][2];
    int i_w = shapes_[0][3];
    if (false) {
      for (int c = 0; c < shapes_[0][1]; c++) {
        for (int i = 0; i < shapes_[0][2]; i++) {
          for (int j = 0; j < shapes_[0][3]; j++) {
            std::cerr << dldata[(c) * i_h * i_w +  i * i_w + j] << " ";
          }
          std::cerr << "\n";
        }
        std::cerr << "\n";
      }
    }

    std::vector<unsigned long> tshape2;
    std::vector<int32_t> tdata2;
    int32_t *dldata2 = static_cast<int32_t*>(args[1].data);
    npy::LoadArrayFromNumpy("/tmp/conv2d_depthwise/in.w.npy", tshape2, tdata2);
    memcpy(dldata2, tdata2.data(), sizeof(int32_t) * tdata2.size());

    NodeAttrs attr;
    LoadOp(params.func_name, attr);
    LoadOpAttr(attr_str, attr);
    auto op_slice = get_func(params, &attr, args, params.num_inputs);
    op_slice();

    int32_t *dldata3 = static_cast<int32_t*>(args[2].data);
    std::vector<unsigned long> tshape3;
    std::vector<int32_t> tdata3;
    npy::LoadArrayFromNumpy("/tmp/conv2d_depthwise/out.y.npy", tshape3, tdata3);
    int ret =  memcmp(dldata3, tdata3.data(), sizeof(int32_t) * tdata3.size());
    printf("match %d | %d\n", ret == 0, ret);
    int o_h = shapes_[2][2];
    int o_w = shapes_[2][3];
    if (false) {
    std::cerr << "Expected\n";
    for (int c = 0; c < n_c; c++) {
      for (int i = 0; i < o_h; i++) {
        for (int j = 0; j < o_w; j++) {
          std::cerr << tdata3.data()[(c) * o_h * o_w +  i * o_w + j] << " ";
        }
        std::cerr << "\n";
      }
        std::cerr << "\n";
    }
    std::cerr << "Got\n";
    for (int c = 0; c < n_c; c++) {
      for (int i = 0; i < o_h; i++) {
        for (int j = 0; j < o_w; j++) {
          std::cerr << dldata3[(c ) * o_h * o_w +  i * o_w + j] << " ";
        }
        std::cerr << "\n";
      }
        std::cerr << "\n";
    }
    }
}
void test_transpose() {
    string attr_str = " {\"axes\": \"(1, 2, 0)\"} ";
    std::vector<int> dims_ = {3,  3};
    vector<std::vector<int64_t>> shapes_ = {{38, 32, 300},{32, 300, 38}};
    CVMOpParam params;
    params.num_inputs = 1;
    params.num_outputs= 1;
    params.func_name = "transpose";
    std::vector<DLTensor> args(params.num_inputs + params.num_outputs);
    for (uint32_t i = 0; i < args.size(); i++) {
      DLTensor* dl;
      CVMArrayAlloc(shapes_[i].data(), dims_[i], dtype_code, dtype_bits, dtype_lanes, ctx, device_id, &dl);
      args[i] = *dl;
    }

    std::vector<unsigned long> tshape;
    std::vector<int32_t> tdata;
    npy::LoadArrayFromNumpy("/tmp/transpose/in.x.npy", tshape, tdata);
    int32_t *dldata = static_cast<int32_t*>(args[0].data);
    memcpy(dldata, tdata.data(), sizeof(int32_t) * tdata.size());

    NodeAttrs attr;
    LoadOp(params.func_name, attr);
    LoadOpAttr(attr_str, attr);
    auto op_slice = get_func(params, &attr, args, params.num_inputs);
    op_slice();

    int32_t *dldata3 = static_cast<int32_t*>(args[1].data);
    std::vector<unsigned long> tshape3;
    std::vector<int32_t> tdata3;
    npy::LoadArrayFromNumpy("/tmp/transpose/out.y.npy", tshape3, tdata3);
    int ret =  memcmp(dldata3, tdata3.data(), sizeof(int32_t) * tdata3.size());
    printf("match %d | %d\n", ret == 0, ret);

    if (true) {
    std::cerr << "Expected\n";
    for (int c = 0; c < shapes_[0][0] *  shapes_[0][1] *  shapes_[0][2] ; c++) {
        std::cerr << tdata3.data()[c] << " ";
    }
    std::cerr << "\n";
    std::cerr << "Got\n";
    for (int c = 0; c < shapes_[1][0] *  shapes_[1][1] *  shapes_[1][2] ; c++) {
        std::cerr << dldata3[c] << " ";
    }
    std::cerr << "\n";
    }
}
void test_take() {
    string attr_str = " {\"axis\": \"0\"} ";
    CVMOpParam params;
    params.func_name = "take";
    params.num_inputs = 2;
    params.num_outputs= 1;
    std::vector<DLTensor> args(params.num_inputs + params.num_outputs);
    std::vector<std::vector<unsigned long>> tshape(args.size());
    std::vector<std::vector<int32_t>> tdata(args.size());
    npy::LoadArrayFromNumpy("/tmp/take/in.x.npy", tshape[0], tdata[0]);
    npy::LoadArrayFromNumpy("/tmp/take/in.w.npy", tshape[1], tdata[1]);
    npy::LoadArrayFromNumpy("/tmp/take/out.y.npy", tshape[2], tdata[2]);
    vector<std::vector<int64_t>> shapes_(args.size());
    std::vector<int> dims_(args.size());
    for (unsigned int idx = 0; idx < args.size(); idx++) {
      shapes_[idx].resize(tshape[idx].size());
      dims_[idx] = (tshape[idx].size());
      std::cout << tshape[idx].size() << "\n";
      for (unsigned int j = 0; j < shapes_[idx].size(); j++) {
        shapes_[idx][j] = tshape[idx][j];
        std::cout << tshape[idx][j] << " ";
      }
      std::cout << "\n";
    }
    DLTensor* cpu_tensor;
    for (uint32_t i = 0; i < args.size(); i++) {
      DLTensor* dl;
      CVMArrayAlloc(shapes_[i].data(), dims_[i], dtype_code, dtype_bits, dtype_lanes, ctx, device_id, &dl);
      args[i] = *dl;
      if (i < params.num_inputs) {
        CVMArrayAlloc(shapes_[i].data(), dims_[i], dtype_code, dtype_bits, dtype_lanes, kDLCPU, 0, &cpu_tensor);
        memcpy(cpu_tensor->data, tdata[i].data(), sizeof(int32_t) * tdata[i].size());
        CVMArrayCopyFromTo(cpu_tensor, dl, nullptr);
        CVMArrayFree(cpu_tensor);
      }
    }

    NodeAttrs attr;
    LoadOp(params.func_name, attr);
    LoadOpAttr(attr_str, attr);
    auto op_slice = get_func(params, &attr, args, params.num_inputs);
    op_slice();

    vector<int32_t> cpu_output_tensor(tdata[params.num_inputs].size());
    {
      int i = params.num_inputs; // first output
      CVMArrayAlloc(shapes_[i].data(), dims_[i], dtype_code, dtype_bits, dtype_lanes, kDLCPU, 0, &cpu_tensor);
      CVMArrayCopyFromTo(&args[i], cpu_tensor, nullptr);
      memcpy(cpu_output_tensor.data(), cpu_tensor->data, sizeof(int32_t) * tdata[i].size());
      CVMArrayFree(cpu_tensor);
    }
    int ret =  memcmp(cpu_output_tensor.data(),
                      tdata[params.num_inputs].data(),
                      sizeof(int32_t) * tdata[params.num_inputs].size());
    printf("match %d | %d\n", ret == 0, ret);

    //if (true) {
    //std::cerr << "Expected " << shapes_[0][0] << " " <<  shapes_[0][1] << " " << shapes_[0][2] << "\n";
    //for (int c = 0; c < tdata[params.num_inputs].size() ; c++) {
    //    std::cerr << tdata[params.num_inputs].data()[c] << " ";
    //}
    //std::cerr << "\n";
    //std::cerr << "Got " << shapes_[0][0] << " " <<  shapes_[0][1] << " " << shapes_[0][2] << "\n";
    //for (int c = 0; c < tdata[params.num_inputs].size() ; c++) {
    //    std::cerr << cpu_output_tensor[c] << " ";
    //}
    //std::cerr << "\n";
    //}
}
const int TYPE_FILE = 8;
const int TYPE_LINK = 10;
const int TYPE_DIR = 4;
int findAllSubDir(std::vector<string> &filelist, const char *basePath, int type)
{
    DIR *dir;
    struct dirent *ptr;

    if ((dir=opendir(basePath)) == NULL)
    {
        perror("Open dir error...");
        exit(1);
    }

    while ((ptr=readdir(dir)) != NULL)
    {
        if(strcmp(ptr->d_name,".")==0 || strcmp(ptr->d_name,"..")==0)    ///current dir OR parrent dir
            continue;
        if(ptr->d_type == type || ptr->d_type == TYPE_LINK){
            filelist.push_back(ptr->d_name);
        }
       // else if(ptr->d_type == TYPE_FILE && type == TYPE_FILE)    //file
       // {
       //   filelist.push_back(ptr->d_name);
       //     //printf("d_name:%s/%s\n",basePath,ptr->d_name);
       //    // string temp = ptr->d_name;
       //     //cout  << temp << endl;
       //    // string sub = temp.substr(temp.length() - 4, temp.length()-1);
       //    // //cout  << sub << endl;
       //    // if(sub == format)
       //    // {
       //    //     string path = basePath;
       //    //     path += "/";
       //    //     path += ptr->d_name;
       //    //     filelist.push_back(path);
       //    // }
       // }
       // else if(ptr->d_type == 10)    ///link file
       // {
       //     //printf("d_name:%s/%s\n",basePath,ptr->d_name);
       // }
       // else if(ptr->d_type == TYPE_DIR && type == TYPE_DIR)    ///dir
       // {
       //     memset(base,'\0',sizeof(base));
       //     strcpy(base,basePath);
       //     strcat(base,"/");
       //     strcat(base,ptr->d_name);
       //     filelist.push_back(ptr->d_name);
       //     findAllSubDir(filelist, base);
       // }
    }
    closedir(dir);
    return 1;
}
void read_one_line(string filename, string& str){
    ifstream infile;
    infile.open(filename);
    if(!infile.is_open()){
      printf("file no exist : %s\n", filename.c_str());
      str = "";
      return;
    }
    getline(infile, str);
    infile.close();
}
template<typename T>
void print(vector<T> &data){
  for(unsigned int i = 0; i < data.size() && i < 100; i++){
    printf("%d ", (int)data[i]);
  }
  printf("\n");

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
//    printf("shape: ");
    for(int i = 0; i < shape_dim; i++){
        int64_t value = 0;
        fscanf(fp, "%ld ", &value);
        shape[i] = value;
//        printf("%d ", shape[i]);
        size *= shape[i];
    }
//    printf("\n");
    //fscanf(fp, "\n");
//    printf("data: ");
    data.resize(size);
    for(unsigned int i = 0; i < size; i++){
        int32_t value = 0;
        fscanf(fp, "%d ", &value);
        data[i] = value;
//        printf("%d ", data[i]);
    }
//    printf("\n");
    fclose(fp);
}
void load_input(int num_inputs, string case_path, vector<vector<uint64_t>>& tshape,
    vector<vector<int32_t>>& tdata, vector<TShape>& ishape, vector<DLTensor>& args){
    DLTensor* cpu_tensor;
    for(int i = 0; i < num_inputs; i++){
      string in_path = case_path + "in_" +  std::to_string(i) + ".txt";
      //cout << in_path << endl;
      //npy::LoadArrayFromNumpy(in_path, tshape[in_i], tdata[in_i]);
      read_data(in_path.c_str(), tshape[i], tdata[i]);
      TShape shp(tshape[i].size());
      for (int ti = 0; ti < shp.ndim(); ++ti) {
        shp[ti] = tshape[i][ti];
      }
      ishape[i] = shp;
      // ishape.emplace_back(shp);
      std::cout << shp << std::endl;
      DLTensor* dl;
      CVMArrayAlloc((int64_t*)tshape[i].data(), tshape[i].size(), dtype_code, dtype_bits, dtype_lanes, ctx, device_id, &dl);
      args[i] = *dl;
      //if (i < params.num_inputs) {
      CVMArrayAlloc((int64_t*)tshape[i].data(), tshape[i].size(), dtype_code, dtype_bits, dtype_lanes, kDLCPU, 0, &cpu_tensor);
      memcpy(cpu_tensor->data, tdata[i].data(), sizeof(int32_t) * tdata[i].size());
      CVMArrayCopyFromTo(cpu_tensor, dl, nullptr);
      printf("call CVMArrayFree by manual.....\n");
      CVMArrayFree(cpu_tensor);
    }
}
const string CASE_DIR = "/data/ops_generator";
//const string CASE_DIR = "/data/zkh";

void test_op(string op_name) {
  printf("\ntest %s\n", op_name.c_str());
	std::vector<string> case_list;
	string case_dir = CASE_DIR + "/" + op_name + "/";
	findAllSubDir(case_list, case_dir.c_str(), TYPE_DIR);

  static auto& finfer_shape =
      Op::GetAttr<cvm::FInferNodeEntryAttr<TShape> >("FInferShape");
  const cvm::Op *op = cvm::Op::Get(op_name);
  auto finfer = finfer_shape.get(op, nullptr);
  if (finfer == nullptr) {
    std::cout << "operator " << op_name
      << "has not registered FInferShape";
    return ;
  }

  static auto& finfer_prec =
      Op::GetAttr<cvm::FInferPrecision>("FInferPrecision");
  std::vector<int> iprec, oprec;
  std::vector<TShape> shapes;
  auto fip = finfer_prec.get(op, nullptr);
  if (fip == nullptr) {
    std::cout << "operator " << op_name
      << "has not registered FInferPrecision";
    return ;
  }

  static auto& fextra_space =
      Op::GetAttr<cvm::FOpExtraSpace >("FOpExtraSpace");
  auto fextra = fextra_space.get(op, nullptr);

  for(unsigned int ci = 0; ci < case_list.size(); ci++){ // for each test cases
		string case_path = case_dir + case_list[ci] + "/";
    cout << "doing test case at " << case_path << endl;
    string attr_path = case_path + "attr.txt";
    string attr_str = "";
    read_one_line(attr_path, attr_str);
    //string attr_str = " {\"axis\": \"" + std::to_string(i-1) + "\"} ";
    std::cout << attr_str << endl;
    vector<string> file_list;
    findAllSubDir(file_list, case_path.c_str(), TYPE_FILE);
    unsigned int num_inputs = 0, num_outputs = 0;  // num_outputs is the expected number of output
    for(auto file_name : file_list){
        if(file_name.find("in_") != string::npos){
            num_inputs += 1;
        }
        if(file_name.find("out_") != string::npos){
            num_outputs += 1;
        }
    }
   // printf("num_inputs = %d, num_outputs = %d\n", num_inputs, num_outputs);
    //if(num_inputs == 0 || num_outputs == 0) continue;

    NodeAttrs attr;
    LoadOp(op_name, attr);
    LoadOpAttr(attr_str, attr);

    CVMOpParam params;
    params.func_name = op_name;
    params.num_inputs = num_inputs;
    //params.num_outputs= num_outputs;
    params.num_outputs =
        attr.op->get_num_outputs ? attr.op->get_num_outputs(attr) : attr.op->num_outputs;
    params.flatten_data = false;
    std::vector<DLTensor> args(params.num_inputs + params.num_outputs);
    std::vector<std::vector<unsigned long>> tshape(args.size());
    std::vector<std::vector<int32_t>> tdata(args.size());
    std::vector<TShape> ishape(params.num_inputs), oshape(params.num_outputs);
    load_input(params.num_inputs, case_path, tshape, tdata, ishape, args);

    // infer shapes
    bool infer_shape_ret;
    string err_path = case_path + "err.txt", err_str = "";
    read_one_line(err_path, err_str);
    try {
      infer_shape_ret = finfer(attr, &ishape, &oshape);
      if(infer_shape_ret){
        std::cout << "FInferShape ishape=[";
        for (auto& shp : ishape) std::cout << shp << ", ";
        std::cout << "] oshape=[";
        for (auto& shp : oshape) std::cout << shp << ", ";
        std::cout << "]\n";
      }
    } catch (const std::exception& e) {
      std::cerr << "FInferShape error with " << e.what() << std::endl;
      infer_shape_ret = false;
    }
    if(infer_shape_ret == false){
      std::cout << "error after FInferShape: " << err_str << std::endl;
      if(err_str == ""){
        string out_path = case_path + "out_0.txt";
        std::cout << out_path << std::endl;
        //npy::LoadArrayFromNumpy(out_path, tshape[num_inputs], tdata[num_inputs]);
        read_data(out_path.c_str(), tshape[num_inputs], tdata[num_inputs]);
        print(tdata[num_inputs]);
        assert(false);
        continue;
      }else{
        cout << "match 1 | 0" << endl;
        continue;
      }
    }

    std::vector<TShape> shapes;
    std::vector<int> iprecs;
    shapes.insert(shapes.end(), ishape.begin(), ishape.end());
    shapes.insert(shapes.end(), oshape.begin(), oshape.end());
    int64_t es[1]{ 0 };
    if (fextra != nullptr) {
      es[0] = fextra(attr, &ishape, &iprecs,
          DLContext{DLDeviceType(ctx), 0});
    }
    DLTensor* extra_space;
    CVMArrayAlloc(es, 1, 
        dtype_code, dtype_bits, dtype_lanes, ctx, device_id, &extra_space);

    // compare num of expected and actual output and shape of each.
    if (params.num_outputs != num_outputs) {
      std::cout << "error with FInfershape or op.get_num_outputs()\n"
                << "should be " << num_outputs << " outputs but we calculated "
                << params.num_outputs << std::endl;
      assert(false);
    }
    for(unsigned int i = 0; i < num_outputs; i++){
			string out_path = case_path + "out_" + std::to_string(i) + ".txt";
			//cout << out_path << endl;
			//npy::LoadArrayFromNumpy(out_path, tshape[num_inputs+i], tdata[num_inputs+i]);
      read_data(out_path.c_str(), tshape[num_inputs+i], tdata[num_inputs+i]);
 //     print(tshape[num_inputs+i]);
 //     print(tdata[num_inputs+i]);
      int shape_cmp = memcmp(tshape[num_inputs+i].data(), oshape[i].data(), sizeof(int64_t) * tshape[num_inputs+i].size());
      if(shape_cmp != 0){
        print(tshape[num_inputs+i]);
        //print(oshape[i]);
        std::cout << oshape[i] << endl;
      }
      assert(shape_cmp == 0);
      DLTensor* dl;
      CVMArrayAlloc((int64_t*)tshape[num_inputs+i].data(), tshape[num_inputs+i].size(), dtype_code, dtype_bits, dtype_lanes, ctx, device_id, &dl);
      args[num_inputs + i] = *dl;
    }

    // do the calculation here. get the operator's implemention and call it.
    auto op = get_func(params, &attr, args, params.num_inputs, extra_space);
    op();

    // compare each expected output and actual output
    for (uint out_no = 0; out_no < num_outputs; out_no++) { // out_no means which output data we are comparing. out_0 or out_1
      vector<int32_t> cpu_output_tensor(tdata[params.num_inputs+out_no].size());
      {
        DLTensor* cpu_tensor;
        int i = params.num_inputs;  // first output
        CVMArrayAlloc((int64_t*)tshape[i+out_no].data(), tshape[i+out_no].size(), dtype_code,
                      dtype_bits, dtype_lanes, kDLCPU, 0, &cpu_tensor);
        CVMArrayCopyFromTo(&args[i+out_no], cpu_tensor, nullptr);
        memcpy(cpu_output_tensor.data(), cpu_tensor->data,
               sizeof(int32_t) * tdata[i+out_no].size());
        printf("call CVMArrayFree by manual.....\n");
        CVMArrayFree(cpu_tensor);
        //    CVMArrayFree(&args[i]);
      }
      int ret =
          memcmp(cpu_output_tensor.data(), tdata[params.num_inputs+out_no].data(),
                 sizeof(int32_t) * tdata[params.num_inputs+out_no].size());
      printf("match %d | %d\n", ret == 0, ret);
      if (ret != 0) {
        for (uint i = 0; i < num_inputs; i++) {
          printf("input%d:\n", i);
          print(tdata[i]);
        }
        printf("correct out:");
        print(tdata[num_inputs+out_no]);
        printf("     my out:");
        print(cpu_output_tensor);
      }
      assert(ret == 0);
      printf("\n");
      CVMArrayFree(extra_space); 
    }
    // for(int i = 0; i < args.size(); i++){
      // CVMArrayFree(&args[i]);
    // }
  }
}
int main() {
  test_op("max_pool2d"); // formal & cpu pass
  test_op("upsampling"); // formal & cpu pass
  test_op("dense");  // formal & cpu pass
  test_op("conv2d"); // formal & cpu pass
  test_op("sum");  // formal & cpu pass
  test_op("max");  // formal & cpu pass
  test_op("slice_like"); // formal & cpu pass
  test_op("tile"); // formal & cpu pass
  test_op("repeat"); // formal & cpu pass
  test_op("get_valid_counts");  // formal & CPU pass

  test_op("strided_slice"); // formal & cpu pass
  test_op("concatenate");// formal & cpu pass
  test_op("transpose");// formal & cpu pass
  test_op("take"); // formal & cpu pass
  //test_op("clip"); // no test case
  //test_op("cvm_clip"); // no test case
  //test_op("cvm_right_shift");  // no test case
  test_op("elemwise_add"); // formal & cpu pass
  //test_op("elemwise_sub"); // no test case
  //test_op("where"); // no test case
  test_op("non_max_suppression");  // formal & cpu pass
  test_op("broadcast_sub"); // formal & cpu pass
  test_op("broadcast_add");  // formal & cpu pass
  test_op("broadcast_mul");  // formal & cpu pass
  test_op("broadcast_max");  // formal & cpu pass
  test_op("broadcast_div");  // formal & cpu pass
  test_op("broadcast_greater");  // formal & cpu pass
  
  cout << "all tests finished" << endl;
  return 0;
}
