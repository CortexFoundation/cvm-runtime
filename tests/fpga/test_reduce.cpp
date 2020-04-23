#include <CL/opencl.h>
#include <iostream>
#include <memory>
#include <functional>
#include <string>
#include <vector>
#include <algorithm>
#include <iostream>
#include <fstream>
#include <string.h>
#include "util.hpp"
using namespace std;

typedef vector<int> TShape;

inline std::vector<int> GetRealAxis(TShape& axis, bool exclude, DLTensor *x){
  for(size_t i = 0; i < axis.size(); i++){
    if(axis[i] < 0) axis[i] += x->ndim;
  }
  std::vector<int> raxis;
  if(!exclude){
    for(size_t i = 0; i < axis.size(); i++){
      raxis.push_back(axis[i]);
    }
  }else{
    raxis.resize(x->ndim - axis.size());
    for(int i = 0, k = 0; i < x->ndim; i++){
      bool flag = false;
      for(size_t j = 0; j < axis.size(); j++){
        if(axis[j] == i) {
          flag = true;
          break;
        }
      }
      if(!flag){
        raxis[k++] = i;
      }
    }
  }
  return raxis;
}

typedef std::function<void(int&, int)> reduce_func;
void Reduce(DLTensor *x, DLTensor *y, TShape& axis, bool exclude, reduce_func const &f){
  int *x_data = static_cast<int*>(x->data);
  int *y_data = static_cast<int*>(y->data);

  std::vector<int> realAxis = GetRealAxis(axis, exclude, x);

  if(exclude && realAxis.size() == 0){
    memcpy(y_data, x_data, getSize(x) * sizeof(int));
  } else if (realAxis.size() == 0) {
    int tmp = 0;
    for(int i = 0; i < getSize(x); i++){
      f(tmp, x_data[i]);
    }
    y_data[0] = tmp;
  } else {
    std::vector<bool> flag(x->ndim, false);
    for(uint i = 0; i < realAxis.size(); i++){
      int val = realAxis[i];
      flag[val] = true;
    }
    std::sort(realAxis.begin(), realAxis.end());

    int axis_size = 1;
    for(uint i = 0; i < realAxis.size(); i++){
      axis_size *= x->shape[realAxis[i]];
    }
    std::vector<int> every_xdim_size(x->ndim, 1);
    for(int i = x->ndim-2; i >= 0; i--){
      every_xdim_size[i] = x->shape[i+1] * every_xdim_size[i+1];
    }
    // foreach yshp, remove the dimension equals 1
    // special case: input shape is (1,), considered.
    int yndim = y->ndim;
    std::vector<int> yshape(y->ndim, 1);
    for(int i = 0, j = 0; i < y->ndim; i++){
      if(y->shape[i] == 1) {
        yndim -= 1;
      }else{
        yshape[j++] = y->shape[i];
      }
    }
    /* For example:
     * xshp : (n1, n2, n3, n4) -> yshp(n1, n4)
     * find x indexes reduce to yi with two steps:
     *  1. x start index, xvar(n1, 0, 0, n4)
     *  2. foreach reduce axis dimension(n2, n3), 
     *      get x possible indexes and add value to result.
     **/
    for(int i = 0; i < getSize(y); i++){
      int in_i = 0, o_i = i;
      for(int j = yndim-1, xj = x->ndim-1; j>=0; j--){
        int col = o_i % yshape[j];
        o_i /= yshape[j];
        while(xj >= 0 && flag[xj--]);
        in_i += col * every_xdim_size[xj+1];
      }
      int tmp = x_data[in_i];
      for(int xi = 1; xi < axis_size; xi++){
        int o_i = xi, tmp_in_i = 0;
        for(int j = realAxis.size() - 1; j>=0; j--){
          int col = o_i % x->shape[realAxis[j]];
          o_i /= x->shape[realAxis[j]];
          tmp_in_i += col * every_xdim_size[realAxis[j]];
        }
        f(tmp, x_data[in_i+tmp_in_i]);
      }
      y_data[i] = tmp;
    }
  }
}

void Reduce_fpga(DLTensor *x, DLTensor *y, TShape& axis, bool exclude, const int type){
  cl_int code;
  int sizeX = getSize(x);
  int sizeY = getSize(y);

  std::vector<int> realAxis(axis.size());
  std::shared_ptr<int> flag(new int[x->ndim]);
  memset(flag.get(), 0, sizeof(int)*x->ndim);
  for(uint i = 0; i < axis.size(); i++){
    int val = axis[i];
    realAxis[i] = val;
    flag.get()[val] = 1;
  }
  std::sort(realAxis.begin(), realAxis.end());
  realAxis.resize(std::unique(realAxis.begin(), realAxis.end()) - realAxis.begin());

  for(int i = 0; i < realAxis.size(); i++){
    cout << realAxis[i] << " ";
  }
  cout << endl;

  int axis_size = 1;
  for(uint i = 0; i < realAxis.size(); i++){
    axis_size *= x->shape[realAxis[i]];
  }
  std::vector<int> every_xdim_size(x->ndim, 1);
  for(int i = x->ndim-2; i >= 0; i--){
    every_xdim_size[i] = x->shape[i+1] * every_xdim_size[i+1];
  }

  int yndim = y->ndim;
  std::vector<int> yshape(y->ndim);
  for(int i = 0; i < y->ndim; i++){
    yshape[i] = y->shape[i];
  }
  for(int i = 0, j = 0; i < y->ndim; i++){
    if(y->shape[i] == 1) {
      yndim -= 1;
    }else{
      yshape[j++] = y->shape[i];
    }
  }

  int axis_ndim = realAxis.size();
  int dev_xshape[MAX_DIM], dev_yshape[MAX_DIM], dev_every_xdim_size[MAX_DIM], dev_flag[MAX_DIM], dev_axis[MAX_DIM];

  get_cuda_shape(x->shape, x->ndim, dev_xshape);
  get_cuda_shape(yshape.data(), y->ndim, dev_yshape);
  get_cuda_shape(axis.data(), axis_ndim, dev_axis);
  get_cuda_shape(every_xdim_size.data(), x->ndim, dev_every_xdim_size);
  get_cuda_shape(flag.get(), x->ndim, dev_flag);

  cl_mem bufA = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(int)*sizeX, NULL, &code);
  assert(code == CL_SUCCESS);
  clEnqueueWriteBuffer(queue, bufA, CL_TRUE, 0, sizeof(int)*sizeX, x->data, 0, nullptr, nullptr);
  cout << "size y = " << sizeY << endl;
  cl_mem bufC = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(int)*sizeY, NULL, &code);
  assert(code == CL_SUCCESS);

  cl_kernel kernel = clCreateKernel(program, "reduce", &code);
  assert(code == CL_SUCCESS);
  int index = 0;
  clSetKernelArg(kernel, index++, sizeof(cl_mem), (void*)&bufA);
  clSetKernelArg(kernel, index++, sizeof(cl_mem), (void*)&bufC);
  clSetKernelArg(kernel, index++, sizeof(int), (void*)&axis_ndim);
  clSetKernelArg(kernel, index++, sizeof(int), (void*)&x->ndim);
  clSetKernelArg(kernel, index++, sizeof(int), (void*)&y->ndim);
  clSetKernelArg(kernel, index++, sizeof(int), (void*)&sizeY);
  clSetKernelArg(kernel, index++, sizeof(int), (void*)&axis_size);
  
  clSetKernelArg(kernel, index++, sizeof(int), (void*)&type);
  clSetKernelArg(kernel, index++, sizeof(int), (void*)&dev_xshape[0]);
  clSetKernelArg(kernel, index++, sizeof(int), (void*)&dev_xshape[1]);
  clSetKernelArg(kernel, index++, sizeof(int), (void*)&dev_xshape[2]);
  clSetKernelArg(kernel, index++, sizeof(int), (void*)&dev_xshape[3]);
  clSetKernelArg(kernel, index++, sizeof(int), (void*)&dev_xshape[4]);
  clSetKernelArg(kernel, index++, sizeof(int), (void*)&dev_xshape[5]);

  clSetKernelArg(kernel, index++, sizeof(int), (void*)&dev_yshape[0]);
  clSetKernelArg(kernel, index++, sizeof(int), (void*)&dev_yshape[1]);
  clSetKernelArg(kernel, index++, sizeof(int), (void*)&dev_yshape[2]);
  clSetKernelArg(kernel, index++, sizeof(int), (void*)&dev_yshape[3]);
  clSetKernelArg(kernel, index++, sizeof(int), (void*)&dev_yshape[4]);
  clSetKernelArg(kernel, index++, sizeof(int), (void*)&dev_yshape[5]);

  clSetKernelArg(kernel, index++, sizeof(int), (void*)&dev_axis[0]);
  clSetKernelArg(kernel, index++, sizeof(int), (void*)&dev_axis[1]);
  clSetKernelArg(kernel, index++, sizeof(int), (void*)&dev_axis[2]);
  clSetKernelArg(kernel, index++, sizeof(int), (void*)&dev_axis[3]);
  clSetKernelArg(kernel, index++, sizeof(int), (void*)&dev_axis[4]);
  clSetKernelArg(kernel, index++, sizeof(int), (void*)&dev_axis[5]);

  clSetKernelArg(kernel, index++, sizeof(int), (void*)&dev_every_xdim_size[0]);
  clSetKernelArg(kernel, index++, sizeof(int), (void*)&dev_every_xdim_size[1]);
  clSetKernelArg(kernel, index++, sizeof(int), (void*)&dev_every_xdim_size[2]);
  clSetKernelArg(kernel, index++, sizeof(int), (void*)&dev_every_xdim_size[3]);
  clSetKernelArg(kernel, index++, sizeof(int), (void*)&dev_every_xdim_size[4]);
  clSetKernelArg(kernel, index++, sizeof(int), (void*)&dev_every_xdim_size[5]);

  clSetKernelArg(kernel, index++, sizeof(int), (void*)&dev_flag[0]);
  clSetKernelArg(kernel, index++, sizeof(int), (void*)&dev_flag[1]);
  clSetKernelArg(kernel, index++, sizeof(int), (void*)&dev_flag[2]);
  clSetKernelArg(kernel, index++, sizeof(int), (void*)&dev_flag[3]);
  clSetKernelArg(kernel, index++, sizeof(int), (void*)&dev_flag[4]);
  clSetKernelArg(kernel, index++, sizeof(int), (void*)&dev_flag[5]);
  clEnqueueTask(queue, kernel, 0, NULL, NULL);

  clEnqueueReadBuffer(queue, bufC, CL_TRUE, 0, sizeof(int)*sizeY, y->data, 0, nullptr, nullptr); 
}

int main(){
  init_opencl("ops.xclbin");

  DLTensor x(4), y(2), y1(2);
  TShape axis;
  bool exclude = false;
  reduce_func f = [](int& tmp, int value)->void {
   // if(tmp < value) tmp = value;
    tmp += value;
  };
  int type = 1;

  x.shape[0] = 4;
  x.shape[1] = 4;
  x.shape[2] = 4;
  x.shape[3] = 8;
  int sizeX = getSize(&x);
  x.data = new int[sizeX];
  y.shape[0] = 4;
  y.shape[1] = 4;
  y1.shape[0] = 4;
  y1.shape[1] = 4;
  int sizeY = getSize(&y);
  y.data = new int[sizeY];
  y1.data = new int[sizeY];

  for(int i = 0; i < sizeX; i++)
    x.data[i] = i % 127;

  axis.push_back(1);
  axis.push_back(3);

  Reduce(&x, &y, axis, exclude, f);
  Reduce_fpga(&x, &y1, axis, exclude, type);

  verify(y.data, y1.data, sizeY);
}
