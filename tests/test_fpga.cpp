#include <cvm/c_api.h>
#include <cvm/model.h>
#include <iostream>
#include <thread>
#include <omp.h>
#include <cvm/runtime/registry.h>
#include <cvm/op.h>
#include "npy.hpp"

#include <iostream>
#include <string>
#include <vector>

#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/types_c.h>

using namespace cv;
using namespace std;

#define CHECK_STATUS(x, msg) \
  if (x != SUCCEED) { \
    cout << "STATUS ERROR: " << x << " " << msg << "\n"; \
    return -1; \
  }

int main(int argc, char**argv){
  if(argc < 2)
    return 0;
  
  Mat image = imread(argv[1], 1);
  //imshow("test", image);
  cout << "channels=" << image.channels() << ", rows=" << image.rows << ", cols=" << image.cols << endl; 

  Mat imageGray;
  cvtColor(image, imageGray, CV_RGB2GRAY);
  cout << "channels=" << imageGray.channels() << ", rows=" << imageGray.rows << ", cols=" << imageGray.cols << endl; 
//  imwrite("gray.png", imageGray);
 // waitKey(0);
  Mat resizeImg;
  //input 1 28 28
  resize(imageGray, resizeImg, Size(28, 28));
  cout << "channels=" << resizeImg.channels() << ", rows=" << resizeImg.rows << ", cols=" << resizeImg.cols << endl; 
 
  string model_root = "/media/nvme/data/dcnet_mnist_v1/data";
  int device_type = 3;
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

  // API only accepts byte array
  vector<char> input, output;
  unsigned long long input_size, output_size;
  CVMAPIGetInputLength(net, &input_size);
  CVMAPIGetOutputLength(net, &output_size);
  input.resize(input_size, 0); // 1 * 1 * 28 * 28);
  output.resize(output_size, 0); //1 * 10);

  for(int i = 0; i < resizeImg.rows; i++){
    for(int j = 0; j < resizeImg.cols; j++){
      input[i*resizeImg.cols + j] = resizeImg.at<uchar>(i,j) / 2; 
    }
  }

  status = CVMAPIInference(net, input.data(), input.size(), output.data());
  CHECK_STATUS(status, "inference failed");
  status = CVMAPIFreeModel(net);
  CHECK_STATUS(status, "free model failed");

  int max_index = 0;
  int max = 1 << 31;
  for (auto i = 0; i < output.size(); i++) {
    int32_t value = output[i];
    std::cout << value << " ";
    if(value > max) {
      max = value;
      max_index = i;
    }
  }
  cout << endl;
  cout << "infer result: " << max_index + 1 << endl;
  return 0;
}
