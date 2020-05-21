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
#include <functional>
#include <memory>

#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/types_c.h>

using namespace cv;
using namespace std;

#define CHECK_STATUS(x, msg) \
  if (x != SUCCEED) { \
    cout << "STATUS ERROR: " << x << " " << msg << "\n"; \
    return -1; \
  }

typedef std::function<void(Mat& image, vector<char>& input)> input_func;
typedef std::function<void(vector<char>&output, Mat& image)> out_func;

template<typename T>
T clip(T x, const T min, const T max){
  if(x < min) x = min;
  if(x > max) x = max;
  return x;
} 

int argmax(vector<char>& output){
  int max_index = 0;
  int max = 1 << 31;
  for (auto i = 0; i < output.size(); i++) {
    int32_t value = output[i];
    //std::cout << value << " ";
    if(value > max) {
      max = value;
      max_index = i;
    }
  }
  //std::cout << std::endl;
  return max_index;
}

int infer(string& model_root, Mat& image, input_func inf, out_func& outf){
  //string model_root = "/media/nvme/data/cvm_mnist/";
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

  inf(image, input);

  status = CVMAPIInference(net, input.data(), input.size(), output.data());
  CHECK_STATUS(status, "inference failed");
  status = CVMAPIFreeModel(net);
  CHECK_STATUS(status, "free model failed");

  //cout << endl;
  //cout << "infer result: " << argmax(output) << endl;
  outf(output, image);
  return 0;
}

void mnist_transform(vector<char>& input, Mat& image){
  //imshow("test", image);
  cout << "channels=" << image.channels() << ", rows=" << image.rows << ", cols=" << image.cols << endl; 
  Mat imageGray;
  cvtColor(image, imageGray, CV_RGB2GRAY);
  cout << "channels=" << imageGray.channels() << ", rows=" << imageGray.rows << ", cols=" << imageGray.cols << endl; 
  //imwrite("gray.png", imageGray);
  //waitKey(0);
  Mat resizeImg;
  //input 1 28 28
  resize(imageGray, resizeImg, Size(28, 28));
  cout << "channels=" << resizeImg.channels() << ", rows=" << resizeImg.rows << ", cols=" << resizeImg.cols << endl; 
  //imwrite("resizeImg.png", resizeImg);
  for(int i = 0; i < resizeImg.rows; i++){
    for(int j = 0; j < resizeImg.cols; j++){
      float tmp = resizeImg.at<uchar>(i,j);
      tmp = (255 - tmp) * 127 / 255;
      input[i*resizeImg.cols + j] = clip(tmp, -127.0f, 127.0f); 
    }
  }
}

void mnist_infer(string image_path){
  Mat image = imread(image_path);//imread(argv[1], 1);
  string model_root = "/media/nvme/data/cvm_mnist/";
  
  input_func inf = [](Mat& image, vector<char>& input){
    mnist_transform(input, image);
  };

  out_func outf = [](vector<char>& output, Mat& image) {
    std::cout << "infer result : " << argmax(output) << std::endl; 
  };
  infer(model_root, image, inf, outf);
}

void ssd_transform(vector<char>& input, Mat& image){
  const int height = image.rows;
  const int width = image.cols;
  const int channel_size = height * width;
  const float ssd_mean[3] = {0.485, 0.456, 0.406};
  const float ssd_std[3] = {0.229, 0.224, 0.225};
  for(int h = 0; h < image.rows; h++){
    for(int w = 0; w < image.cols; w++){
      unsigned char pixel[3];
      pixel[0] = image.at<Vec3b>(h,w)[0];
      pixel[1] = image.at<Vec3b>(h,w)[1];
      pixel[2] = image.at<Vec3b>(h,w)[2];
      for(int i = 0; i < 3; i++){
        float x = pixel[i];
        x = x / 255.0f;
        x = (x - ssd_mean[i]) / ssd_std[i] * 48.1060606060606;
        pixel[i] = clip(x, -127.0f, 127.0f);
      }
      input[h*width + w] = pixel[0];
      input[channel_size + h*width + w] = pixel[1];
      input[2*channel_size + h * width + w] = pixel[2];
    } 
  }
}
void get_labels(std::vector<std::string>& label_list, std::string& label_path){
  fstream f(label_path, fstream::in);
  if(!f){
    std::cout << "open file error: " << label_path << std::endl;
    return;
  }
  string line;
  while(getline(f, line)){
    label_list.push_back(line);
  }
  f.close();
}
void ssd_infer(string image_path){
  Mat image = imread(image_path); 
  //const int height = 512, width = 512;
  //vector<char> input(height * width * 3);
  string model_root = "/media/nvme/data/ssd_512_mobilenet1.0_coco_tfm/";

  //ssd_transform(input, image);
  input_func inf = [](Mat& image, vector<char>& input){
    Mat resizeImg;
    const int height = 512, width = 512;
    resize(image, resizeImg, Size(width, height));
    ssd_transform(input, image);
  };

  out_func outf = [](vector<char>& output, Mat& image){
    const float ext[3] = {1.0f, 536870910.0f, 853144.6988601037f};
    const float threshold = 0.5;
    int32_t *int32_output = (int32_t*)output.data();
    int n = output.size()/4;
    Mat resizeImg;
    resize(image, resizeImg, Size(512, 512));

    string label_coo = "/media/nvme/data/ssd_512_mobilenet1.0_coco_tfm/label_coco.txt";
    std::vector<string> label_list;
    get_labels(label_list, label_coo);

    for(int i = 0; i < n; i += 6){
      if(int32_output[i] == -1) break;

      float item[6];
      for(int j = 0; j < 6; j++){
        item[j] = int32_output[i+j]; 
      }

      item[0] = item[0] / ext[0];
      item[1] = item[1] / ext[1];
      for(int j = 2; j < 6; j++){
        item[j] = item[j] / ext[2]; 
      }
      if(item[1] < threshold) break;

      for(int j = 2; j < 6; j++){
        item[j] = clip(item[j], 0.0f, 512.0f);
      } 

      Rect rect((int)item[2], (int)item[3], (int)item[4], (int)item[5]);
      rectangle(resizeImg, rect, Scalar(255,0,0), 1, LINE_8, 0);
      Point point((int)item[2]+2, (int)item[3]+40);
      int id = (int)item[0];
      if(label_list.size() > id){
        string label = label_list[(int)item[0]];
        std::cout << label << std::endl;
        putText(resizeImg, label, point, FONT_HERSHEY_SIMPLEX, 2, Scalar(0,0,255), 4, 8);
      }
    }
    imwrite("ssd.png", resizeImg);
  };
  infer(model_root, image, inf, outf);
}

void mobilenet_infer(string image_path){
  Mat image = imread(image_path); 
  //const int height = 512, width = 512;
  //vector<char> input(height * width * 3);
  string model_root = "/media/nvme/data/mobilenet1.0/";

  //ssd_transform(input, image);
  input_func inf = [](Mat& image, vector<char>& input){
    Mat resizeImg;
    const int height = 224, width = 224;
    resize(image, resizeImg, Size(width, height));
    ssd_transform(input, resizeImg);
  };

  out_func outf = [](vector<char>& output, Mat& image){
    int index = argmax(output);
    string label_coo = "/media/nvme/data/label_imagenet.txt";
    std::vector<string> label_list;
    get_labels(label_list, label_coo);
    if(index < label_list.size()){
      std::cout << label_list[index] << std::endl; 
    }
  };
  infer(model_root, image, inf, outf);
}

void resnet50_infer(string image_path){
  Mat image = imread(image_path); 
  //const int height = 512, width = 512;
  //vector<char> input(height * width * 3);
  string model_root = "/media/nvme/data/resnet50_v2/";

  //ssd_transform(input, image);
  input_func inf = [](Mat& image, vector<char>& input){
    Mat resizeImg;
    const int height = 224, width = 224;
    resize(image, resizeImg, Size(width, height));
    ssd_transform(input, resizeImg);
  };

  out_func outf = [](vector<char>& output, Mat& image){
    int index = argmax(output);
    string label_coo = "/media/nvme/data/label_imagenet.txt";
    std::vector<string> label_list;
    get_labels(label_list, label_coo);
    if(index < label_list.size()){
      std::cout << label_list[index] << std::endl; 
    }
  };
  infer(model_root, image, inf, outf);
}

int main(int argc, char**argv){
  if(argc < 2) return 0;

  mnist_infer(argv[1]);

  //ssd_infer(argv[1]);
  
  //mobilenet_infer(argv[1]);

  //resnet50_infer(argv[1]);
  return 0;
}
