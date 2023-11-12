#define API_EXPORTS

#ifdef API_EXPORTS
#define API __declspec(dllexport)
#else
#define API __declspec(dllimport)
#endif

#include <opencv2/opencv.hpp>

#include "cpm.hpp"
#include "infer.hpp"
#include "yolo.hpp"

using namespace std;

static const char *class_labels[] = {"FK", "HD", "NG"};

shared_ptr<yolo::Infer> engine;

yolo::Image cvimage(const cv::Mat &image) { return yolo::Image(image.data, image.cols, image.rows); }

extern "C" API void load_engine(const char* engine_path, int type){
  yolo::Type yolo_type;
  switch(type) {
  case 5: yolo_type = yolo::Type::V5;break;
  case 8: yolo_type = yolo::Type::V8;break;
  default: yolo_type = yolo::Type::V5;break;
  }
  engine = yolo::load(engine_path, yolo_type);
}

extern "C" API int* infer(int* img, int rows, int cols, int cnd, int* size) {
  cv::Mat image(rows, cols, CV_8UC3, img);
  auto boxs = engine->forward(cvimage(image));
  *size = boxs.size();
  int* data = new int[*size];
  int i = 0;
  for (auto &box : boxs) {
    data[5*i] = box.left;
    data[5*i+1] = box.top;
    data[5*i+2] = box.right - box.left;
    data[5*i+3] = box.bottom - box.top;
    data[5*i+4] = box.class_label;
    i++;
  }

  return data;
}
