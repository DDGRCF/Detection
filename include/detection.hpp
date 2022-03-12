#ifndef DETECTION_HPP
#define DETECTION_HPP

#include <string>
#include <vector>
#include <opencv2/opencv.hpp>
#include "loguru.hpp"

class Object
{
    public:
    Object();
    Object(cv::Rect r, int l, float s): rect(r), label(l), score(s){}
    Object(cv::Rect r, cv::Mat m, int l, float s): rect(r), mask(m), label(l), score(s){}
    cv::Rect rect;
    cv::Mat mask;
    int label;
    float score;
};


class Detection
{
    public:
        std::vector<int> input_size;
        std::vector<int> origin_size;
        float conf_thre;
        float mask_thre;
        int num_classes;
        Detection(const std::vector<int> in_size = {640, 640});
        Detection(std::vector<int> input_size, const int nc=80, const float ct=0.3, const float mt=0.5);
        virtual ~Detection();
        virtual void load_ckpt(const std::string & ckpt) = 0;
        virtual void set_device(const int device) = 0;
        virtual void detectImage(const std::string & image_file) = 0;
        virtual void detectVideo(const std::string & image_file) = 0;
        virtual void drawObjects(
            cv::Mat & image, 
            const std::vector<Object> & objects, 
            const std::vector<int> & origin_size, 
            const int obj_thickness, const int thickness,
            const std::string save_path=nullptr);
        static cv::Mat image_resize(cv::Mat & image, const std::vector<int> resize_size);
};

Detection::~Detection()
{
    LOG_F(INFO, "Detection Destoried");
}

Detection::Detection(const std::vector<int> in_size)
{
    num_classes = 80;
    conf_thre = 0.3;
    mask_thre = 0.5;
    input_size.resize(2);
    input_size.assign(in_size.begin(), in_size.end());
    origin_size.resize(2);
}

Detection::Detection(const std::vector<int> in_size, const int nc, const float ct, const float mt)
{
    num_classes = nc;
    conf_thre = ct;
    mask_thre = mt;
    input_size.resize(2);
    input_size.assign(in_size.begin(), in_size.end());
    origin_size.resize(2);
}

cv::Mat Detection::image_resize(cv::Mat & image, const std::vector<int> resize_size)
{
    const int input_h = resize_size[0];
    const int input_w = resize_size[1];
    double r = std::min(input_h / (image.rows * 1.0), input_w / (image.cols * 1.0));
    int unpad_w = r * image.cols;
    int unpad_h = r * image.rows;
    cv::Mat re(unpad_h, unpad_w, CV_8UC3);
    cv::resize(image, re, re.size());
    cv::Mat out(input_h, input_w, CV_8UC3);
    re.copyTo(out(cv::Rect(0, 0, re.cols, re.rows)));
    return out;
}

#endif