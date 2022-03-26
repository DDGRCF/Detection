#ifndef DETECTION_HPP
#define DETECTION_HPP

#include <cstdlib>
#include <string>
#include <vector>
#include <opencv2/opencv.hpp>
#include "loguru.hpp"
#include "utils.hpp"

namespace DECT
{
    class Object
    {
        public:
        Object();
        Object(cv::Rect r, int l): rect(r), label(l){}
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
            int num_classes;
            Detection(const std::vector<int> in_size = {640, 640});
            Detection(std::vector<int> input_size, const int nc=80);
            virtual ~Detection();
            virtual void load_ckpt(const std::string ckpt) = 0;
            virtual void set_device(const int device) = 0;
            virtual void detect(
                const std::string file_path,
                const float conf_thre=0.3,
                const float mask_thre=0.5,
                const std::string save_path=nullptr,
                const std::vector<std::string> class_names={},
                const std::vector<std::vector<float>> color_list={},
                const bool is_show=false,
                const bool verbose=false,
                const std::string type="image",
                const int device_id=-1,
                const bool use_local_device=false);
        protected:
            virtual void detectImage(
                const std::string image_path, 
                const float conf_thre=0.3,
                const float mask_thre=0.5,
                const std::string save_path=nullptr,
                const std::vector<std::string> class_names={},
                const std::vector<std::vector<float>> color_list={},
                const bool is_show=false,
                const bool verbose=false) = 0;
            virtual void detectVideo(
                const std::string video_path,
                const float conf_thre=0.3,
                const float mask_thre=0.5,
                const std::string save_path=nullptr,
                const std::vector<std::string> class_names={},
                const std::vector<std::vector<float>> color_list={},
                const bool is_show=false,
                const bool verbose=false,
                const int device_id=-1,
                const bool use_local_device=false) = 0;
            
            virtual cv::Mat dataPreprocess(cv::Mat & image) = 0;
            template <typename T>
            void doInference(cv::Mat & input_images,
                            std::vector<T> & output_results){};
            template <typename T>
            void decodeOutputs(std::vector<Object> & objects,
                            const std::vector<T> & output_results,
                            const float conf_thre=0.3,
                            const float mask_thre=0.5){};
            virtual void drawObjects(
                cv::Mat & image, 
                const std::vector<Object> & objects, 
                const std::vector<int> & origin_size, 
                const int obj_thickness=1, const int thickness=1,
                const std::string & save_path=nullptr,
                const std::vector<std::string> & class_names = {},
                const std::vector<std::vector<float>> & color_list = {});
            static cv::Mat image_resize(cv::Mat & image, const std::vector<int> resize_size);

    };

    Detection::~Detection()
    {
        LOG_F(INFO, "Detection Deleted");
    }

    Detection::Detection(const std::vector<int> in_size)
    {
        num_classes = 80;
        input_size.resize(2);
        input_size.assign(in_size.begin(), in_size.end());
        origin_size.resize(2);
    }

    Detection::Detection(const std::vector<int> in_size, const int nc)
    {
        num_classes = nc;
        input_size.resize(2);
        input_size.assign(in_size.begin(), in_size.end());
        origin_size.resize(2);
    }
    
    void Detection::detect(
        const std::string file_path,
        const float conf_thre,
        const float mask_thre,
        const std::string save_path,
        const std::vector<std::string> class_names,
        const std::vector<std::vector<float>> color_list,
        const bool is_show,
        const bool verbose,
        const std::string type,
        const int device_id,
        const bool use_local_device)
    {
        if (type == "image")
        {
            LOG_F(INFO, "Begin detecting the image...");
            detectImage(
                file_path, 
                conf_thre, 
                mask_thre, 
                save_path, 
                class_names,
                color_list, 
                is_show,
                verbose);

        }
        else if (type == "video")
        {
            LOG_F(INFO, "Begin detecting the video...");
            detectVideo(
                file_path,
                conf_thre,
                mask_thre,
                save_path,
                class_names,
                color_list,
                is_show,
                verbose,
                device_id,
                use_local_device
            );
        }
        else
        {
            LOG_F(ERROR, "Cant find the matched image type of");
            abort();
        }

    }
 
    void Detection::drawObjects(
        cv::Mat & image, 
        const std::vector<Object> & objects, 
        const std::vector<int> & origin_size, 
        const int obj_thickness, const int txt_thickness,
        const std::string & save_path,
        const std::vector<std::string> & class_names,
        const std::vector<std::vector<float>> & color_list)
    {
        for (int index = 0; index < objects.size(); index++)
        {
            const Object obj = objects[index];
            cv::Rect box = obj.rect;
            const auto class_id = obj.label;
            const auto score = obj.score;
            cv::Mat mask = obj.mask;
            std::string text;
            cv::Scalar txt_color;
            cv::Scalar color = cv::Scalar(color_list[class_id][0], color_list[class_id][1], color_list[class_id][2]);
            float c_mean = cv::mean(color)[0];
            txt_color = (c_mean > 0.) ? cv::Scalar(0., 0., 0.) : cv::Scalar(255, 255, 255);
            cv::rectangle(image, box, color * 255, obj_thickness);
            text = class_names[class_id] + " | " + std::to_string(score);
            int baseline = 0;
            cv::Size label_size =cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.4, 1, &baseline);
            cv::Scalar txt_bk_color = color * 0.7 * 255;
            auto x = box.x; 
            auto y = box.y + 1;
            y = (y > image.rows) ? image.rows : y;
            cv::rectangle(image, cv::Rect(cv::Point(x, y), cv::Size(label_size.width, label_size.height + baseline)), txt_bk_color, -1);
            cv::putText(image, text, cv::Point(x, y+label_size.height), cv::FONT_HERSHEY_SIMPLEX, 0.4, txt_color, txt_thickness);
            cv::Mat color_map(origin_size[0], origin_size[1], CV_8UC3, color * 255);
            cv::Mat target_mask = cv::Mat::zeros(color_map.size(), CV_8UC3);
            color_map.copyTo(target_mask, mask(cv::Rect(0, 0, target_mask.cols, target_mask.rows)));
            cv::addWeighted(image, 1., target_mask, 0.5, 0., image);
        }
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
}
#endif