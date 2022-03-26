#ifndef TSPDETECTION_HPP
#define TSPDETECTION_HPP
#include <torch/torch.h>
#include <torch/script.h>
#include "cuda_runtime_api.h"
#include "detection.hpp"


namespace DECT
{
    class TSPDetection: public Detection
    {
        public:
            TSPDetection(
                const std::vector<int> in_size,
                const int num_classes=80): Detection(in_size, num_classes), device(torch::kCPU){}
            ~TSPDetection(){}
            virtual void set_device(const int device_id) override;
            virtual void load_ckpt(const std::string ckpt) override; 
        private:
            torch::Device device;
            torch::jit::script::Module model;
            virtual cv::Mat dataPreprocess(cv::Mat & image);
            void doInference(cv::Mat & input_image, std::vector<torch::Tensor> & output_results);
            void decodeOutputs(std::vector<Object> & objects,
                            const std::vector<torch::Tensor> & output_results,
                            const float conf_thre=0.3,
                            const float mask_thre=0.5);
            virtual void detectImage(
                const std::string image_path, 
                const float conf_thre=0.3,
                const float mask_thre=0.5,
                const std::string save_path=nullptr,
                const std::vector<std::string> class_names={},
                const std::vector<std::vector<float>> color_list={},
                const bool is_show=false,
                const bool verbose=false) override;
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
                const bool use_local_device=false) override;

    };

    void TSPDetection::load_ckpt(const std::string ckpt)
    {
        model = torch::jit::load(ckpt, device);
    }

    void TSPDetection::set_device(const int device_id) 
    {
        if (device_id != -1)
        {
            cudaSetDevice(device_id); 
            torch::Device device_(torch::kCUDA, 0);
            device = device_;
            LOG_F(INFO, "Model will be inference in cuda: %d", device_id);
        }
        else
        {
            LOG_F(INFO, "Model will be inference in cpu");
        }
    }
    cv::Mat TSPDetection::dataPreprocess(cv::Mat & image)
    {
        cv::Mat resize_image = Detection::image_resize(image, input_size);
        return resize_image;
    }

    void TSPDetection::detectImage(
        const std::string image_path, 
        const float conf_thre,
        const float mask_thre,
        const std::string save_path,
        const std::vector<std::string> class_names,
        const std::vector<std::vector<float>> color_list,
        const bool is_show,
        const bool verbose)
    {
      std::vector<std::string> image_file_set;
      check_file(image_path, image_file_set, verbose, {"png", "jpg"});
      CHECK_F(!image_file_set.empty(), "Can't find any image");
      for (auto image_file : image_file_set)
      {
        check_suffix(image_file, {"png", "jpg"}, verbose);
        cv::Mat image = cv::imread(image_file);
        cv::Mat input_image = dataPreprocess(image);
        origin_size[0] = image.rows;
        origin_size[1] = image.cols;

        std::vector<torch::Tensor> output_results;
        auto start = std::chrono::system_clock::now();
        doInference(input_image, output_results);
        auto end = std::chrono::system_clock::now();
        if (verbose)
        LOG_F(INFO, "Inference time is %d ms", std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count());
        std::vector<Object> objects;
        decodeOutputs(objects, output_results, conf_thre, mask_thre);
        drawObjects(image, objects, origin_size, 1, 1, save_path, class_names, color_list);
        if (save_path.c_str() != nullptr) 
        {
            cv::imwrite(save_path, image);
        }
        if (is_show)
        {
            cv::imshow(image_file, image);
            cv::waitKey(0);
        }
      }
    }

    void TSPDetection::detectVideo(
        const std::string video_file,
        const float conf_thre,
        const float mask_thre,
        const std::string save_path,
        const std::vector<std::string> class_names,
        const std::vector<std::vector<float>> color_list,
        const bool is_show,
        const bool verbose,
        const int device_id,
        const bool use_local_device)
    {
      if (!use_local_device)
      {
         check_suffix(video_file, {"mp4", "avi"}, verbose);
      }
      cv::VideoCapture capture;
      if (use_local_device)
      {
         if (verbose)
         {
            LOG_F(INFO, "Begin Detect Camera: %d", device_id);
         }
         capture.open(device_id);
      }
      else
      {
         if (verbose)
         {
            LOG_F(INFO, "Begin Detect Video: %s", video_file.c_str());
         }
         capture.open(video_file.c_str());
      }
      cv::Size size = cv::Size(capture.get(cv::CAP_PROP_FRAME_WIDTH), capture.get(cv::CAP_PROP_FRAME_HEIGHT));
      cv::VideoWriter writer;
      if (save_path.c_str() != nullptr)
      {
         writer.open(save_path, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), 10, size, true);
      }
      cv::Mat frame;
      origin_size[0] = size.height; origin_size[1] = size.width;
      while (capture.read(frame))
      {
         auto start = std::chrono::system_clock::now();
         cv::Mat input_frame = dataPreprocess(frame); 
         std::vector<torch::Tensor> output_results;
         doInference(input_frame, output_results);
         std::vector<Object> objects;
         decodeOutputs(objects, output_results, conf_thre, mask_thre);
         drawObjects(frame, objects, origin_size, 1, 1, save_path, class_names, color_list);
         auto end = std::chrono::system_clock::now();
         auto inference_time = std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count();
         std::string FPS_text = "FPS: " + std::to_string(inference_time * 0.025);
         cv::Size label_size = cv::getTextSize(FPS_text.c_str(), cv::FONT_HERSHEY_SIMPLEX, 0.4, 1, 0);
         cv::putText(frame, FPS_text.c_str(), (cv::Point2f){10.f, 10.f + (float)label_size.height}, cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(255, 0, 0), 1);
         if (save_path.c_str() != nullptr)
         {
            writer.write(frame);
         }
         if (is_show)
         {
            cv::imshow(video_file, frame);
            cv::waitKey(10);
         }
      }
    }

    void TSPDetection::doInference(cv::Mat & input_image, std::vector<torch::Tensor> & output_results)
    {
        output_results.resize(2);
        torch::Tensor tensor_image = torch::from_blob(input_image.data, {1, input_image.rows, input_image.cols, 3}, torch::kByte);
        tensor_image = tensor_image.to(device);
        tensor_image = tensor_image.permute({0, 3, 1, 2});
        tensor_image = tensor_image.toType(torch::kFloat);
        auto outputs = model.forward({tensor_image}).toTuple();
        torch::Tensor bbox_out = outputs->elements()[0].toTensor();
        torch::Tensor mask_out = outputs->elements()[1].toTensor();
        output_results[0] = bbox_out;
        output_results[1] = mask_out;
    }

   void TSPDetection::decodeOutputs(std::vector<Object> & objects,
                                    const std::vector<torch::Tensor> & output_results,
                                    const float conf_thre,
                                    const float mask_thre)
    {
      torch::Tensor bbox_out = output_results[0];
      torch::Tensor mask_out = output_results[1];
      CHECK_F(bbox_out.size(0) == mask_out.size(0), "Shape Error");
      double scale_ratio = std::min(input_size[0] / (origin_size[0] * 1.0), input_size[1] / (origin_size[1] * 1.0));
      for (int index=0; index < bbox_out.size(0); index++)
      {
            const auto mask_out_h = mask_out.size(2);
            const auto mask_out_w = mask_out.size(3);
            cv::Mat mask = cv::Mat::zeros(mask_out_h, mask_out_w, CV_32FC1);
            CHECK_F(mask.isContinuous(), "Only support continuous mask");
            memcpy(mask.data, (mask_out.select(0, index)).data_ptr<float>(), sizeof(float) * mask_out_h * mask_out_w);
            auto score = bbox_out[index][4].item().toFloat();
            if (score > conf_thre)
            {
                float x1 = bbox_out[index][0].item().toFloat() / scale_ratio;
                float y1 = bbox_out[index][1].item().toFloat() / scale_ratio;
                float x2 = bbox_out[index][2].item().toFloat() / scale_ratio;
                float y2 = bbox_out[index][3].item().toFloat() / scale_ratio;
                float w = x2 - x1; float h = y2 - y1;
                cv::Mat bit_mask = cv::Mat::zeros(mask_out_h, mask_out_w, CV_8UC1);
                auto class_id = static_cast<int32_t>(bbox_out[index][5].item().toFloat());
                bit_mask = mask > mask_thre;
                cv::Mat in_bit_mask = cv::Mat::zeros((mask_out_h / scale_ratio), (mask_out_w / scale_ratio), CV_8UC1);
                cv::resize(bit_mask, in_bit_mask, in_bit_mask.size());
                objects.emplace_back(
                    (cv::Rect){static_cast<int>(x1), static_cast<int>(y1), static_cast<int>(w), static_cast<int>(h)},
                    in_bit_mask,
                    class_id,
                    score
                );
            }
      }
    }
}
#endif