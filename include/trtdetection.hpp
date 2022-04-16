#ifndef TRT_DETECTION_HPP
#define TRT_DETECTION_HPP

#include <iostream>
#include <fstream>
#include <dirent.h>
#include <thread>
#include <algorithm>
// #include <omp.h>

#include "detection.hpp"
#include "stdlib.h"
#include "cuda_runtime_api.h"
#include "loguru.hpp"
#include "NvInferPlugin.h"
#include "NvInfer.h"
#include "utils.hpp"

#include <sys/stat.h>

// using name space
namespace DECT
{
   using namespace nvinfer1;

   #define CHECK_C(status) \
      do\
      {\
         CHECK_F(status == 0, "CUDA failure");\
      } while(0)

   // node info definition 
   class NodeInfo
   {
      public:
         NodeInfo(){name = "None"; index=-1; data_size=0;};
         NodeInfo(const std::string n, const long i): name(n), index(i){data_size=0;};
         NodeInfo(const std::string n, const Dims d, const long i): name(n), dim(d), index(i){data_size=0;};
         std::string name;
         size_t getDataSize();
         Dims dim;
         long index;
      protected:
         size_t data_size;
   };

   size_t NodeInfo::getDataSize()
   {
      if (data_size <= 0)
      {
         auto data_size_ = 1;
         for (int i = 0; i < dim.nbDims; i++)
         {
            data_size_ *= dim.d[i];
         }
         data_size = data_size_;
         return data_size;
      }
      else
      {
         return data_size;
      }
   }

   // TRT Logger definition
   class TRTLogger: public ILogger {
      void log (Severity severity, const char * msg) noexcept override;
   };

   void TRTLogger::log(Severity severity, const char * msg) noexcept
   {
      switch (severity) 
      {
         case Severity::kINTERNAL_ERROR:
            LOG_F(FATAL, msg); break; 
         case Severity::kERROR: 
            LOG_F(ERROR, msg); break;
         case Severity::kWARNING: 
            LOG_F(WARNING, msg); break;
         case Severity::kINFO: 
            LOG_F(INFO, msg); break;
         default:
            LOG_F(INFO, msg); break;
      }
   }

   static TRTLogger TLogger;

   class TRTDetection: public Detection
   {
      public:
         NodeInfo input_node;
         std::vector<NodeInfo> output_nodes;
         TRTDetection(
            const std::vector<int> in_size,
            const std::string input_name,
            const std::vector<std::string> output_names,
            const int num_classes=80);

         ~TRTDetection(){};
         virtual void load_ckpt(const std::string ckpt) override;
         virtual void set_device(const int device) override;
      private:
         ICudaEngine * engine;
         IExecutionContext * context;
         virtual void detectImage(
            const std::string image_file, 
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

         virtual cv::Mat dataPreprocess(cv::Mat & image) override;
         void doInference(cv::Mat & input_images,
                          std::vector<float *> & output_results);

         void decodeOutputs(std::vector<Object> & objects,
                           const std::vector<float *> & output_results,
                           const float conf_thre=0.3,
                           const float mask_thre=0.5);
   };

   TRTDetection::TRTDetection(
      const std::vector<int> in_size,
      const std::string input_name,
      const std::vector<std::string> output_names,
      const int num_classes): Detection(in_size, num_classes)
   {
      engine = nullptr;
      context = nullptr;
      input_node.name = input_name;
      for (int i = 0; i < output_names.size(); i++)
      {
         output_nodes.emplace_back(output_names[i], 0);
      }
   }

   void TRTDetection::detectImage(
      const std::string image_path, 
      const float conf_thre, 
      const float mask_thre,
      const std::string save_path,
      const std::vector<std::string> class_names,
      const std::vector<std::vector<float>> color_list,
      const bool is_show,
      const bool verbose)
   {
      // const int max_thread_num = std::min(static_cast<int32_t>(std::thread::hardware_concurrency()), 8);
      // ThreadPool pool(max_thread_num);
      std::vector<std::string> image_file_set;
      check_file(image_path, image_file_set, verbose, {"png", "jpeg", "jpg"});
      CHECK_F(!image_file_set.empty(), "Can't find any matched image");
      for (auto image_file : image_file_set)
      {
         auto start = std::chrono::system_clock::now();
         check_suffix(image_file, {"png", "jpg", "jpeg"}, verbose);
         cv::Mat image = cv::imread(image_file);
         cv::Mat input_image = dataPreprocess(image);
         origin_size[0] = image.rows;
         origin_size[1] = image.cols;
         std::vector<float *> output_results;
         output_results.resize(output_nodes.size());
         for (int i = 0; i < output_nodes.size(); i++)
         {
            output_results[i] = new float [output_nodes[i].getDataSize()];
         }
         doInference(input_image, output_results);
         std::vector<Object> objects;
         decodeOutputs(objects, output_results, conf_thre, mask_thre);
         drawObjects(image, objects, origin_size, 1, 1, save_path, class_names, color_list);
         if (save_path.c_str() != nullptr) 
         {
            cv::imwrite(save_path, image);
         }
         auto end = std::chrono::system_clock::now();
         if (verbose)
            LOG_F(INFO, "Inference time is %d ms", std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count());
         for (auto output_result: output_results)
         {
            delete [] output_result;
         }
         if (is_show)
         {
            cv::imshow(image_file, image);
            cv::waitKey(0);
         }
      }
   }

   void TRTDetection::detectVideo(
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
         std::vector<float *> output_results;
         output_results.resize(output_nodes.size());
         for (int i = 0; i < output_nodes.size(); i++)
         {
            output_results[i] = new float [output_nodes[i].getDataSize()];
         }
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
         for (auto output_result: output_results)
         {
            delete [] output_result;
         }
         if (is_show)
         {
            cv::imshow(video_file, frame);
            cv::waitKey(10);
         }
      }
   }

   cv::Mat TRTDetection::dataPreprocess(cv::Mat & image)
   {
      cv::Mat resize_image = Detection::image_resize(image, input_size);
      // resize_image.convertTo(resize_image, CV_32FC3);
      // resize_image.convertTo(resize_image, CV_32FC3, 1.f / 255.f);
      // cv::subtract(resize_image, cv::Scalar(0.485f, 0.456f, 0.406f), resize_image, cv::noArray(), -1);
      // cv::divide(resize_image, cv::Scalar(0.229f, 0.224f, 0.225f), resize_image, 1, -1);
      // return resize_image;

      cv::Mat blob = cv::dnn::blobFromImage(resize_image);
      return blob;
   }

   void TRTDetection::doInference(cv::Mat & input_image, 
                                 std::vector<float *> & output_results)
   {
      // const ICudaEngine & engine = context.getEngine();
      const int buffers_Nb = static_cast<int32_t> (engine->getNbBindings());
      CHECK_F(buffers_Nb == static_cast<int32_t> (1 + output_results.size()), "Input + Output dims");
      void * buffers[buffers_Nb];
      // std::unique_ptr<float> input(new float [input_node.getDataSize()]);
      float * input = new float[input_node.getDataSize()];
      auto channels = input_image.channels();
      auto image_h = input_image.rows;
      auto image_w = input_image.cols;
      // set max threads
      // auto max_threads = static_cast<int>(std::thread::hardware_concurrency());
      // omp_set_num_threads(std::min(max_threads, 8));
      // #pragma omp parallel for 
      // for (size_t c = 0; c < channels; c++)
      // {
      //    for (size_t h = 0; h < image_h; h++)
      //    {
      //       for (size_t w = 0; w < image_w; w++)
      //       {
      //          input[c * image_w * image_h + h * image_w + w] = static_cast<float>(input_image.at<cv::Vec3f>(h, w)[c]);
      //       }
      //    }
      // }
      memcpy(input, input_image.data, sizeof(float) * input_node.getDataSize());
      // Malloc Input Data
      CHECK_C(cudaMalloc(&buffers[input_node.index], input_node.getDataSize() * sizeof(float)));
      // Malloc Output Data 
      for (auto node: output_nodes)
      {
         CHECK_C(cudaMalloc(&buffers[node.index], node.getDataSize() * sizeof(float)));
      }
      // Create stream
      cudaStream_t stream;
      CHECK_C(cudaStreamCreate(&stream));
      // Copy Input Memory
      CHECK_C(cudaMemcpyAsync(buffers[input_node.index], input, input_node.getDataSize() * sizeof(float), cudaMemcpyHostToDevice, stream));
      // Inference 
      context->enqueue(1, buffers, stream, nullptr);
      // Copy Output Memory
      for (int i = 0; i < output_results.size(); i++)
      {
         CHECK_C(cudaMemcpyAsync(output_results[i], buffers[output_nodes[i].index], output_nodes[i].getDataSize() * sizeof(float), cudaMemcpyDeviceToHost, stream));
      }
      // Synchoronize
      cudaStreamSynchronize(stream);
      cudaStreamDestroy(stream);
      // Free Momery
      CHECK_C(cudaFree(buffers[input_node.index]));
      for (int i = 0; i < output_nodes.size(); i++)
      {
         NodeInfo & node = output_nodes[i];
         CHECK_C(cudaFree(buffers[node.index]));
      }

      // delete [] input;
   }

   void TRTDetection::decodeOutputs(std::vector<Object> & objects,
                                    const std::vector<float *> & output_results,
                                    const float conf_thre,
                                    const float mask_thre)
   {
      NodeInfo bbox_node = output_nodes[0];
      NodeInfo mask_node = output_nodes[1];
      CHECK_F(bbox_node.dim.d[0] == mask_node.dim.d[0], "mask and bbox out put");
      float * bbox_out = output_results[0];
      float * mask_out = output_results[1];
      double scale_ratio = std::min(input_size[0] / (origin_size[0] * 1.0), input_size[1] / (origin_size[1] * 1.0));
      for (int index=0; index < bbox_node.dim.d[0]; index++)
      {
         const auto mask_out_h = mask_node.dim.d[2];
         const auto mask_out_w = mask_node.dim.d[3];
         cv::Mat mask = cv::Mat::zeros(mask_out_h, mask_out_w, CV_32FC1);
         CHECK_F(mask.isContinuous(), "Only support continuous mask");
         memcpy(mask.data, mask_out + index * mask_out_h * mask_out_w, sizeof(float) * mask_out_h * mask_out_w);
         auto score = bbox_out[index * 6 + 4];
         if (score > conf_thre)
         {
            auto x1 = static_cast<int32_t>(bbox_out[index * 6 + 0] / scale_ratio);
            auto y1 = static_cast<int32_t>(bbox_out[index * 6 + 1] / scale_ratio);
            auto x2 = static_cast<int32_t>(bbox_out[index * 6 + 2] / scale_ratio);
            auto y2 = static_cast<int32_t>(bbox_out[index * 6 + 3] / scale_ratio);
            auto w = x2 - x1; auto h = y2 - y1;
            auto class_id = static_cast<int32_t>(bbox_out[index * 6 + 5]);
            cv::Mat bit_mask = cv::Mat::zeros(mask_out_h, mask_out_w, CV_8UC1);
            bit_mask = mask > mask_thre;
            cv::Mat in_bit_mask = cv::Mat::zeros((mask_out_h / scale_ratio), (mask_out_w / scale_ratio), CV_8UC1);
            cv::resize(bit_mask, in_bit_mask, in_bit_mask.size());
            objects.emplace_back(
               (cv::Rect){x1, y1, w, h},
               in_bit_mask,
               class_id,
               score
            );
         }
      }
   }

   void TRTDetection::set_device(const int device){
      CHECK_F(device != -1, "Can't support inference in cpu");
      LOG_F(INFO, "Model will be inference in cuda: %d", device);
      cudaSetDevice(device);
   }

   void TRTDetection::load_ckpt(const std::string ckpt){
      char * trtModelStream(nullptr);
      size_t size{0};
      LOG_F(INFO, "Engine file is %s", ckpt.c_str());
      std::ifstream file(ckpt, std::ios::binary);
      CHECK_F(file.good(), "File is not opened correctly");
      file.seekg(0, file.end);
      size = file.tellg();
      file.seekg(0, file.beg);
      trtModelStream = new char[size];
      CHECK_F(trtModelStream != nullptr, "TRTModelStream data buffer is created fail");
      file.read(trtModelStream, size);
      file.close();
      CHECK_F((bool)initLibNvInferPlugins(&TLogger, ""), "Init NvInfer Plugin fail");
      IRuntime * runtime = createInferRuntime(TLogger);
      CHECK_F(runtime != nullptr, "runtime");
      engine = runtime->deserializeCudaEngine(trtModelStream, size);
      CHECK_F(engine != nullptr, "engine");
      context = engine->createExecutionContext();
      CHECK_F(context != nullptr, "context");
      input_node.index = engine->getBindingIndex(input_node.name.c_str());
      auto out_dims = engine->getBindingDimensions(input_node.index);
      input_node.dim = out_dims;
      // for (int i = 0; i < output_nodes.size(); i++)
      // {
      //    NodeInfo & node = output_nodes[i];
      //    auto out_dims = engine->getBindingDimensions(node.index);
      //    node.dim = out_dims;
      // }
      for (std::vector<NodeInfo>::iterator node_ptr=output_nodes.begin(); node_ptr != output_nodes.end(); node_ptr++)
      {
         node_ptr->index = engine->getBindingIndex((node_ptr->name).c_str());
         node_ptr->dim = engine->getBindingDimensions(node_ptr->index);
      }
      delete [] trtModelStream;
   }

}

#endif
