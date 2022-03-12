#ifndef TRT_DETECTION_HPP
#define TRT_DETECTION_HPP

#include <iostream>
#include <fstream>
#include <dirent.h>
#include <opencv2/opencv.hpp>
#include <thread>
#include "NvInferPlugin.h"
#include "NvInfer.h"
#include "detection.hpp"
#include "stdlib.h"
#include "cuda_runtime_api.h"
#include "loguru.hpp"


// using name space
using namespace nvinfer1;

#define CHECK_C(status) \
   do\
   {\
      CHECK_F(status != 0, "CUDA failure");\
   } while(0)

// node info definition 
class NodeInfo
{
   public:
      friend class TRTDetection;
      NodeInfo(){name = "None"; index=-1;};
      NodeInfo(const std::string n, const long i): name(n), index(i){};
      NodeInfo(const std::string n, const Dims d, const long i): name(n), dim(d), index(i){};
      std::string name;
   private:
      size_t getDataSize();
      Dims dim;
      long index;
};

size_t NodeInfo::getDataSize()
{
   auto data_size = 1;
   for (int i = 0; i < dim.nbDims; i++)
   {
      data_size *= dim.d[i];
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
         const int num_classes=80, 
         const float conf_thre=0.3,
         const float mask_thre=0.5);

      ~TRTDetection(){};
      virtual void load_ckpt(const std::string & ckpt) override;
      virtual void set_device(const int device) override;
      virtual void detectImage(const std::string & image_file) override;
      virtual void detectVideo(const std::string & video_file) override;

   private:
      ICudaEngine * engine;
      IExecutionContext * context;
      cv::Mat dataPreprocess(cv::Mat & images);
      void doInference(IExecutionContext & context, 
                       cv::Mat & input_images,
                       std::vector<float *> & output_results,
                       NodeInfo & input_nodes,
                       std::vector<NodeInfo> & output_nodes);
      void decodeOutputs(std::vector<Object> & objects,
                        const std::vector<float *> & output_results,
                        const std::vector<int> origin_size,
                        const std::vector<int> input_size,
                        const int num_classes,
                        const float conf_thre=0.3,
                        const float mask_thre=0.5);

};

TRTDetection::TRTDetection(
   const std::vector<int> in_size,
   const std::string input_name,
   const std::vector<std::string> output_names,
   const int num_classes, 
   const float conf_thre,
   const float mask_thre): Detection(in_size, num_classes, conf_thre, mask_thre)
{
   engine = nullptr;
   context = nullptr;
   input_node.name = input_name;
   for (int i = 0; i < output_names.size(); i++)
   {
      output_nodes.emplace_back(output_names[i], 0);
   }
}


void TRTDetection::decodeOutputs(std::vector<Object> & objects,
                                 const std::vector<float *> & output_results,
                                 const std::vector<int> origin_size,
                                 const std::vector<int> input_size,
                                 const int num_classes, 
                                 const float conf_thre,
                                 const float mask_thre)
{
   NodeInfo bbox_node = output_nodes[0];
   NodeInfo mask_node = output_nodes[1];
   CHECK_F(bbox_node.dim.d[0] == mask_node.dim.d[1], "mask and bbox out put");
   float * mask_out = output_results[0];
   float * bbox_out = output_results[1];
   double scale_ratio = std::min(input_size[0] / (origin_size[0] * 1.0), input_size[1] / (origin_size[1] * 1.0));
   for (int index=0; index < output_nodes[0].dim.d[0]; index++)
   {
      const auto mask_out_h = mask_node.dim.d[1];
      const auto mask_out_w = mask_node.dim.d[2];
      cv::Mat mask = cv::Mat::zeros(mask_out_h, mask_out_w, CV_32FC1);
      CHECK_F(mask.isContinuous(), "Only support continuous mask");
      float x1 = bbox_out[index * 6 + 0] / scale_ratio;
      float y1 = bbox_out[index * 6 + 1] / scale_ratio;
      float x2 = bbox_out[index * 6 + 2] / scale_ratio;
      float y2 = bbox_out[index * 6 + 3] / scale_ratio;
      float w = x2 - x1; float h = y2 - y1;
      float score = bbox_out[index * 6 + 4];
      if (score > conf_thre)
      {
         auto class_id = static_cast<int32_t>(bbox_out[index * 6 + 5]);
         cv::Mat bit_mask = cv::Mat::zeros(mask_out_h, mask_out_w, CV_8UC1);
         cv::Mat bit_mask = mask > mask_thre;
         cv::Size in_mask_size = cv::Size(mask_out_w / scale_ratio, mask_out_h / scale_ratio);
         cv::Mat in_bit_mask = cv::Mat::zeros(in_mask_size, CV_8UC1);
         cv::resize(bit_mask, in_bit_mask, in_mask_size);
         // emplace back 不用使用 {} 而 push back 需要使用
         objects.emplace_back(
            (cv::Rect){x1, y1, x2, y1},
            in_bit_mask,
            class_id,
            score
         );
      }
   }
}

void TRTDetection::detectImage(const std::string & image_file)
{
   // const int max_thread_num = std::min(static_cast<int32_t>(std::thread::hardware_concurrency()), 8);
   // ThreadPool pool(max_thread_num);
   cv::Mat image = cv::imread(image_file);
   cv::Mat input_image = dataPreprocess(image);
   origin_size[0] = image.rows;
   origin_size[1] = image.cols;
   // origin_size[0] = image.rows;
   // origin_size[1] = image.cols;
   // double scale_ratio = std::min(input_size[0] / (origin_size[0] * 1.0), input_size[1] / (origin_size[0] * 1.0));
   auto start_time = std::chrono::system_clock::now();
   std::vector<float *> output_results;
   doInference(*context, input_image, output_results, input_node, output_nodes);
   auto end_time = std::chrono::system_clock::now();
   LOG_F(INFO, "Inference time is %d ms", std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count());
   std::vector<Object> objects;
   decodeOutputs(objects, output_results, origin_size, input_size, num_classes, conf_thre, mask_thre);
}

void TRTDetection::detectVideo(const std::string & video_file)
{
   LOG_F(INFO, "Video file path is %s", (const char *)video_file.data());
}

cv::Mat TRTDetection::dataPreprocess(cv::Mat & image)
{
   cv::Mat resize_image = Detection::image_resize(image, input_size);
   resize_image.convertTo(resize_image, CV_32FC3, 1.f / 255.f);
   cv::subtract(resize_image, cv::Scalar(0.485f, 0.456f, 0.406f), resize_image, cv::noArray(), -1);
   cv::divide(resize_image, cv::Scalar(0.229f, 0.224f, 0.225f), resize_image, 1, -1);
   return resize_image;
}

void TRTDetection::doInference(IExecutionContext & context, 
                               cv::Mat & input_image, 
                               std::vector<float *> & output_results, 
                               NodeInfo & input_node,
                               std::vector<NodeInfo> & output_nodes)
{
   const ICudaEngine & engine = context.getEngine();
   const int buffers_Nb = static_cast<int32_t> (engine.getNbBindings());
   CHECK_F(buffers_Nb == static_cast<int32_t> (1 + output_results.size()), "Input + Output dims");
   void * buffers[buffers_Nb];
   float * input = new float [input_node.getDataSize()];
   auto channels = input_image.channels();
   auto image_h = input_image.rows;
   auto image_w = input_image.cols;
   for (size_t c = 0; c < channels; c++)
   {
      for (size_t h = 0; h < image_h; h++)
      {
         for (size_t w = 0; w < image_w; w++)
         {
            input[c * image_w * image_h + h * image_w + w] = (float)input_image.at<cv::Vec3f>(h, w)[c];
         }
      }
   }
   // Malloc Input Data
   CHECK_C(cudaMalloc(&buffers[input_node.index], input_node.getDataSize() * sizeof(float)));
   // Malloc Output Data 
   for (std::vector<NodeInfo>::iterator node = output_nodes.begin(); node != output_nodes.end(); node++)
   {
      CHECK_C(cudaMalloc(&buffers[node->index], node->getDataSize() * sizeof(float)));
   }

   // Create stream
   cudaStream_t stream;
   CHECK_C(cudaStreamCreate(&stream));
   // Copy Input Memory
   CHECK_C(cudaMemcpyAsync(buffers[input_node.index], input, input_node.getDataSize() * sizeof(float), cudaMemcpyHostToDevice, stream));
   // Inference 
   context.enqueue(1, buffers, stream, nullptr);
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
   for (std::vector<NodeInfo>::iterator node = output_nodes.begin(); node != output_nodes.end(); node++)
   {
      CHECK_C(cudaFree(buffers[node->index]));
   }
   delete [] input;
}

void TRTDetection::set_device(const int device){
   CHECK_F(device != -1, "Can't support inference in cpu");
   LOG_F(INFO, "Model will be inference in cuda: %d", device);
   cudaSetDevice(device);
}

void TRTDetection::load_ckpt(const std::string & ckpt){
   char * trtModelStream(nullptr);
   size_t size{0};
   LOG_F(INFO, "Engine file is %s", (const char *)ckpt.data());
   std::ifstream file(ckpt, std::ios::binary);
   CHECK_F(file.good(), "File is not opened correctly");
   file.seekg(0, file.end);
   size = file.tellg();
   file.seekg(0, file.beg);
   trtModelStream = new char[size];
   CHECK_F(trtModelStream != nullptr, "trtModelStream data buffer is created fail");
   file.read(trtModelStream, size);
   file.close();
   CHECK_F((bool)initLibNvInferPlugins(&TLogger, ""), "Init NvInfer Plugin");
   IRuntime * runtime = createInferRuntime(TLogger);
   CHECK_F(runtime != nullptr, "runtime");
   engine = runtime->deserializeCudaEngine(trtModelStream, size);
   CHECK_F(engine != nullptr, "engine");
   context = engine->createExecutionContext();
   CHECK_F(context != nullptr, "context");
   input_node.index = engine->getBindingIndex((const char *)input_node.name.data());
   auto out_dims = engine->getBindingDimensions(input_node.index);
   input_node.dim = out_dims;
   for (std::vector<NodeInfo>::iterator node = output_nodes.begin(); node != output_nodes.end(); node++)
   {
      node->index = engine->getBindingIndex((const char *)(node->name).data());
      auto out_dims = engine->getBindingDimensions(node->index);
      node->dim = out_dims;
   }
   delete [] trtModelStream;
}

#endif
