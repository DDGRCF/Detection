#include <iostream>
#include <fstream>
#include <numeric>
#include <chrono>
#include <opencv2/opencv.hpp>
#include <vector>
#include <string>
#include <stdlib.h>
#include "utils.hpp"
#include "cxxopts.hpp"
#include "loguru.hpp"


#ifdef USE_ONNXRUNTIME
    #include <onnxruntime_cxx_api.h> 
#endif

#ifdef USE_TENSORRT
    #include "trtdetection.hpp"
#endif

#ifdef USE_LIBTORCH
    #include "tspdetection.hpp"
#endif

#ifdef USE_YAML
    #include <yaml-cpp/yaml.h>
    #include <yaml-cpp/node/parse.h>
#endif


int main (int argc, char * argv[]){
    loguru::init(argc, argv);
    check_dir("./bin/logs");
    loguru::add_file("./bin/logs/depoly.log", loguru::Append, loguru::Verbosity_MAX);
    loguru::add_file("./bin/logs/latest_readable.log", loguru::Truncate, loguru::Verbosity_INFO);
    loguru::g_stderr_verbosity = 1;

    cxxopts::Options parser(argv[0], "A CondInst / BoxInst Depoly");
    parser.allow_unrecognised_options().add_options()
            ("ckpt-path", "torchscript / onnx / tensorrt model engine / weights model", cxxopts::value<std::string>())
            ("data-path", "data to show", cxxopts::value<std::string>())
            ("detect-type", "Image / Video", cxxopts::value<std::string>()->default_value("image"))
            ("device", "device to inference, for cpu: -1 / for gpu :0,1,2", cxxopts::value<int>()->default_value("0"))
            ("input-size", "net_input_size", cxxopts::value<std::vector<int>>()->default_value("{640,640}"))
            ("input-name", "names of input node", cxxopts::value<std::string>()->default_value("{\"input\"}"))
            ("output-name", "names of output node", cxxopts::value<std::vector<std::string>>()->default_value("{\"bbox_out\",\"mask_out\"}"))
            ("num-classes", "number of classes", cxxopts::value<int>()->default_value("80"))
            ("score-thre", "score for vis", cxxopts::value<float>()->default_value("0.3"))
            ("mask-thre", "mask for vis", cxxopts::value<float>()->default_value("0.5"))
            ("is-show", "whether to show images or videos", cxxopts::value<bool>()->default_value("false"))
            ("save-path", "Save Paht", cxxopts::value<std::string>()->default_value("None"))
            ("is-verbose", "Verbose for Detecting", cxxopts::value<bool>()->default_value("true"))
            ("camera-id", "Camera for detecting", cxxopts::value<int>()->default_value("0"))
            ("use-local-device", "Whether use local device", cxxopts::value<bool>()->default_value("false"))
            ("config-path", "config" , cxxopts::value<std::string>()->default_value("None"))
            ("h,help", "warning if set beyand number of arguments");

    auto opt = parser.parse(argc, argv);
    if (opt.count("help")){
        std::cout << parser.help() << std::endl;
    }
    const std::string model_type = check_model(opt["ckpt-path"].as<std::string>());

#ifdef USE_YAML
    YAML::Node config = YAML::LoadFile(opt["config-path"].as<std::string>());
    const std::vector<std::string> class_names = config["class-names"] ? config["class-names"].as<std::vector<std::string>>() : coco_class_names;
    const std::vector<std::vector<float>> class_colors = config["class-colors"] ? config["class-colors"].as<std::vector<std::vector<float>>>() : coco_color_list;
    config["input-size"] = opt["input-size"].as<std::vector<int>>();
    config["input-name"] = opt["input-name"].as<std::string>();
    config["output-name"] = opt["output-name"].as<std::vector<std::string>>();
    config["num-classes"] = opt["num-classes"].as<int>();
#else 
    const std::vector<std::string> class_names = coco_class_names;
    const std::vector<std::vector<float>> class_colors = coco_color_list;
#endif 
    // Load the Model
    DECT::Detection * detection;
    if (model_type == "TensorRT")
    {
#ifdef USE_TENSORRT
    detection = new DECT::TRTDetection(
        opt["input-size"].as<std::vector<int>>(), 
        opt["input-name"].as<std::string>(), 
        opt["output-name"].as<std::vector<std::string>>(),
        opt["num-classes"].as<int>());
#endif
    }
    else if (model_type == "TorchScript")
    {
#ifdef USE_LIBTORCH
    detection = new DECT::TSPDetection(
        opt["input-size"].as<std::vector<int>>(),
        opt["num-classes"].as<int>());
#endif
    }
    else 
    {
        LOG_F(ERROR, "Error");
        abort();
    }
    // Detect
    detection->set_device(opt["device"].as<int>());
    detection->load_ckpt(opt["ckpt-path"].as<std::string>());
    detection->detect(
        opt["data-path"].as<std::string>(), 
        opt["score-thre"].as<float>(),
        opt["mask-thre"].as<float>(),
        opt["save-path"].as<std::string>(),
        class_names,
        class_colors,
        opt["is-show"].as<bool>(),
        opt["is-verbose"].as<bool>(),
        opt["detect-type"].as<std::string>(),
        opt["camera-id"].as<int>(),
        opt["use-local-device"].as<bool>());

#ifdef USE_YAML
    std::ofstream config_out("./bin/configs/config_out.yaml");
    config_out << config;
#endif
    // Delete the targets
    delete detection;
}

