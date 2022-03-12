#include <iostream>
#include <fstream>
#include <sstream>
#include <numeric>
#include <chrono>
#include <dirent.h>
#include <vector>
#include <string>
#include <stdlib.h>
#include "utils.h"
#include "cxxopts.hpp"
#include "loguru.hpp"
#include <opencv2/opencv.hpp>
#include "trtdetection.hpp"

int main (int argc, char * argv[]){
    loguru::init(argc, argv);
    loguru::add_file("./bin/logs/depoly.log", loguru::Append, loguru::Verbosity_MAX);
    loguru::add_file("./bin/logs/latest_readable.log", loguru::Truncate, loguru::Verbosity_INFO);
    loguru::g_stderr_verbosity = 1;

    cxxopts::Options parser(argv[0], "A CondInst / BoxInst Depoly");
    parser.allow_unrecognised_options().add_options()
            ("ckpt-path", "torchscript / onnx / tensorrt model engine / weights model", cxxopts::value<std::string>())
            ("data-path", "data to show", cxxopts::value<std::string>())
            ("device", "device to inference, for cpu: -1 / for gpu :0,1,2", cxxopts::value<int>()->default_value("0"))
            ("input-size", "net_input_size", cxxopts::value<std::vector<int>>()->default_value("{640,640}"))
            ("input-name", "names of input node", cxxopts::value<std::string>()->default_value("{\"input\"}"))
            ("output-name", "names of output node", cxxopts::value<std::vector<std::string>>()->default_value("{\"bbox_out\",\"mask_out\"}"))
            ("num-classes", "number of classes", cxxopts::value<int>()->default_value("80"))
            ("score-thre", "score for vis", cxxopts::value<float>()->default_value("0.3"))
            ("mask-thre", "mask for vis", cxxopts::value<float>()->default_value("0.5"))
            ("is-show", "whether to show images or videos", cxxopts::value<bool>()->default_value("false"))
            ("save-path", "Save Paht", cxxopts::value<std::string>()->default_value("None"))
            ("h,help", "waringin if set beyand number of arguments");

    auto opt = parser.parse(argc, argv);
    if (opt.count("help")){
        std::cout << parser.help() << std::endl;
    }

    TRTDetection trtdetection(
        opt["input-size"].as<std::vector<int>>(), 
        opt["input-name"].as<std::string>(), 
        opt["output-name"].as<std::vector<std::string>>(),
        opt["num-classes"].as<int>(),
        opt["score-thre"].as<float>(),
        opt["mask-thre"].as<float>());
    trtdetection.load_ckpt(opt["ckpt-path"].as<std::string>());
    trtdetection.set_device(opt["device"].as<int>());
}
