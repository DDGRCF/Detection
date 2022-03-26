#ifndef UTILS_H
#define UTILS_H

#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <string>
#include <vector> 
#include <utility>
#include "loguru.hpp"
#include <glob.h>


int check_dir(const std::string path, bool verbose=false)
{
	int len = path.length();
	char tmpDirPath[256] = { 0 };
	for (int i = 0; i < len; i++)
	{
		tmpDirPath[i] = path[i];
		if (tmpDirPath[i] == '\\' || tmpDirPath[i] == '/')
		{
			if (access(tmpDirPath, 0) == -1)
			{
				int ret = mkdir(tmpDirPath, 00700);
				if (ret == -1) 
                {
                    return ret;
                }
                else if (verbose && ret != -1)
                {
                    LOG_F(INFO, "Create Dir %s", tmpDirPath);
                }
			}
		}
	}
	return 0;
}

int check_file(const std::string path, std::vector<std::string> & file_set, bool verbose=false, std::vector<std::string> patterns={})
{
    /**
     * @brief Check a single file or a set of dir's file 
     * @param
     * path: the path of file of dir
     * @return
     * 0: path presents file
     * 1: path presents dir
     * -1: path don't exist
     */
    struct stat s;
    if (stat(path.c_str(), &s) == 0)
    {
        if (s.st_mode & S_IFDIR)
        {
            for (auto pattern: patterns)
            {
                const std::string pattern_path = path + "/*." + pattern;
                glob_t gl;
                // CHECK_F(glob(pattern_path.c_str(), GLOB_ERR, nullptr, &gl) != 0, "Failed to load %s", pattern_path.c_str());
                glob(pattern_path.c_str(), GLOB_ERR, nullptr, &gl);
                for (int i = 0; i < gl.gl_pathc; i++)
                {
                    std::string subpath = gl.gl_pathv[i];
                    file_set.emplace_back(subpath);
                }
                globfree(&gl);
            }
            if (verbose)
                LOG_F(INFO, "Find Existing Dir: %s", path.c_str());
        }
        else if (s.st_mode & S_IFREG)
        {
            file_set.emplace_back(path);
            if (verbose)
                LOG_F(INFO, "Find File: %s", path.c_str());
        }
        else
        {
            if (verbose)
                LOG_F(ERROR, "Find Agnostic File Type: %s", path.c_str());
            return -1;
        }
    }
    else
    {
        if (verbose)
            LOG_F(ERROR, "File %s don't exist", path.c_str());
        return -1;
    }
}

int check_suffix(const std::string path, std::vector<std::string> patterns={}, bool verbose=false)
{
    const std::string suffix = path.substr(path.find_last_of('.')+1);
    for (auto pattern : patterns)
    {
        if (pattern == suffix)
        {
            if (verbose)
            {
                // LOG(INFO) << path << "matchs suffix of " << suffix << std::endl;
                LOG_F(INFO, "%s is matching suffix", path.c_str());
            }
            return 0;
        }
    }
    if (verbose)
    {
        LOG_F(ERROR, "Error suffix %s", suffix.c_str());
    }
    return -1;
}

std::string check_model(const std::string ckpt)
{
    const std::string ckpt_suffix = ckpt.substr(ckpt.find_last_of('.')+1);
    if (ckpt_suffix == "engine")
    {
        LOG_F(INFO, "Load the Model of TensorRT");
        return "TensorRT";
    }
    else if (ckpt_suffix == "onnx")
    {
        LOG_F(INFO, "Load the Model of ONNX");
        return "ONNX";
    }
    else if (ckpt_suffix == "pt")
    {
        LOG_F(INFO, "Load the Model of TorchScript");
        return "TorchScript";
    }
    else
    {
        LOG_F(INFO, "Don't support model type");
        abort();
    }

}

const std::vector<std::string> coco_class_names = {        
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
    "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
    "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush"
};

const std::vector<std::vector<float>> coco_color_list = {
    {0.000, 0.447, 0.741},
    {0.850, 0.325, 0.098},
    {0.929, 0.694, 0.125},
    {0.494, 0.184, 0.556},
    {0.466, 0.674, 0.188},
    {0.301, 0.745, 0.933},
    {0.635, 0.078, 0.184},
    {0.300, 0.300, 0.300},
    {0.600, 0.600, 0.600},
    {1.000, 0.000, 0.000},
    {1.000, 0.500, 0.000},
    {0.749, 0.749, 0.000},
    {0.000, 1.000, 0.000},
    {0.000, 0.000, 1.000},
    {0.667, 0.000, 1.000},
    {0.333, 0.333, 0.000},
    {0.333, 0.667, 0.000},
    {0.333, 1.000, 0.000},
    {0.667, 0.333, 0.000},
    {0.667, 0.667, 0.000},
    {0.667, 1.000, 0.000},
    {1.000, 0.333, 0.000},
    {1.000, 0.667, 0.000},
    {1.000, 1.000, 0.000},
    {0.000, 0.333, 0.500},
    {0.000, 0.667, 0.500},
    {0.000, 1.000, 0.500},
    {0.333, 0.000, 0.500},
    {0.333, 0.333, 0.500},
    {0.333, 0.667, 0.500},
    {0.333, 1.000, 0.500},
    {0.667, 0.000, 0.500},
    {0.667, 0.333, 0.500},
    {0.667, 0.667, 0.500},
    {0.667, 1.000, 0.500},
    {1.000, 0.000, 0.500},
    {1.000, 0.333, 0.500},
    {1.000, 0.667, 0.500},
    {1.000, 1.000, 0.500},
    {0.000, 0.333, 1.000},
    {0.000, 0.667, 1.000},
    {0.000, 1.000, 1.000},
    {0.333, 0.000, 1.000},
    {0.333, 0.333, 1.000},
    {0.333, 0.667, 1.000},
    {0.333, 1.000, 1.000},
    {0.667, 0.000, 1.000},
    {0.667, 0.333, 1.000},
    {0.667, 0.667, 1.000},
    {0.667, 1.000, 1.000},
    {1.000, 0.000, 1.000},
    {1.000, 0.333, 1.000},
    {1.000, 0.667, 1.000},
    {0.333, 0.000, 0.000},
    {0.500, 0.000, 0.000},
    {0.667, 0.000, 0.000},
    {0.833, 0.000, 0.000},
    {1.000, 0.000, 0.000},
    {0.000, 0.167, 0.000},
    {0.000, 0.333, 0.000},
    {0.000, 0.500, 0.000},
    {0.000, 0.667, 0.000},
    {0.000, 0.833, 0.000},
    {0.000, 1.000, 0.000},
    {0.000, 0.000, 0.167},
    {0.000, 0.000, 0.333},
    {0.000, 0.000, 0.500},
    {0.000, 0.000, 0.667},
    {0.000, 0.000, 0.833},
    {0.000, 0.000, 1.000},
    {0.000, 0.000, 0.000},
    {0.143, 0.143, 0.143},
    {0.286, 0.286, 0.286},
    {0.429, 0.429, 0.429},
    {0.571, 0.571, 0.571},
    {0.714, 0.714, 0.714},
    {0.857, 0.857, 0.857},
    {0.000, 0.447, 0.741},
    {0.314, 0.717, 0.741},
    {0.50, 0.5, 0}
};

#endif