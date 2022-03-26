cmake_minimum_required(VERSION 3.8)
project(yolox-inst)
add_definitions(-std=c++14)
set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14 -lpthread -ldl -Wall -D_MWAITXINTRIN_H_INCLUDED")
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14 -lpthread -ldl -Wall -ofast -Wfatal-errors -D_MWAITXINTRIN_H_INCLUDED")

# Options
option(USE_YAML "Whether or not to use yaml to conver params" ON)
option(USE_TENSORRT "Wheter or not to use tensorrt" ON)
option(USE_LIBTORCH "Whether or not to use libtorch" OFF)
option(USE_ONNXRUNTIME "Wheter or not to use onnxrumtime" OFF)

find_package(Threads)
set(EXTRA_LIBS ${EXTRA_LIBS} ${CMAKE_THREAD_LIBS_INIT})
# find_package(OpenMP)
# set(EXTRA_LIBS ${})

find_package(CUDA REQUIRED)
include_directories(/usr/local/cuda/include)
link_directories(/usr/local/cuda/lib64)
set(EXTRA_LIBS ${EXTRA_LIBS} cudart)
if (FOUND_CUDA)
    message(STATUS "CUDA Library status:")
    message(STATUS "    version: ${CUDA_VERSION}" \n)
endif ()

# include_directories(/home/r/CUDNN/cudnn10.2/include)
# link_directories(/home/r/CUDNN/cudnn10.2/lib64)

# find Opencv Packages
set(OpenCV_DIR /home/r/OPENCV/opencv-3.4.8/share/OpenCV)
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
set(EXTRA_LIBS ${EXTRA_LIBS} ${OpenCV_LIBS})
if (FOUND_OpenCV)
    message(STATUS "OpenCV Library status:")
    message(STATUS "    version: ${OpenCV_VERSION}" \n)
endif ()

# find YAML-CPP
if (USE_YAML)
    set(YAML-CPP_DIR /home/r/YAML/yaml-0.6.0/share/cmake/yaml-cpp)
    add_definitions(-DUSE_YAML)
    find_package(YAML-CPP)
    include_directories(/home/r/YAML/yaml-0.6.0/include)
    link_directories(/home/r/YAML/yaml-0.6.0/lib)
    set(EXTRA_LIBS ${EXTRA_LIBS} yaml-cpp)
endif (USE_YAML)

if (USE_LIBTORCH)
    set(Torch_DIR /home/r/LibTorch/libtorch_cu110_170/share/cmake/Torch)
    set(ATen_DIR /home/r/LibTorch/libtorch_cu110_170/share/cmake/ATen)
    add_definitions(-DUSE_LIBTORCH)
    find_package(Torch)
    find_package(ATen)
    include_directories(${TORCH_INCLUDE_DIRS})
    include_directories(${ATEN_INCLUDE_DIRS})
    set(EXTRA_LIBS ${EXTRA_LIBS} ${ATEN_LIBRARIES} ${TORCH_LIBRARIES})
endif (USE_LIBTORCH)

# include and link TensorRT 
if (USE_TENSORRT)
    add_definitions(-DUSE_TENSORRT)
    include_directories(/home/r/TensorRT/cuda-10.2/TensorRT-8.4.0.6/include)
    link_directories(/home/r/TensorRT/cuda-10.2/TensorRT-8.4.0.6/lib)
    set(EXTRA_LIBS ${EXTRA_LIBS} nvinfer nvinfer_plugin)
endif (USE_TENSORRT)

if (USE_ONNXRUNTIME)
    set(onnxruntime_DIR /home/r/ONNX/onnxruntime-linux-x64-1.10.0/share/cmake/onnxruntime)
    add_definitions(-DUSE_ONNXRUNTIME)
    find_package(onnxruntime)
    set(EXTRA_LIBS ${EXTRA_LIBS} onnxruntime)
endif (USE_ONNXRUNTIME)

# find local header files and src files 
include_directories(${PROJECT_SOURCE_DIR}/include)
aux_source_directory(${PROJECT_SOURCE_DIR}/src DIR_SRCS)

# generate exec file
add_executable(${CMAKE_PROJECT_NAME} ${DIR_SRCS})
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)

# link static and dynamic libraries 
target_link_libraries(${CMAKE_PROJECT_NAME} dl)
target_link_libraries(${CMAKE_PROJECT_NAME} ${EXTRA_LIBS})

# add_definitions(-O0 -pthread)
add_definitions(-O2 -pthread)
