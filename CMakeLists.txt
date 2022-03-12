cmake_minimum_required(VERSION 3.8)
project(yolox-inst)
add_definitions(-std=c++11)
set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
# set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -lpthread -ldl -Wall -ofast -Wfatal-errors -D_MWAITXINTRIN_H_INCLUDED")

# Set Packages Path
set(OpenCV_DIR /home/rcf/OPENCV/opencv-3.4.8/share/OpenCV)
# set(Torch_DIR /home/rcf/LibTorch/libtorch_17_10_2/share/cmake/Torch)
# set(ATen_DIR /home/rcf/LibTorch/libtorch_17_10_2/share/cmake/ATen)

# find CUDA Packages
find_package(CUDA REQUIRED)
include_directories(/home/rcf/CUDA/cuda-10.2/include)
link_directories(/home/rcf/CUDA/cuda-10.2/lib64)

# find Opencv Packages
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# find Torch Packages
# find_package(Torch REQUIRED)
# include_directories(${TORCH_INCLUDE_DIRS})
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

# find ATen Packages
# find_package(ATen REQUIRED)
# include_directories(${ATEN_INCLUDE_DIRS})

# find Tread Packages
# find_package(Threads)

# include and link TensorRT 
include_directories(/home/rcf/TensorRT/TensorRT-8.4.0.6/include)
link_directories(/home/rcf/TensorRT/TensorRT-8.4.0.6/lib)

# find local header files and src files 
include_directories(${PROJECT_SOURCE_DIR}/include)
aux_source_directory(${PROJECT_SOURCE_DIR}/src DIR_SRCS)

add_executable(${CMAKE_PROJECT_NAME} ${DIR_SRCS})
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)

# link static and dynamic libraries 
target_link_libraries(${CMAKE_PROJECT_NAME} ${CMAKE_THREAD_LIBS_INIT})
target_link_libraries(${CMAKE_PROJECT_NAME} dl)

# target_link_libraries(${CMAKE_PROJECT_NAME} ${ATEN_LIBRARIES})
# target_link_libraries(${CMAKE_PROJECT_NAME} ${TORCH_LIBRARIES})

target_link_libraries(${CMAKE_PROJECT_NAME} nvinfer)
target_link_libraries(${CMAKE_PROJECT_NAME} nvinfer_plugin)
target_link_libraries(${CMAKE_PROJECT_NAME} cudart)

target_link_libraries(${CMAKE_PROJECT_NAME} ${OpenCV_LIBS})

add_definitions(-O0 -pthread)
