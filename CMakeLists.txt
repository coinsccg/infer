cmake_minimum_required(VERSION 2.6)
project(infre)
add_definitions(-std=c++11)

option(CUDA_USE_STATIC_CUDA_RUNTIME OFF)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_BUILD_TYPE Debug)

# 如果你是不同显卡，请设置为显卡对应的号码参考这里：https://developer.nvidia.com/zh-cn/cuda-gpus#compute
set(CUDA_GEN_CODE "-gencode=arch=compute_75,code=sm_75")

# 设置opencv\cuda\cudnn\tensorrt的目录
set(OpenCV_DIR   "C:\\Opencv\\opencv\\build\\include")
set(CUDA_DIR     "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.2")
set(CUDNN_DIR    "C:\\Program Files\\Cudann")
set(TENSORRT_DIR "C:\\Program Files\\TensorRT-8.4.2.4")

# 自动查找cuda和opencv的包
find_package(CUDA REQUIRED)
find_package(OpenCV)

# 导入opencv、tensorrrt、cuda、cudann的头文件
include_directories(
    ${PROJECT_SOURCE_DIR}/src
    ${OpenCV_INCLUDE_DIRS}
    ${CUDA_DIR}\\include
    ${TENSORRT_DIR}\\include
    ${CUDNN_DIR}\\include
)

# 导入tensorrrt、cuda、cudann的库文件
link_directories(
    ${TENSORRT_DIR}\\lib
    ${CUDA_DIR}\\lib\\x64
    ${CUDNN_DIR}\\lib
)

set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -std=c++11 -Wall -O0 -pthread -w -g")
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -std=c++11 -O0 -Xcompiler -fPIC -g -w ${CUDA_GEN_CODE}")

# 查找项目所有文件
file(GLOB_RECURSE cpp_srcs ${PROJECT_SOURCE_DIR}/src/*.cpp)
file(GLOB_RECURSE cuda_srcs ${PROJECT_SOURCE_DIR}/src/*.cu)

list(REMOVE_ITEM cpp_srcs main.cpp)

cuda_add_library(cucodes STATIC ${cuda_srcs})

# 生成可执行程序
#add_executable(infre ${cpp_srcs})

# 可执行文件链接动态库
#target_link_libraries(cucodes nvinfer nvonnxparser)
#target_link_libraries(cucodes cuda cublas cudart cudnn)
#target_link_libraries(infre ${OpenCV_LIBS})
#target_link_libraries(infre cucodes)

add_library(infre SHARED ${cpp_srcs})
target_link_libraries(infre nvinfer nvonnxparser cuda cublas cudart cudnn ${OpenCV_LIBS} cucodes)
