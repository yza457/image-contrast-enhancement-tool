cmake_minimum_required(VERSION 2.8)

project(img2gpu)

find_package(CUDA REQUIRED)

# Hardcoding this on the asb9840u-b01. Unable to find the non standard cuda directory
set(CUDA_TOOLKIT_ROOT_DIR   "/usr/local/cuda-10.0/")
set(CMAKE_CXX_FLAG "-g -Wall -D_FORCE_INLINES")

# Adjust according to your platform
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-arch compute_35)
#set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}; -D_FORCE_INLINES -O3 -gencode arch=compute_35,code=sm_35)

# The second option is to make use of utility functions defined in samples/common/inc
include_directories(${CUDA_INCLUDE_DIRS} "./common/inc")

message(STATUS ${CUDA_INCLUDE_DIRS})
message(STATUS ${CUDA_TOOLKIT_ROOT_DIR})
# set(CMAKE_MODULE_PATH "/usr/share/cmake/Modules" ${CMAKE_MODULE_PATH})

set(CPPSRCS
   main.cpp
   histogram-equalization.cpp
   contrast-enhancement.cpp
)

set(CUDASRCS
   gpu.cu
   # 2.cu
   # etc 
)

CUDA_ADD_EXECUTABLE(${PROJECT_NAME} ${CUDASRCS} ${CPPSRCS})

