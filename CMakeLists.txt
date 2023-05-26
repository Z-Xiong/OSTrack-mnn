cmake_minimum_required(VERSION 3.16)

set(CMAKE_C_COMPILER "gcc")
set(CMAKE_CXX_COMPILER "g++")
set(CMAKE_CXX_FLAGS "-fPIC -std=c++14")
set(CMAKE_C_FLAGS "-fPIC -std=c11  -O3 -mavx512f")
set(CMAKE_BUILD_TYPE "Release")
project(ostrack-mnn-demo)

set(CMAKE_CXX_STANDARD 14)

include_directories(${PROJECT_SOURCE_DIR} /usr/include/opencv4 /home/zxiong/hd0/Github/MNN/include)

link_directories(/usr/lib/x86_64-linux-gnu /home/zxiong/hd0/Github/MNN/build-avx512)

add_executable(ostrack-mnn-gcc-release main.cpp ostrack_mnn.cpp ostrack_mnn.h)

target_link_libraries(ostrack-mnn-gcc-release libMNN.so libopencv_core.so libopencv_highgui.so libopencv_imgproc.so libopencv_video.so libopencv_videoio.so libopencv_imgcodecs.so)
#-lglslang -lshaderc_shared
