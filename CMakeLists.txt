cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(TensorRT_Parser)

add_definitions(-D_GLIBCXX_USE_CXX11_ABI=0)

set(CMAKE_INCLUDE_CURRENT_DIR ON)
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)


#cuda
include_directories(/usr/local/cuda/include)
link_directories(/usr/local/cuda/lib64)

#tensorrt
include_directories(/home/tanphatnguyen/TensorRT-7.2.3.4/include)
link_directories(/home/tanphatnguyen/TensorRT-7.2.3.4/lib)

find_package(OpenCV REQUIRED)

#aux_source_directory(src DIR_SRCS)

link_libraries(stdc++fs)

add_executable(main main.cpp
			onnxparser.h
			onnxparser.cpp)


target_link_libraries( main ${OpenCV_LIBS})
target_link_libraries( main cudart)
target_link_libraries( main nvinfer)
target_link_libraries( main nvparsers)
target_link_libraries( main stdc++fs)
target_link_libraries( main nvonnxparser.so)
