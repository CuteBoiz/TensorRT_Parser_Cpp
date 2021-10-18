cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(TensorRT_Parser)

set(CMAKE_INCLUDE_CURRENT_DIR ON)
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)

#Cuda
include_directories(/usr/local/cuda-11.1/include)
link_directories(/usr/local/cuda-11.1/lib64)

#TensorRT
include_directories(/home/phatnt/TensorRT-7/include)
link_directories(/home/phatnt/TensorRT-7/lib)


#aux_source_directory(src DIR_SRCS)

link_libraries(stdc++fs)

add_executable(main 
			main.cpp
			utils.h
			utils.cpp
			)

target_link_libraries( main cudart)
target_link_libraries( main nvinfer)
target_link_libraries( main nvparsers)
target_link_libraries( main stdc++fs)
target_link_libraries( main nvonnxparser.so)

