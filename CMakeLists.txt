cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(tensorrt_cpp)

set(CMAKE_INCLUDE_CURRENT_DIR ON)
set(CMAKE_CXX_STANDARD 17)

# Check input TensorRT path
if ("${TRT}" STREQUAL "")
    message(FATAL_ERROR "Missing argument '-DTRT'! Use 'cmake -DTRT:=path/to/tensorrt_library'")
endif()

set(TRT_Include_1 ${TRT}include)
set(TRT_Include_2 ${TRT}/include)

set(TRT_Lib_1 ${TRT}lib)
set(TRT_Lib_2 ${TRT}/lib)

if (EXISTS ${TRT_Include_1}) 
    set(TRT_Include ${TRT_Include_1})  
    message(STATUS "'${TRT_Include_1}' is exist")
elseif (EXISTS ${TRT_Include_2})
    set(TRT_Include ${TRT_Include_2})
    message(STATUS "'${TRT_Include_2}' is exist")
else ()
    message(FATAL_ERROR "'${TRT_Include_1}' or '${TRT_Include_2}' is not exist!")
endif()

if (EXISTS ${TRT_Lib_1}) 
    set(TRT_Lib ${TRT_Lib_1})  
    message(STATUS "'${TRT_Lib_1}' is exist")
elseif (EXISTS ${TRT_Lib_2})
    set(TRT_Lib ${TRT_Lib_2})
    message(STATUS "'${TRT_Lib_2}' is exist")
else ()
    message(FATAL_ERROR "'${TRT_Lib_1}' or '${TRT_Lib_2}' is not exist!")
endif()

include_directories(${TRT_Include})
link_directories(${TRT_Lib})

# Cuda
include_directories(/usr/local/cuda/include)
link_directories(/usr/local/cuda/lib64)

# OpenCV
find_package(OpenCV REQUIRED)
find_package(yaml-cpp REQUIRED)

set(SRC_FILES 
    src/main.cpp
    src/tensor.h 
    src/tensor.cpp
    src/logger.hpp
    src/utils.h
    src/utils.cpp 
    src/data.h
    src/data.cpp
    src/onnx_parser.h
    src/onnx_parser.cpp
    src/tensorrt_parser.h
    src/tensorrt_parser.cpp
)

add_executable(${PROJECT_NAME} ${SRC_FILES})

target_link_libraries( ${PROJECT_NAME} ${OpenCV_LIBS})
target_link_libraries( ${PROJECT_NAME} cudart)
target_link_libraries( ${PROJECT_NAME} nvinfer)
target_link_libraries( ${PROJECT_NAME} nvparsers)
target_link_libraries( ${PROJECT_NAME} stdc++fs)
target_link_libraries( ${PROJECT_NAME} nvonnxparser.so)
target_link_libraries( ${PROJECT_NAME} yaml-cpp)