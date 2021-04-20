# TensorRT_Parser_Cpp

Onnx and TensorRT model inference in C++

## I. Prerequiste:

- [CUDA/CUDNN/TensorRT](https://github.com/CuteBoiz/Ubuntu_Installation/blob/master/cuda.md)
- [OpenCV](https://github.com/CuteBoiz/Ubuntu_Installation/blob/master/opencv.md)

## II. How To Use:

### 1. Clone and set path.

```sh
git clone https://github.com/CuteBoiz/TensorRT_Parser_Cpp
cd TensorRT_Parser_Cpp
gedit CMakeList #Then change my TensorRT path to your TensorRT path(include and lib)
```

### 2. Build

```sh
mkdir build && build
cmake ..
make
```

### 3. Export Onnx model(.onnx) to TensorRT model (.trt):

The .onnx can be run on any system with diffenrece platform (Operating system/ CUDA / CuDNN / TensorRT) but take a lot of time to parse.
Convert to tensorRT file (.trt) help you save a lot of parsing time ( 4-10 min) but can only run on fixed system you've built.

```sh
./main -e "model_path"
```
Example:
```sh
./main -e ../2020_0421_0925.onnx
```

### 4. Inference:

This repo include both .trt and .onnx infer

```sh
./main -i "model_path" "images_folder_path"
```

Example:
```sh
./main -i ../2020_0421_0925.trt ../Dataset/Test/
```
