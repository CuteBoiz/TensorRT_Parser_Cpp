# TensorRT_Parser_Cpp

 TensorRT model **convert** (from ***Onnx*** engine) and **inference** in C++.

The Onnx model can be run on any system with difference platform (Operating system/ CUDA / CuDNN / TensorRT) but take a lot of time to parse.
Convert the Onnx model to TensorRT model (.trt) help you save a lot of parsing time (4-10 min) but can only run on fixed system you've built.

## I. Prerequiste.

- [CUDA/CUDNN/TensorRT Installation Guide](https://github.com/CuteBoiz/Ubuntu_Installation/blob/master/cuda.md)
- [Install OpenCV From Source with CUDA support](https://github.com/CuteBoiz/Ubuntu_Installation/blob/master/opencv.md)

## II. Setup.

#### 1. Clone and set path.

```sh
git clone https://github.com/CuteBoiz/TensorRT_Parser_Cpp
cd TensorRT_Parser_Cpp
gedit CMakeList #Then change my TensorRT path to your TensorRT path(include and lib)
```

#### 2. Add dirent.h to C++ library (Windows Only)

 move dirent.h file from [Additional files](https://github.com/CuteBoiz/TensorRT_Parser_Cpp/tree/main/Addition%20files) to `Visual-Studio-Installed-Path\2017\Community\VC\Tools\MSVC\14.16.27023\include`

#### 3. Build.

```sh
mkdir build && build
cmake ..
make
```

## III. Export Onnx model to TensorRT model (.trt).
  - Export:
    ```sh
    ./main -e "model_path" "max_batch_size" ("fp16")
    ```
    **Note:** The only GPUs with full-rate FP16 Fast mode performance are Tesla P100, Quadro GP100, and Jetson TX1/TX2.

  - Example:
    ```sh
    ./main -e ../2020_0421_0925.onnx 10
    ./main -e ../2020_0421_0925.onnx 1 fp16
    ```

## IV. Export Onnx model to TensorRT model (.trt) with dynamic input shape.
  - Export:
    ```sh
    ./main -ed "model_path" "max_batch_size" "input tensor name" "dimension1" "dimension2" "dimension3" ("fp16")
    ```
    **Note:** To get input tensor name and shape of model: Use [Netron](https://github.com/lutzroeder/netron).

  - Example:
    ```sh
    ./main -ed ../2020_0421_0925.onnx 10 input_1 128 128 3 
    ./main -ed ../2020_0421_0925.onnx 1 input:0 3 640 640 fp16
    ```

## V. Inference:
  - Inference:
    ```sh
    ./main -i "model_path" "images_folder_path" "batch_size"
    ```

  - Example:
    ```sh
    ./main -i ../2020_0421_0925.trt ../Dataset/Test/ 10
    ```
    
## VI. TO-DO

- **Fix split image on GPU bug.**
- Check multiple outputs model.
- Simplify the main process.
- Return result vectors.
