# <div align=center> TensorRT Parser Cpp </div>

TensorRT module in C/C++ 

The Onnx engine can be run on any system with difference platform (Os/Cuda/CuDNN/TensorRT version).

## I. Prerequiste

### A.Linux
- **yaml-cpp**
    ```sh
    git clone https://github.com/jbeder/yaml-cpp
    cd yaml-cpp
    mkdir build && cd build
    cmake .. -DYAML_BUILD_SHARED_LIBS=on 
    ```
- **Install [Cuda/CuDNN/TensorRT](https://github.com/CuteBoiz/Ubuntu_Installation/blob/master/wiki/cuda.md)**

- **[OpenCV with CUDA support](https://github.com/CuteBoiz/Ubuntu_Installation/blob/master/wiki/opencv.md) (C++/Python)**


### B. Windows

- **[Install VisualStudio/Cuda/CuDNN/TensorRT](https://github.com/CuteBoiz/TensorRT_Dev_VS)**

- **Download [dirent.h](https://github.com/tronkko/dirent/blob/master/include/dirent.h) then put inside this folder**
    ```sh
    Visual-Studio-Installed-Path\201x\Community\VC\Tools\MSVC\xx.xx.xxxxx\include
    ````
  
## II. Download & Build

```sh
git clone https://github.com/CuteBoiz/TensorRT_Parser_Cpp.git
cd TensorRT_Parser_Cpp
mkdir build && cd build
cmake .. -DTRT:=/path/to/tensorrt #ex: cmake .. -DTRT:=/home/pi/Libraries/TensorRT-8.4.3.1
make
```

## III. Convert Onnx (.onnx) to TensorRT (.trt).

```sh
./tensorrt_cpp convert /path/to/config.yaml_file
```

<details> 
<summary><b>Examples</b></summary>
 
- **Export Onnx engine to TensorRT engine.**
 
  ```sh
  ./tensorrt_cpp convert ../config/onnx_config.yaml
  ./tensorrt_cpp convert ../config/onnx_config_dynamic.yaml
  ```

</details>

## IV. TensorRT Inference. </div>

```sh
./main infer /path/to/trt_engine /path/to/data  (softmax) (gpuID)
```

*Data could be path to video/image/images folder*
*gpuID for select gpuID in multi-gpu system inference*

<details> 
<summary><b>Examples</b></summary>
 
- **TensorRT engine Inference.**
 
  ```sh
  ./tensorrt_cpp infer  home/usrname/classifier.trt image.jpg 
  ./tensorrt_cpp infer  classifier.trt ./test_images 1
  ./tensorrt_cpp infer  classifier.trt video.mp4 softmax
  ./tensorrt_cpp infer  ../classifier.trt ../images/ softmax 6
  ```

</details>
 
## Features:
- **Support**
  - [x] Multiple inputs.
  - [x] Multiple outputs.
  - [x] Non-image input.
  - [x] Channel 1st and last image input (CHW/HWC).
  - [x] 2D,3D,4D,5D tensor softmax.
  - [x] kINT/kBOOL/kFLOAT tensor.
- **Additions**
  - [x] Switch Primary GPU. 
  - [ ] Add CudaStream (Multiple GPU inference).


