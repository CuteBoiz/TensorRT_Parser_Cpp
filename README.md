# TensorRT_Parser_Cpp

 TensorRT model **convert** (from ***Onnx*** engine) and **inference** in C++.

The Onnx model can be run on any system with difference platform (Operating system/ CUDA / CuDNN / TensorRT) but take a lot of time to parse.
Convert the Onnx model to TensorRT model (.trt) help you save a lot of parsing time (4-10 min) but can only run on fixed system you've built.

## I. Prerequiste.

- [CUDA/CUDNN/TensorRT](https://github.com/CuteBoiz/Ubuntu_Installation/blob/master/cuda.md)
- [OpenCV From Source with CUDA support](https://github.com/CuteBoiz/Ubuntu_Installation/blob/master/opencv.md)

## II. Setup.

#### 1. Clone and set path.

```sh
git clone https://github.com/CuteBoiz/TensorRT_Parser_Cpp
cd TensorRT_Parser_Cpp
gedit CMakeList #Then change my TensorRT path to your TensorRT path(include and lib)
```

#### 2. Add `dirent.h` to C++ library. *(For Visual Studio Only)*

 move dirent.h file from [Additional files](https://github.com/CuteBoiz/TensorRT_Parser_Cpp/tree/main/Addition%20files) to `Visual-Studio-Installed-Path\201x\Community\VC\Tools\MSVC\14.16.27023\include`

#### 3. Build.

```sh
mkdir build && build
cmake ..
make
```

## III. Export Onnx model to TensorRT model (.trt).
```sh
./main export --weight (--maxbatchsize) (--fp16) (--workspace) (--tensor) (--dims) (--gpu)
```
- Arguments:
    - `--weight` `string`: path to onnx engine `required`.
    - `--fp16` `store_true`: use FP16 fast mode (x2 inference time) **default=false**.
    - `--maxbatchsize` `int`:  inference max batchsize **default=1**.
    - `--workspace` `int`: max workspace size **default=1300 MB**.
    - `--tensor` `string`: input tensor's name ***(dynamic shape input only)***.
    - `--dims` `array(int)`: input tensor's dimension ***(dynamic shape input only)***. 
    - `--gpu` `int` : gpu number **(default=0)**.

   **Note:** The only GPUs with full-rate FP16 Fast mode performance are Tesla P100, Quadro GP100, and Jetson TX1/TX2.

## IV. Inference:
```sh
./main infer --weight --data (--batchsize) (--softmax) (--gpu)
```
- Arguments:
    - `--weight` `string`: path to tensorrt engine `required`.
    - `--data` `string`: path to inference images's folder `required`.
    - `--batchSize` `int`: inference batchsize **default=1**.
    - `--softmax` `store_true`: add softmax to last layer of engine **default=false**.
    - `--gpu` `int`: gpu number **(default=0)**.

## VI. TO-DO

- [ ] Multiple inputs model.
- [x] Multiple outputs model.
- [ ] Add Channel last image allocate.
- [x] Switch Primary GPU. 
- [x] Multi-type cast for arguments (Easy to maintain).
- [ ] Non-image input model.
- [ ] Add examples.
- [ ] 2D,3D tensor sofmax execute.
- [ ] Remove "Segmentation fault (core dumped)" at ending of inference. 
