# <div align=center> TensorRT_Parser_Cpp </div>

<div align=center>
 <p><i> The Onnx model can be run on any system with difference platform (Operating system/ CUDA / CuDNN / TensorRT) but take a lot of time to parse.
Convert the Onnx model to TensorRT model (.trt) help you save a lot of parsing time (4-10 min) but can only run on fixed system you've built. </i></p>
 </div>

## <div align=center> I. Prerequiste. </div>

- [Install Cuda/CuDNN/TensorRT](https://github.com/CuteBoiz/Ubuntu_Installation/blob/master/cuda.md)
- [OpenCV with CUDA support (C++/Python)](https://github.com/CuteBoiz/Ubuntu_Installation/blob/master/opencv.md)

- Clone and set path.
  ```sh
  git clone https://github.com/CuteBoiz/TensorRT_Parser_Cpp
  cd TensorRT_Parser_Cpp
  gedit CMakeList #Then change my TensorRT path to your TensorRT path(include and lib)
  ```

- Move dirent.h file from [Additional files](https://github.com/CuteBoiz/TensorRT_Parser_Cpp/tree/main/Addition%20files) to `Visual-Studio-Installed-Path\201x\Community\VC\Tools\MSVC\14.16.27023\include` **(Visual Studio Only)**
- Build.
  ```sh
  mkdir build && build
  cmake ..
  make
  ```

## <div align=center> II. Export Onnx engine to TensorRT engine (.trt).  </div>
```sh
./main export --weight (--maxbatchsize) (--fp16) (--maxworkspace) (--tensor) (--gpu)
```
<details> 
<summary><b>Arguments Details</b></summary>
    
   |Arguments Details   |Type           |Default        |Note
   |---                 |---            |---            |---
   |`--weight`          |`string`       |`required`     |**Path to onnx engine.**
   |`--fp16`            |`store_true`   |`false`        |**Use FP16 fast mode (x2 inference time).**
   |`--maxbatchsize`    |`int`          |`1`            |**Inference max batchsize.**
   |`--maxworkspace`    |`int`          |`1300(MB)`     |**Max workspace size (MB).**
   |`--tensor`          |`string_array` |`None`         |**Input tensor(s) for dynamic shape input *(dynamic shape input only)*.**
   |`--gpu`             |`int`          |`0`            |**Primary gpu index.**

   **Note:** The only GPUs with full-rate FP16 Fast mode performance are Tesla P100, Quadro GP100, and Jetson TX1/TX2.
    
</details> 

<details> 
<summary><b>Examples</b></summary>
 
- **Export Onnx engine to TensorRT engine.**
 
  ```sh
  ./main export --weight classifier.trt
  ./main export --weight classifier.trt --maxbatchsize 3 --maxworkspace 1500
  ./main export --weight classifier.trt --fp16 --gpu 2 --maxbatchsize 6 
  ```
 
- **Export Onnx engine with Dynamic shape input (batchsize x 3 x 416 x416).**
 
  ```sh
   --tensor tensorName,dims1(,dims2,dims3)  (Does not include batchsize dims)
   ./main export --weight classifier.trt --tensor input,3,416,416 --maxbatchize 7
   ./main export --weight classifier.trt --tensor input.1,3,416,416 input.2,12 input.3,7,4
   ```
 
</details>

## <div align=center> III. TensorRT engine Inference. </div>
```sh
./main infer --weight --data (--batchsize) (--softmax) (--gpu)
```
<details> 
<summary><b>Arguments Details</b></summary>
    
|Arguments      |Type           |Default    |Note
|---            |---            |---        |---
|`--weight`     |`string`       |`required` |**Path to tensorrt engine.**
|`--data`       |`string`       |`required` |**Path to inference image/video/images's folder.**
| `--batchsize` |`int`          |`1`        |**Inference batchsize.**
| `--softmax`   |`store_true`   |`false`    |**Add softmax to last layer of engine.**
| `--gpu`       |`int`          |`0`        |**Primary gpu index.**
 
</details> 
    
<details> 
<summary><b>Examples</b></summary>
 
- **TensorRT engine Inference.**
 
  ```sh
  ./main infer --weight classifier.trt --data image.jpg --softmax
  ./main infer --weight classifier.trt --data ./images/ --batchsize 4
  ./main infer --weight classifier.trt --data video.mov --batchsize 3 --softmax
  ```
 
- **Multiple inputs engine inference**
 
  ```sh
    Edit 'Inference' function (Class TRTParser(TRTParser.h and TRTParser.cpp)):
       - Add 2nd input's data arguments.
       - Add AllocateImageInput or AllocateNonImageInput for buffer[1](input2) below 'AllocateImageInput' (buffer[0](input1)).
       - Remove 'nrofInputs > 1' condition
    Edit main.cpp 
       - Add 2nd input's data to engine.Inference(...) coresponding with above edition. 
   
   ./main infer --weight classifier.trt --data ./infer_images/ --batchsize 3 --softmax
   ```

</details>
 
## To-Do

- [x] Multiple inputs model.
- [x] Multiple outputs model.
- [x] Add Channel last image allocate.
- [x] Switch Primary GPU. 
- [x] Multi-type cast for arguments (Easy to maintain).
- [x] Non-image input model.
- [x] Add examples.
- [x] 2D,3D tensor sofmax execute.
- [x] Remove "Segmentation fault (core dumped)" at ending of inference.
- [x] CUDA allocate exception handle.
- [x] Support INT model.
- [ ] Add CudaStream.
