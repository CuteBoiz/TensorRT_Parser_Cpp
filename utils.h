/*
Ultitities for convert and infer tensorrt eninge.

author: phatnt.
modified date: 2021-09-29
 */

#ifndef _UTILS_HPP_
#define _UTILS_HPP_

#include <NvInfer.h>
#include <NvOnnxParser.h>
#include <cuda_runtime_api.h>

#include <memory>
#include <fstream>
#include <ostream>
#include <string.h>
#include <iostream>
#include <dirent.h>
using namespace std;

#define MAX_WORKSPACE_SIZE (1e6 * 1300) //1000 Mb

struct TRTDestroy{
    template< class T >
    void operator()(T* obj) const{
        if (obj){
            obj->destroy();
        }
    }
};
template< class T >
using TRTUniquePtr = unique_ptr< T, TRTDestroy >;

static class Logger : public nvinfer1::ILogger{
public:
    void log(Severity severity, const char* msg) noexcept override{
        if ((severity == Severity::kERROR) || (severity == Severity::kINTERNAL_ERROR)) {
            cout << msg << endl;
        }
    }
} gLogger;


bool CheckFileIfExist(const string filePath);
/*
Check existance of a file.
Args:
    filePath :path to file.
Return:
    <bool> exist-true / not exist-false
 */


bool ReadFilesInDir(const char *p_dir_name, vector<string> &file_names);
/*
Read all file in a folder.
Args:
    p_dir_name : path to folder.
    file_names : returned file's name array.
Return:
    <bool> status checking.
 */



nvinfer1::ICudaEngine* LoadOnnxEngine(const string onnxEnginePath, const unsigned max_batchsize, 
                                             const bool fp16=false,
                                             const string input_tensor_name = "", 
                                             const vector<unsigned> dimension = {}, 
                                             const bool dynamic_shape = false);
/*
Load onnx engine as an ICudaEngine.
Args:
    onnxEnginePath:     path to onnx engine.
    max_batchsize:      max inference batchsize.
    fp16:               export model with fp16 fast mode.
    input_tensor_name:  input tensor's name (dynamic_shape convert only).
    dimension:          dimension of input tensor (dynamic_shape convert only).
    dynamic_shape:      convert model with dynamic shape
 */


bool ExportOnnx2Trt(const string onnxEnginePath, const unsigned max_batchsize, 
                            bool fp16 = false,
                            const string input_tensor_name = "",
                            const vector<unsigned> dimension = {},
                            const bool dynamic_shape = false);

/*
Export onnx engine to tensorrt engine .
Args:
    onnxPath:           path to onnx engine.
    max_batchsize:      max inference batchsize.
    fp16:               export model with fp16 fast mode.
    input_tensor_name:  input tensor's name (dynamic_shape convert only).
    dimension:          dimension of input tensor (dynamic_shape convert only).
    dynamic_shape:      convert model with dynamic shape
 */


#endif
