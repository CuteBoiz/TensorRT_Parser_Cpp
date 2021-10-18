/*
Ultitities for convert and infer tensorrt engine.

author: phatnt.
modified date: 2021-10-15
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
#include <errno.h>

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace std;


struct Tensor{
    string tensorName;
    bool isCHW;
    unsigned tensorSize;
    nvinfer1::Dims dims;
    nvinfer1::DataType type;
    nvinfer1::TensorFormat format;

    Tensor();
    Tensor(nvinfer1::ICudaEngine* engine, const unsigned bindingIndex);
};

struct ExportConfig{
    string onnxEnginePath;
    unsigned maxBatchsize;
    size_t maxWorkspaceSize;
    bool useFP16;
    bool useDynamicShape;
    vector<string> tensorNames;
    vector<vector<unsigned>> tensorDims;

    ExportConfig();
    bool Update(const string onnxEnginePath, const unsigned maxBatchsize, const size_t maxWorkspaceSize, const bool fp16);
    bool Update(const string onnxEnginePath, const unsigned maxBatchsize, const size_t maxWorkspaceSize, const bool fp16, vector<string> tensorNames, vector<vector<unsigned>> tensorDims);
};



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


vector<string> splitString(string s, string delimiter);
/*
Split string
Args:
    s: a string
    delimiter: splited char
Return:
    vector<string>: splited string.
 */
 

inline bool CudaCheck(cudaError_t status);
/*
Check cudaStatus while using a Cuda command.
Args:
    status: cuda status when using a cuda command.
Return:
    <bool> success checking.
 */

static ostream& operator << (ostream& os, const Tensor& x);
/*
Print operator for struct Tensor.
 */

bool CheckFileIfExist(const string filePath);
/*
Check existance of a file.
Args:
    filePath: path to file.
Return:
    <bool> exist-true / not exist-false
 */

bool CheckFolderIfExist(const string folderPath);
/*
Check existance of a folder.
Args:
    filePath: path to folder.
Return:
    <bool> exist-true / not exist-false
 */

bool ReadFilesInDir(const char *p_dir_name, vector<string> &file_names);
/*
Read all file in a folder.
Args:
    p_dir_name: path to folder.
    file_names: returned file's name array.
Return:
    <bool> status checking.
 */

bool SetPrimaryCudaDevice(const unsigned gpuNum);
/*
Set primary gpu for export or infer.
Args:
    gpuNum: device number.
Return:
    <bool> Succsess checking.
 */

string GetArgumentsValue(const int argc, char** argv, unsigned& argsIndex, const string type);
/*
Get Arguments value for argc and argv.
Args:
    argc:       number of input args.
    argv:       value of input args.
    argsIndex:  index of arguments.
    type:       arguments's type.("string/file"/"folder"/"int"/"float/store_true").
Return:
    <string> string value with coresponding arguments.
 */

bool CheckRequiredArguments(const vector<string> required_args, const vector<string> args);
/*
Check required arguments.
Args:
    required_args:  required arguments.
    arguments:      input arguments.
Return:
    <bool>: valid check
 */

bool CheckValidArgument(const vector<string> required_args, const vector<string> valid_args, const string args);
/*
Check valid arguments.
Args:
    required_args:  required arguments.
    valid_args:     valid arguments.
    arguments:      input arguments.
Return:
    <bool>: valid check.
 */


nvinfer1::ICudaEngine* LoadOnnxEngine(const ExportConfig config);
/*
Load onnx engine as an ICudaEngine.
Args:
    config: config for onnx parse.
Return:
    <ICudaEngine> Executable Cuda engine. 
 */

bool ShowEngineInfo(nvinfer1::ICudaEngine* engine);
/*
Show TensorRT engine info.
Args:
    enigne: a Parsed TensorRT enigne.
Return:
    <bool> Success checking;
 */

bool ExportOnnx2Trt(const ExportConfig config);
/*
Export onnx engine to tensorrt engine .
Args:
    config: config for onnx parse.
Return:
    <bool> Success checking.
 */

vector< vector< cv::Mat >> PrepareImageBatch(string folderPath, const unsigned batchSize);
/*
Prepare batch for infernces.
Args:
    folderPath: path to inference images folder.
    batchSize:  inference batchsize.
Return:
    vector< vector< cv::Mat >> batched images.
 */

#endif
