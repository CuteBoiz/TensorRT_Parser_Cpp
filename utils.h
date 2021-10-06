/*
Ultitities for convert and infer tensorrt engine.

author: phatnt.
modified date: 2021-10-06
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
using namespace std;


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


struct ExportConfig{
    string onnxEnginePath;
    unsigned maxBatchsize;
    size_t maxWorkspaceSize;
    bool useFP16;
    bool useDynamicShape;
    string inputTensorName;
    vector<unsigned> tensorDims;

    ExportConfig(){
        onnxEnginePath = "";
        maxBatchsize = 0;
        maxWorkspaceSize = 0;
        useFP16 = false;
        useDynamicShape = false;
        inputTensorName = "";
        tensorDims = {};
    }

    bool Update(string enginePath, unsigned i_maxbatchsize, size_t workspaceSize, bool fp16){
        onnxEnginePath = enginePath;
        maxBatchsize = i_maxbatchsize;
        maxWorkspaceSize = workspaceSize;
        useFP16 = fp16;
        useDynamicShape = false;
        inputTensorName = "";
        tensorDims = {};
        if (maxBatchsize <= 0){
            cerr <<"[ERROR] Max batchsize must be more than 0! \n";
            return false;
        }
        if (!CheckFileIfExist(onnxEnginePath)){
            cerr <<"[ERROR] '"<< onnxEnginePath << "' not found! \n";
            return false;
        }
        return true;
    }
    bool Update(string enginePath, unsigned i_maxbatchsize, size_t workspaceSize, bool fp16, string tensorName, vector<unsigned> dims){
        onnxEnginePath = enginePath;
        maxBatchsize = i_maxbatchsize;
        maxWorkspaceSize = workspaceSize;
        useFP16 = fp16;
        useDynamicShape = true;
        inputTensorName = tensorName;
        for (unsigned i = 0; i < dims.size(); i++){
            tensorDims.emplace_back(dims.at(i));
        }
        if (maxBatchsize <= 0){
            cerr <<"[ERROR] Max batchsize must be more than 0! \n";
            return false;
        }
        if (!CheckFileIfExist(onnxEnginePath)){
            cerr <<"[ERROR] '"<< onnxEnginePath << "' not found! \n";
            return false;
        }
        if (inputTensorName == ""){
            cerr << "[ERROR] Input tensor name is empty! \n";
            return false;
        }
        if (tensorDims.size() != 3){
            cerr << "[ERROR] Dimension of dynamic shape must be 3! \n";
            return false;
        }
        return true;
    }
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


bool CheckRequiredArguments(const vector<string> required_args, const vector<string> args);
/*
Check required arguments.
Args:
    required_args:  required arguments.
    arguments:      input arguments.
Return:
    <bool>: valid check
 */

bool CheckValidValue(string value, string type);
/*
Check valid arguments's value.
Args:
    value:  arguments's value.("file"/"folder"/"unsigned"/"double").
    type:   arguments's value type.
Return:
    <bool>: valid check.
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


bool ExportOnnx2Trt(const ExportConfig config);

/*
Export onnx engine to tensorrt engine .
Args:
    config: config for onnx parse.
Return:
    <bool> Success checking.
 */

#endif
