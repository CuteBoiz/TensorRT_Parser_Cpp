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

#define MAX_WORKSPACE_SIZE (1 << 20)

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
    void log(Severity severity, const char* msg) override{
        if ((severity == Severity::kERROR) || (severity == Severity::kINTERNAL_ERROR)) {
            cout << msg << endl;
        }
    }
} gLogger;

bool checkFileIfExist(const string filePath);

bool readFilesInDir(const char *p_dir_name, vector<string> &file_names);

nvinfer1::ICudaEngine* loadOnnxEngine(const string onnxPath, const unsigned max_batchsize, 
                                             const bool fp16=false,
                                             const string input_tensor_name = "", 
                                             const vector<unsigned> dimension = {}, 
                                             const bool dynamic_shape = false);

bool convertOnnx2Trt(const string onnxEnginePath, const unsigned max_batchsize, 
                            bool fp16 = false,
                            const string input_tensor_name = "",
                            const vector<unsigned> dimension = {},
                            const bool dynamic_shape = false);

#endif