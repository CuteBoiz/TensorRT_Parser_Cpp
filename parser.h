#ifndef ONNX_PARSER_H
#define ONNX_PARSER_H

#include <iostream>
#include <string.h>
#include <fstream>
#include <ostream>
#include <vector>
#include <algorithm>
#include <typeinfo>
#include <memory>

#include <opencv2/cudawarping.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <NvInferRuntime.h>
#include <NvInfer.h>
#include <NvOnnxParser.h>
#include <cuda_runtime_api.h>

using namespace std;

#define MAX_WORKSPACE (1 << 30)

#ifndef TRTDEST
#define TRTDEST
struct TRTDestroy
{
    template< class T >
    void operator()(T* obj) const
    {
        if (obj)
        {
            obj->destroy();
        }
    }
};

template< class T >
using TRTUniquePtr = std::unique_ptr< T, TRTDestroy >;

#endif //TRTDEST

class OnnxParser{

private:
	string model_path;
	int batch_size;
	nvinfer1::ICudaEngine* engine;
	nvinfer1::IExecutionContext* context;

	size_t getSizeByDim(const nvinfer1::Dims& dims);
	void preprocessImage(cv::Mat image, float* gpu_input, const nvinfer1::Dims& dims);
	void postprocessResults(float *gpu_output, const nvinfer1::Dims &dims);
public:
	OnnxParser(string model_path, int batch_sz);
	~OnnxParser();
	
	void inference(cv::Mat image);
	bool export_trt();
	
};

#endif