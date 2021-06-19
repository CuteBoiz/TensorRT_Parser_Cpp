#pragma once
#ifndef TRT_PARSER_H
#define TRT_PARSER_H

#include <iostream>
#include <string.h>
#include <fstream>
#include <ostream>
#include <vector>
#include <algorithm>
#include <typeinfo>
#include <memory>
#include <stdio.h>

#include <opencv2/cudawarping.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <NvInferRuntime.h>
#include <NvInfer.h>
#include <NvOnnxParser.h>
#include <cuda_runtime_api.h>

using namespace std;

#define MAX_WORKSPACE_SIZE (1 << 30)

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

class TRTParser {
private:
	string enginePath;
	nvinfer1::ICudaEngine* engine;
	nvinfer1::IExecutionContext* context;

	nvinfer1::ICudaEngine* getTRTEngine();
	size_t getSizeByDim(const nvinfer1::Dims& dims);
	void preprocessImage(vector<cv::Mat> image, float* gpu_input, const nvinfer1::Dims& dims);
	void postprocessResult(float *gpu_output, int size, const nvinfer1::Dims &dims, bool softMax);
public:
	TRTParser();
	bool init(string enginePath);
	~TRTParser();

	void inference(vector<cv::Mat> image, bool softMax=false);
};

nvinfer1::ICudaEngine* getOnnxEngine(string onnxPath, unsigned max_batchsize, bool fp16, string input_tensor_name="", vector<unsigned> dimension={}, bool dynamic_shape=false);
bool exportTRTEngine(string onnxEnginePath, unsigned max_batchsize, bool fp16, string input_tensor_name="", vector<unsigned> dimension={}, bool dynamic_shape=false);

#endif //TRT_PARSER_H