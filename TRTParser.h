#pragma once
#ifndef TRT_PARSER_H
#define TRT_PARSER_H

#include <iostream>

#include <opencv2/cudawarping.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "utils.h"
using namespace std;

class TRTParser {
private:
	nvinfer1::ICudaEngine* engine;
	nvinfer1::IExecutionContext* context;

	nvinfer1::ICudaEngine* loadTRTEngine(const string enginePath);
	size_t getSizeByDim(const nvinfer1::Dims& dims);
	void preprocessImage(vector<cv::Mat> image, float* gpu_input, const nvinfer1::Dims& dims);
	vector<float> postprocessResult(float *gpu_output, const unsigned batch_size, const unsigned output_size, const bool softMax);
public:
	TRTParser();
	bool init(const string enginePath);
	~TRTParser();
	bool inference(vector<cv::Mat> image, const bool softMax);
};

#endif //TRT_PARSER_H
