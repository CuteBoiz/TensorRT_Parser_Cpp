/*
TensorRT Parser Class.

author: phatnt.
modified date: 2021-10-11

 */
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
	unsigned maxBatchSize;
	size_t engineSize;
	vector< Tensor > inputTensors;
	vector< Tensor > outputTensors;
	
	nvinfer1::ICudaEngine* engine;
	nvinfer1::IExecutionContext* context;

	nvinfer1::ICudaEngine* LoadTRTEngine(const string enginePath);
	size_t GetDimensionSize(const nvinfer1::Dims& dims);

	bool AllocateNonImageInput(void *pData, float* gpuInputBuffer, const unsigned inputIndex);
	bool AllocateImageInput(vector<cv::Mat> images, float* gpuInputBuffer, const unsigned inputIndex);
	vector<float> PostprocessResult(float *gpuOutputBuffer, const unsigned batch_size, const unsigned outputIndex, const bool softMax);

public:
	TRTParser();
	~TRTParser();
	bool Init(const string enginePath);
	/*
	Create tensorrt engine.
	Args:
		enginePath: path to tensorrt engine.
	Return:
		<bool> Success checking.
	 */
	
	bool Inference(vector<cv::Mat> images, const bool softMax);
	/*
	TensorRT inference.
	Args:
		images: infer images array.
		softMax: add softmax to last layer of model.
	Return:
		<bool> Success checking.
	 */
};

#endif //TRT_PARSER_H
