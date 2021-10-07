/*
TensorRT Parser Class.

author: phatnt.
modified date: 2021-09-29

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
	unsigned imgH, imgW, imgC, maxBatchSize;
	bool isCHW;
	size_t engineSize;
	vector< nvinfer1::Dims > inputDims;
	vector< nvinfer1::Dims > outputDims;
	
	nvinfer1::ICudaEngine* engine;
	nvinfer1::IExecutionContext* context;

	nvinfer1::ICudaEngine* LoadTRTEngine(const string enginePath);
	size_t GetSizeByDim(const nvinfer1::Dims& dims);
	void PreprocessImage(vector<cv::Mat> images, float* gpu_input);
	vector<float> PostprocessResult(float *gpu_output, const unsigned batch_size, const unsigned output_size, const bool softMax);
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
