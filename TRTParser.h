/*
TensorRT Parser Class.

author: phatnt.
modified date: 2021-10-22

 */
#pragma once
#ifndef TRT_PARSER_H
#define TRT_PARSER_H

#include <iostream>
#include <dirent.h>

#include <opencv2/cudawarping.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "utils.h"
using namespace std;

struct InputData{
	cv::Mat image;
	string imagePath;

	InputData(cv::Mat image, string imagePath);
};

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

vector< vector< InputData >> PrepareImageBatch(string folderPath, const unsigned batchSize);
/*
Prepare batch for infernces.
Args:
    folderPath: path to inference images folder.
    batchSize:  inference batchsize.
Return:
    vector< vector< cv::Mat >> batched images.
 */

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
	
	bool Inference(vector<InputData> batchedInput, const bool softMax);
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
