/*
TensorRT Parser.

Author: phatnt.
Modified date: Apr-14-2023

 */
#pragma once
#ifndef TRT_PARSER_H
#define TRT_PARSER_H

#include <iostream>
#include <fstream>

#include <NvInfer.h>
#include <opencv2/cudawarping.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "tensor.h"
#include "utils.h"
#include "logger.hpp" 


/**
 * Class TensorRT: Parser Onnx or TensorRT weight + inference
 * 
 */
class TensorRT{
public:
    TensorRT();
    ~TensorRT();
    bool init(const std::string enginePath);
    std::vector<tensor::TensorValue> inference(std::vector<cv::Mat> input);
private:
    unsigned m_maxBatchSize;
    unsigned long m_engineSize;
    std::vector<tensor::TensorAttribute> m_inputTensors;
    std::vector<tensor::TensorAttribute> m_outputTensors;
    nvinfer1::ICudaEngine* m_engine;
    nvinfer1::IExecutionContext* m_context;
    std::vector< void*> m_buffers;

    std::vector<std::vector<cv::Mat>> batchSplit(std::vector<cv::Mat> images);
    bool allocateImage(std::vector<cv::Mat> images, float* buffer, const unsigned index);
    bool allocateNonImage(void *pData, float* buffer, const unsigned index);
    std::vector<float> postProcessing(float *buffer, const unsigned batchSize, const unsigned index);
};

#endif //TRT_PARSER_H
