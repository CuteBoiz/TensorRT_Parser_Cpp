/*Struct Tensor for tensorrt

author: phatnt
date: 2023-Apr-14
*/
#pragma once
#ifndef _TENSOR_H_
#define _TENSOR_H_

#include <iostream>
#include <vector>
#include <algorithm>
#include <cmath>
#include <NvInfer.h>
#include "utils.h"

#define MAXSIZE 2000 // Max tensor size that can be print 

namespace tensor{

	/**
	 * @brief Struct for ICudaEngine's binding attribute
	 * 
	 */
	struct TensorAttribute{
		std::string name;
		std::string type;
		bool isInput;
		bool isCHW;
		bool isImage;
		unsigned size;
		std::vector<unsigned>shape;

		TensorAttribute(const nvinfer1::ICudaEngine* engine, const unsigned bindingIndex);
	};
	
	/**
	 * @brief Struct to get value and shape return from inference
	 * 
	 */
	struct TensorValue{
		std::string name;
		std::vector<unsigned>shape;
		std::vector<float> value;
		
		TensorValue(const std::string name, const std::vector<unsigned> shape, const std::vector<float> value);
	};

	/**
	 * @brief print operator for TensorAttribute
	 * 
	 * @param os 
	 * @param x 
	 * @return std::ostream& 
	 */
	std::ostream& operator << (std::ostream& os, const TensorAttribute& x);

	/**
	 * @brief print operator for TensorValue 
	 * 
	 * @param os 
	 * @param x 
	 * @return std::ostream& 
	 */
	std::ostream& operator << (std::ostream& os, const TensorValue& x);

	/**
	 * @brief Caculate softmax for a tensor
	 * 
	 * @param x 
	 */
	void softmax(TensorValue& x);
}

/**
 * @brief print operator for TensorRT engine
 * 
 * @param os 
 * @param x 
 * @return std::ostream& 
 */
std::ostream& operator << (std::ostream& os, const nvinfer1::ICudaEngine* x);


#endif //_TENSOR_H_