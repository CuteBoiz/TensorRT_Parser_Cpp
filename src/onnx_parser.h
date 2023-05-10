/*
Routines for parse Onnx engine then convert to TensorRT engine

Author: phatnt
Date: 2023-May-09

*/

#pragma once
#ifndef _ONNX_PARSER_H_
#define _ONNX_PARSER_H_

#include <iostream>
#include <NvInfer.h>
#include <NvOnnxParser.h>
#include <yaml-cpp/yaml.h>
#include "utils.h"
#include "logger.hpp"

namespace onnx{

    /**
     * Convert onnx engine to tensorrt engine
     * 
     * @param configPath: Path to onnx config file
     * @return true: convert success
     * @return false: convert fail
     */
    bool convertEngine(const std::string configPath);

}

#endif //_ONNX_PARSER_H_