/*
Routines support for tensorrt engine

Author: phatnt
Date: 2023-May-08
*/
#pragma once
#ifndef _LOGGER_HPP_
#define _LOGGER_HPP_

#include <iostream>
#include <memory>
#include <NvInfer.h>

/**
 * Logger for Tensorrt engine.
 * 
 */
static class Logger : public nvinfer1::ILogger{
public:
    void log(Severity severity, const char* msg) noexcept override{
        if ((severity == Severity::kERROR) || (severity == Severity::kINTERNAL_ERROR)) {
            std::cout <<"\033[1;31m[ERROR]: " << msg << "\033[0m\n";
        }
    }
} gLogger;

/**
 * Destroy object if error raised. 
 * 
 */
struct TRTDestroy {
    template<class T> 
    void operator()(T* obj) const {
        delete obj;
    }
};

template< class T >
using TRTUniquePtr = std::unique_ptr< T, TRTDestroy >;

#endif //_LOGGER_H_