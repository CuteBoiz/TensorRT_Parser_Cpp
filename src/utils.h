/*
Ultitities

author: phatnt.
modified date: 2023-May-04
 */
#pragma once
#ifndef _UTILS_H_
#define _UTILS_H_

#include <iostream>
#include <fstream>
#include <errno.h>
#include <memory>
#include <vector>
#include <sys/stat.h>
#include <cuda.h>
#include <cuda_runtime_api.h>


namespace utils{
    /**
     * @brief Check a path atrribute
     * 
     * @param path
     * @return 0: not exist
     * @return 1: is a file
     * @return 2: is a folder
     */
    int checkPathAttribute(const std::string path);


    /**
     * @brief Check a path is file and exist
     * 
     * @param filePath 
     * @return true: file is exist
     * @return false: is a folder or not exist
     */
    bool checkIsFileAndExist(const std::string filePath);


    /**
     * @brief Check a path is folder and exist
     * 
     * @param folderPath 
     * @return true: folder is exist
     * @return false: is a file or not exist
     */
    bool checkIsFolderAndExist(const std::string folderPath);


    /**
     * @brief Copy a binary file 
     * 
     * @param src: source file
     * @param dst: destination
     * @return true: copy success
     *         false: copy fail
     */
    bool copyBinaryFile(const std::string src, const std::string dst);


    /**
     * @brief Cuda function check
     * 
     * @param status 
     * @return true: success
     * @return false: fail
     */
    bool cudaCheck(const cudaError_t status);


    /**
     * @brief Get the Shape Size
     * 
     * @param shape 
     * @return unsigned 
     */
    unsigned getShapeSize(const std::vector<unsigned> shape);
    

    /**
     * @brief Set the Cuda number (If own multi-gpu system)
     * 
     * @param gpuNum 
     * @return true: set success
     * @return false: set fail
     */
    bool setCudaNum(const unsigned gpuNum);


}


#endif //_UTILS_H_
