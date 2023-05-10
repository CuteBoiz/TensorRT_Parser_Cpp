/*
Routine to process input data

Author: phatnt
Date: 2023-May-08

*/

#pragma once
#ifndef _DATA_H_
#define _DATA_H_

#include <iostream>
#include "utils.h"
#include <opencv2/highgui/highgui.hpp>
#include <dirent.h>

namespace data{

    static const std::vector<std::string> imageExts{"bmp", "png", "jpeg", "jpg", "svg", "webp"};
    static const std::vector<std::string> videoExts{"mp4", "mov", "avi", "wmv", "flv", "mkv", "webm"};

    /**
     * @brief Struct to get imagePath and images to inference 
     * 
     */
    struct InputData{
        std::vector<std::string> imagePaths;
        std::vector<cv::Mat> images; 
    };

    /**
     * @brief read all files inside a dataPath
     * 
     * @param p_dir_name 
     * @param file_names 
     * @return true: success
     * @return false: fail
     */
    bool readFilesInDir(const char *dataPath, std::vector<std::string> &file_names);


    /**
     * @brief Get images from dataPath
     * 
     * @param dataPath 
     * @return InputData list of images and imagePaths
     */
    InputData prepareData(std::string dataPath);

}

#endif //_DATA_H_