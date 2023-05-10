/*
Main rountine to use TensorRT Parser

Author: phatnt
Date: 2023-May-08
*/

#include <iostream>
#include <chrono>
#include <string.h>

#include "utils.h"
#include "data.h"
#include "onnx_parser.h"
#include "tensorrt_parser.h"

int main(int argc, char* argv[]){
    std::string mode = std::string(argv[1]);
    if (mode == "convert" && argc == 3){
        // Convert Mode
        std::cout << "[INFO] Converting onnx engine \n";
        std::string configPath = std::string(argv[2]);
        if (!utils::checkIsFileAndExist(configPath)){
            return -1;
        }
        if (!onnx::convertEngine(configPath)){
            return -1;
        }
    }
    else if (mode == "infer" && (argc >= 4 && argc <= 6)){
        // Inference Mode
        std::string enginePath = std::string(argv[2]);
        std::string dataPath = std::string(argv[3]);
        std::string useSoftmax = "";
        unsigned gpuID = 0;
        bool isSoftmax = false;
        if (argc == 6){
            useSoftmax = std::string(argv[4]);
            gpuID = std::stoi(argv[5]);
        }
        if(argc == 5){
            try{
                gpuID = std::stoi(argv[4]);
            }
            catch (std::exception &e){
                useSoftmax = std::string(argv[4]);
            }
        }
        if (useSoftmax == "softmax"){
            isSoftmax = true;
        }
        if (gpuID){
            if (!utils::setCudaNum(gpuID)){
                std::cout << "\033[1;33m[WARNING] Could not set primary gpu to '"<< gpuID << "'! Set to default gpu 0.\033[0m\n";
            }
        }
        if (!utils::checkIsFileAndExist(enginePath)){
            return -1;
        }
        if (!utils::checkPathAttribute(dataPath)){
            std::cerr << "\033[1;31m[ERROR] '"<< dataPath <<"' not exist!\033[0m\n";
            return -1;
        }
        // Model
        TensorRT model;
        if (!model.init(enginePath)){
            std::cerr << "\033[1;31m[ERROR] Could not initialize tensorrt engine!\033[0m\n";
            return -1;
        }
        // Inference
        data::InputData data = data::prepareData(dataPath);
        std::vector<tensor::TensorValue> preds;
        try {
            auto start = std::chrono::system_clock::now();
            preds = model.inference(data.images);
            auto end = std::chrono::system_clock::now();
            // Result
            std::cout << "[INFO] Total inferenced images: " << data.imagePaths.size() << " in ";
            std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms. \n";
            if (preds.size() % data.imagePaths.size() != 0){
                std::cerr << "\033[1;31m[ERROR] Unmatched quantities of output and inputs '" << preds.size() << ":" << data.imagePaths.size() << "' !\033[0m\n";
                return -1;
            }
            std::cout << "[INFO] Output: \n";
            unsigned ratio = preds.size() / data.imagePaths.size();
            for (unsigned i = 0; i < data.imagePaths.size(); i++){
                std::cout << "\033[1;32m" << data.imagePaths[i] << "'s result:\033[0m\n";
                for (unsigned j = 0; j < ratio; j++){
                    unsigned id = j*data.imagePaths.size() + i;
                    if (isSoftmax){
                        softmax(preds[id]);
                    } 
                    std::cout << preds[id];
                }
            }
        }
        catch (std::exception& e){
            std::cerr << "\033[1;31m[ERROR] " << e.what() << "\033[0m\n";
            return -1;
        }
    }
    else {
        std::cerr << "\033[1;31m[ERROR] Invalid argument! \033[0m\n";
    }
    return 0;
}