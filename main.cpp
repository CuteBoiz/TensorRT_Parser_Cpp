#include <iostream>
#include <dirent.h>
#include <chrono>
#include "TRTParser.h"

static inline int read_files_in_dir(const char *p_dir_name, std::vector<std::string> &file_names) {
    DIR *p_dir = opendir(p_dir_name);
    if (p_dir == nullptr) {
        return -1;
    }

    struct dirent* p_file = nullptr;
    while ((p_file = readdir(p_dir)) != nullptr) {
        if (strcmp(p_file->d_name, ".") != 0 &&
            strcmp(p_file->d_name, "..") != 0) {
            std::string cur_file_name(p_file->d_name);
            file_names.push_back(cur_file_name);
        }
    }
    closedir(p_dir);
    return 0;
}

int main(int argc,char** argv){
	/*
	arguments:
	[mode] 	-e : export onnx to trt
				[model path]
			-i : infer onnx or trt model.
				[model path]
				[images folder path]
				[model image size]
				[batch size]
	*/
	static string enginePath;
	static string folderPath;
	static int modelImageSize = 0;
	static int batchSize = 0;


	if (argc == 3 && std::string(argv[1]) == "-e"){
		enginePath = argv[2];
		std::ifstream f(enginePath);
		if (!f.good()){
			cerr <<"ERROR: "<< enginePath << " not found! \n";
			return -1;
		}
		else {
			cout << enginePath << " Found!, Try To Exporting Model ... \n";
		}
		if (enginePath.substr(enginePath.find_last_of(".") + 1) == "onnx"){
			if (saveTRTEngine(enginePath)){
				cout << "Export to TensorRT Success! \n"; 
				return 0;
			}
			else{
				cerr << "ERROR: Export Failed! \n"; 
				return -1;
			}
		}
	} 
	else if (argc == 6 && std::string(argv[1]) == "-i"){
		enginePath = argv[2];
		folderPath = argv[3];
		modelImageSize = atoi(argv[4]);
		batchSize = atoi(argv[5]);
		if (modelImageSize <= 0) {
			cerr << "ERROR: model image size must larger than 0 \n";
			return -1;
		}
		if (batchSize <= 0){
			cerr << "ERROR: batch size must larger than 0 \n";
			return -1;
		}
		if (folderPath[folderPath.length() - 1] != '/' && folderPath[folderPath.length() -1] != '\\') {
			folderPath = folderPath + '/';
		}
		cv::Mat image;
		std::ifstream f(enginePath);
		if (!f.good()){
			cerr <<"ERROR: " <<enginePath << " not found! \n";
			return -1;
		}
		else {
			cout << enginePath << " Found!, Parsing Model ... \n";
		}
		std::vector<std::string> fileNames;
		if (read_files_in_dir(folderPath.c_str(), fileNames) < 0) {
	        std::cout << "read_files_in_dir failed." << std::endl;
	        return -1;
	    }

	    string model_extention = enginePath.substr(enginePath.find_last_of(".") + 1);
		if (model_extention == "trt"){
			TRTParser engine;
			engine.init(enginePath);
			int i = 0;
			std::vector<cv::Mat> images;
			for (int f = 0; f < (int)fileNames.size(); f += i) {
				uint32_t index = 0;
				for (i = 0; index < batchSize && (f + i) < (int)fileNames.size(); i++) {
					std::string fileExtension = fileNames[f + i].substr(fileNames[f + i].find_last_of(".") + 1);
					if (fileExtension == "bmp" || fileExtension == "png" || fileExtension == "jpeg" || fileExtension == "jpg") {
						cout << fileNames[f + i] << endl;
						cv::Mat image = cv::imread(folderPath + fileNames[f + i]);
						cv::Size dim = cv::Size(modelImageSize, modelImageSize);
						cv::resize(image, image, dim, cv::INTER_AREA);
						images.emplace_back(image);
						index++;
					}
				}
				if (images.size() == 0) {
					continue;
				}
				auto start = chrono::system_clock::now();
				engine.inference(images);
				auto end = chrono::system_clock::now();
				cout << chrono::duration_cast<chrono::milliseconds>(end - start).count() << "ms. \n";

				images.clear();
			}
		}
		else{
			cerr << "Undefined extension of " << enginePath <<". Model path must be .trt! \n";
			return -1;
		}
	}
	else{
		cerr << "Undefined arguments. \n [-e] [enginePath] to export trt model \n [-i] [enginePath] [imagesFolderPath] [modelImageSize] [batchSize]. to infer trt model \n";
		return -1;
	}
	
	return 0;
}

