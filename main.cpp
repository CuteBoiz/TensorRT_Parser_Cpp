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
				[max batchsize]
				[fp16]

			-ed: export dynamic shape
				[model path]
				[max batchsize]
				[input tensor name]
				[dimension 1]
				[dimension 2]
				[dimension 3]
				[fp16]
				
			-i : infer onnx or trt model.
				[model path]
				[images folder path]
				[batch size]

	*/
	static string enginePath;
	static unsigned max_batchsize = 0;
	static string folderPath;
	static unsigned modelImageSize = 0;
	static unsigned batchSize = 0;
	static bool fp16 = false;

	if ((argc == 4 || argc == 5) && std::string(argv[1]) == "-e"){
		enginePath = string(argv[2]);
		max_batchsize = stoi(argv[3]);

		if (max_batchsize <= 0){
			cerr <<"ERROR: "<< "max batchsize must be more than 0! \n";
			return -1;
		}
		std::ifstream f(enginePath);
		if (!f.good()){
			cerr <<"ERROR: "<< enginePath << " not found! \n";
			f.close();
			return -1;
		}
		f.close();
		if (argc == 5 && argv[5] == "fp16"){
			fp16 = true;
		}
		if (enginePath.substr(enginePath.find_last_of(".") + 1) == "onnx"){
			if (exportTRTEngine(enginePath, max_batchsize, fp16)){
				cout << "Export to TensorRT Success! \n"; 
				return 0;
			}
			else{
				cerr << "ERROR: Export Failed! \n"; 
				return -1;
			}
		}
	}
	else if ((argc == 8 || argc == 9) && std::string(argv[1]) == "-ed"){
		enginePath = string(argv[2]);
		max_batchsize = stoi(argv[3]);
		string input_tensor_name = string(argv[4]);
		vector<unsigned> dimension;

		for (unsigned i = 5; i < 8; i++){
			if (stoi(argv[i]) > 0){
				dimension.emplace_back(stoi(argv[i]));
			}
			else{
				cerr << "ERROR: Dimension must be more than 0 \n";
				return -1;
			}
		}
		if (max_batchsize <= 0){
			cerr <<"ERROR: "<< "max batchsize must be more than 0! \n";
			return -1;
		}
		std::ifstream f(enginePath);
		if (!f.good()){
			cerr <<"ERROR: "<< enginePath << " not found! \n";
			f.close();
			return -1;
		}
		f.close();
		if (argc == 9 && (string(argv[8]) == "fp16" || string(argv[8]) == "FP16")){
			fp16 = true;
		}
		if (enginePath.substr(enginePath.find_last_of(".") + 1) == "onnx"){
			if (exportTRTEngine(enginePath, max_batchsize, fp16, input_tensor_name, dimension, true)){
				cout << "Export to TensorRT Success! \n";
				dimension.clear();
				return 0;
			}
			else{
				cerr << "ERROR: Export Failed! \n";
				dimension.clear();
				return -1;
			}
		}
	} 
	else if (argc == 5 && std::string(argv[1]) == "-i"){
		enginePath = string(argv[2]);
		folderPath = string(argv[3]);
		batchSize = stoi(argv[4]);

		if (batchSize <= 0){
			cerr << "ERROR: batch size must larger than 0 \n";
			return -1;
		}
		if (folderPath[folderPath.length() - 1] != '/' && folderPath[folderPath.length() -1] != '\\') {
			folderPath = folderPath + '/';
		}
		cv::Mat image;
		std::ifstream f(enginePath);
		std::vector<std::string> fileNames;
		std::vector<cv::Mat> images;
		TRTParser engine;
		unsigned nrof_iamges = 0;
		unsigned i = 0;

		if (!f.good()){
			cerr <<"ERROR: " <<enginePath << " not found! \n";
			f.close();
			return -1;
		}
		else {
			f.close();
			cout << enginePath << " Found!, Parsing Model ... \n";
		}
		if (read_files_in_dir(folderPath.c_str(), fileNames) < 0) {
	        std::cout << "read_files_in_dir failed." << std::endl;
	        return -1;
	    }		
		if (!engine.init(enginePath)){
			cerr << "ERROR: Could not parse tensorRT engine! \n";
			return -1;
		}
		
		for (unsigned f = 0; f < (unsigned)fileNames.size(); f += i) {
			unsigned index = 0;
			for (i = 0; index < batchSize && (f + i) < (unsigned)fileNames.size(); i++) {
				std::string fileExtension = fileNames[f + i].substr(fileNames[f + i].find_last_of(".") + 1);
				if (fileExtension == "bmp" || fileExtension == "png" || fileExtension == "jpeg" || fileExtension == "jpg") {
					cout << fileNames[f + i] << endl;
					cv::Mat image = cv::imread(folderPath + fileNames[f + i]);
					images.emplace_back(image);
					index++;
					nrof_iamges++;
				}
				else{
					//cout << fileNames[f + i] << " is not a image! \n";
				}
			}
			if (images.size() == 0) {
				continue;
			}
			auto start = chrono::system_clock::now();
			engine.inference(images, true);
			auto end = chrono::system_clock::now();
			cout << chrono::duration_cast<chrono::milliseconds>(end - start).count() << "ms. \n";
			
			for (unsigned j = 0; j < images.size(); j++){
				images.at(j).release();
			}
			
			images.clear();
		}
		cout << "Total inferenced image number: " << nrof_iamges << endl;
		fileNames.clear();
	}
	else{
		cout << "Undefined arguments. \n";
		cout <<	"[-e] [enginePath] [max batchsize] ([fp16]) to export trt model \n ";
		cout << "[-ed][enginePath] [max batchsize] [input tensor name] [d1] [d2] [d3] ([fp16]) to export dynamic shape trt model \n";
		cout << "[-i] [enginePath] [imagesFolderPath] [batchSize] to infer trt model \n";
		return -1;
	}
	
	return 0;
}

