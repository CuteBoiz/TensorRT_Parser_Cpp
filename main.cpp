#include <iostream>
#include <dirent.h>
#include "TRTParser.h"

#define BATCH_SIZE 1

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
	*/
	string enginePath;
	string folderPath;

	if (argc == 3 && std::string(argv[1]) == "-e"){
		enginePath = argv[2];
		std::ifstream f(enginePath);
		if (!f.good()){
			cerr << enginePath << " not found! \n";
			return -1;
		}
		else {
			cout << enginePath << " Found!, Try To Exporting Model ... \n";
		}
		if (enginePath.substr(enginePath.find_last_of(".") + 1) == "onnx"){
			if (saveTRTEngine(enginePath)){
				cout << "Export to TensorRT Success! \n"; return 0;
			}
			else{
				cout << "Export Failed! \n"; return -1;
			}
		}
	} 
	else if (argc == 4 && std::string(argv[1]) == "-i"){
		enginePath = argv[2];
		folderPath = argv[3];
		if (folderPath[folderPath.length() - 1] != '/' && folderPath[folderPath.length() -1] != '\\') {
			folderPath = folderPath + '/';
		}
		cv::Mat image;
		std::ifstream f(enginePath);
		if (!f.good()){
			cerr << enginePath << " not found! \n";
			return -1;
		}
		else {
			cout << enginePath << " Found!, Parsing Model ... \n";
		}
		std::vector<std::string> file_names;
		if (read_files_in_dir(folderPath.c_str(), file_names) < 0) {
	        std::cout << "read_files_in_dir failed." << std::endl;
	        return -1;
	    }

	    string model_extention = enginePath.substr(enginePath.find_last_of(".") + 1);
		if (model_extention == "trt"){
			TRTParser model.init(enginePath, BATCH_SIZE);
			
		}
		else{
			cerr << "Undefined extension of " << enginePath <<". Model path must be .trt! \n";
			return -1;
		}
	}
	else{
		cerr << "Undefined arguments. \n [-e] [enginePath] to export trt model \n [-i] [enginePath] [imagesFolderPath]. to infer trt model \n";
		return -1;
	}
	
	return 0;
}

