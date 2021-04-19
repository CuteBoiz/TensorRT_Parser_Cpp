#include <iostream>
#include <onnxparser.h>
#include <trtparser.h>
#include <chrono>
#include <dirent.h>

static inline int read_files_in_dir(const char *p_dir_name, std::vector<std::string> &file_names) {
    DIR *p_dir = opendir(p_dir_name);
    if (p_dir == nullptr) {
        return -1;
    }

    struct dirent* p_file = nullptr;
    while ((p_file = readdir(p_dir)) != nullptr) {
        if (strcmp(p_file->d_name, ".") != 0 &&
            strcmp(p_file->d_name, "..") != 0) {
            //std::string cur_file_name(p_dir_name);
            //cur_file_name += "/";
            //cur_file_name += p_file->d_name;
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
	[mode] 	-i :infer
			-e : export onnx to trt
	[model path]
	[images folder path]
	*/
	string model_path;
	string folder_path;

	if (argc == 3 && std::string(argv[1]) == "-e"){
		model_path = argv[2];
		std::ifstream f(model_path);
		if (!f.good()){
			cerr << model_path << " not found! \n";
			return -1;
		}
		else {
			cout << model_path << " Found!, Try To Exporting Model ... \n";
		}
		OnnxParser model(model_path, 1);
		if (model_path.substr(model_path.find_last_of(".") + 1) == "onnx"){
			if (model.export_trt()){
				cout << "Export to TensorRT Success! \n"; return 0;
			}
			else{
				cout << "Export Failed! \n"; return -1;
			}
		}
		else{
			cerr << "Cannot export model! The extension must be .onnx! \n";
			return -1;
		}
	} 
	else if (argc == 4 && std::string(argv[1]) == "-i"){
		model_path = argv[2];
		folder_path = argv[3];
		if (folder_path[folder_path.length() - 1] != '/' && folder_path[folder_path.length() -1] != '\\') {
			folder_path = folder_path + '/';
		}
		cv::Mat image;
		std::ifstream f(model_path);
		if (!f.good()){
			cerr << model_path << " not found! \n";
			return -1;
		}
		else {
			cout << model_path << " Found!, Parsing Model ... \n";
		}
		std::vector<std::string> file_names;
		if (read_files_in_dir(folder_path.c_str(), file_names) < 0) {
	        std::cout << "read_files_in_dir failed." << std::endl;
	        return -1;
	    }
		if (model_path.substr(model_path.find_last_of(".") + 1) == "onnx"){
			OnnxParser model(model_path, 1);
			for (int f = 0; f < (int)file_names.size(); f++){
				string file_extension = file_names[f].substr(file_names[f].find_last_of(".") + 1);
				if (file_extension == "bmp" || file_extension == "png" || file_extension == "jpeg"){
					auto start = std::chrono::system_clock::now();
					cout << file_names[f] << endl;
					image = cv::imread(folder_path + file_names[f]);
					model.inference(image);
					auto end = std::chrono::system_clock::now();
	        		std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;
				}
			}
		}
		else if (model_path.substr(model_path.find_last_of(".") + 1) == "trt"){
			TRTParser model(model_path, 1);
			for (int f = 0; f < (int)file_names.size(); f++){
				string file_extension = file_names[f].substr(file_names[f].find_last_of(".") + 1);
				if (file_extension == "bmp" || file_extension == "png" || file_extension == "jpeg"){
					auto start = std::chrono::system_clock::now();
					cout << file_names[f] << endl;
					image = cv::imread(folder_path + file_names[f]);
					model.inference(image);
					auto end = std::chrono::system_clock::now();
	        		std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;
				}
			}
		}
		else{
			cerr << "Undefined extension of " << model_path <<". Model path must be .onnx or .trt! \n";
			return -1;
		}
	}
	else{
		cerr << "Undefined arguments. [-e] [model_path] or [-i] [model_path] [images_path]. \n";
		return -1;
	}
	
	return 0;
}

