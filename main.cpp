#include <iostream>
#include <chrono>
#include "TRTParser.h"


static bool check(unsigned maxBatchSize, string enginePath);
static bool convert(int argc, char** argv);
static bool convertWithDynamicInputShape(int argc, char **argv);
static bool infer(int argc, char **argv);

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
	
	if ((argc == 4 || argc == 5) && string(argv[1]) == "-e"){
		if(convert(argc, argv)){
			return 0; 
		}
		return -1;
	}
	else if ((argc == 8 || argc == 9) && string(argv[1]) == "-ed"){
		if (convertWithDynamicInputShape(argc, argv)){
			return 0;
		}
		return -1;
		
	} 
	else if (argc == 5 && string(argv[1]) == "-i"){
		if (infer(argc, argv)){
			return 0;
		}
		return -1;
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


bool check(unsigned maxBatchSize, string enginePath){
	if (maxBatchSize <= 0){
		cerr <<"ERROR: "<< "max batchsize must be more than 0! \n";
		return false;
	}
	if (!checkFileIfExist(enginePath)){
		cerr <<"ERROR: "<< enginePath << " not found! \n";
		return false;
	}
	return true;
}

bool convert(int argc, char** argv){
	string enginePath = string(argv[2]);
	unsigned maxBatchSize = stoi(argv[3]);
	bool fp16 = (argc == 5 && argv[5] == "fp16") ? true : false;
	if (!check(maxBatchSize, enginePath)) return -1;
	
	if (enginePath.substr(enginePath.find_last_of(".") + 1) == "onnx"){
		if (convertOnnx2Trt(enginePath, maxBatchSize, fp16)){
			cout << "Export to TensorRT Success! \n"; 
			return true;
		}
		else{
			cerr << "ERROR: Export Failed! \n"; 
			return false;
		}
	}
}

bool convertWithDynamicInputShape(int argc, char **argv){
	string enginePath = string(argv[2]);
	unsigned maxBatchSize = stoi(argv[3]);
	string input_tensor_name = string(argv[4]);
	bool fp16 = (argc == 9 && (string(argv[8]) == "fp16")) ? true : false;
	if (!check(maxBatchSize, enginePath)) return false;
	vector<unsigned> dimension;
	for (unsigned i = 5; i < 8; i++){
		if (stoi(argv[i]) > 0){
			dimension.emplace_back(stoi(argv[i]));
		}
		else{
			cerr << "ERROR: Dimension must be more than 0 \n";
			return false;
		}
	}
	
	if (convertOnnx2Trt(enginePath, maxBatchSize, fp16, input_tensor_name, dimension, true)){
		cout << "Export to TensorRT Success! \n";
		dimension.clear();
		return true;
	}
	else{
		cerr << "ERROR: Export Failed! \n";
		dimension.clear();
		return false;
	}
}

bool infer(int argc, char **argv){
	string enginePath = string(argv[2]);
	string folderPath = string(argv[3]);
	unsigned batchSize = stoi(argv[4]);
	if (!check(batchSize, enginePath)) return -1;
	
	if (folderPath[folderPath.length() - 1] != '/' && folderPath[folderPath.length() -1] != '\\') {
		folderPath = folderPath + '/';
	}
	cv::Mat image;
	std::vector<std::string> fileNames;
	std::vector<cv::Mat> images;
	TRTParser engine;
	unsigned nrof_iamges = 0;
	unsigned i = 0;

	if (readFilesInDir(folderPath.c_str(), fileNames) < 0) {
        std::cout << "read_files_in_dir failed." << std::endl;
        return false;
    }		
	if (!engine.init(enginePath)){
		cerr << "ERROR: Could not parse tensorRT engine! \n";
		return false;
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
				cout << fileNames[f + i] << " is not an image! \n";
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
	return true;
}