/*
Convert Onnx engine to TensorRT Engine And Infer.

Author: phatnt
Date modified: 2021-09-29

 */

#include <iostream>
#include <chrono>
#include "utils.h"
#include "TRTParser.h"

using namespace std;

static bool Check(unsigned maxBatchSize, string enginePath);
static bool Convert(int argc, char** argv);
static bool ConvertWithDynamicShapeInput(int argc, char **argv);
static bool TRT_Inference(int argc, char **argv);

int main(int argc,char** argv){
	/*
	arguments:
	[mode] 	Export onnx => trt
			-e [model path] [max batchsize] ([fp16])

			Export onnx => trt (with dynamic shape (1 input only))
			-ed [model path] [max batchsize] [input tensor name] [dimension 1] [dimension 2] [dimension 3] ([fp16])
			
			Infer TensorRT engine.
			-i [model path] [images folder path] [inference batchsize]

	*/
	
	if ((argc >= 4 && argc <= 5) && string(argv[1]) == "-e"){
		if(Convert(argc, argv)){
			return 0; 
		}
		return -1;
	}
	else if ((argc >= 8 && argc <= 9) && string(argv[1]) == "-ed"){
		if (ConvertWithDynamicShapeInput(argc, argv)){
			return 0;
		}
		return -1;
		
	} 
	else if ((argc >= 5 && argc <= 6)&& string(argv[1]) == "-i"){
		if (TRT_Inference(argc, argv)){
			return 0;
		}
		return -1;
	}
	else{
		cout << "[ERROR]: Undefined arguments. \n";
		cout <<	"[-e] [OnnxEnginePath] [max batchsize] ([fp16]) to export onnx => trt model \n\n";
		cout << "[-ed][OnnxEnginePath] [max batchsize] [input tensor name] [d1] [d2] [d3] ([fp16])\n to export dynamic shape trt model(with 1 Input ONLY) \n\n";
		cout << "[-i] [trtEnginePath] [imagesFolderPath] [batchSize] to infer trt model \n";
		return -1;
	}
	
	return 0;
}


bool Check(unsigned maxBatchSize, string enginePath){
	/*
	Check condition of maxBatchSize and existance of enginePath.
	Input:
		- maxBatchSize 	<unsigned int> max inference's batchsize.
		- enginePath: 	<string> path to engine.
	Return:
		<bool> Condition checked result.

	 */
	if (maxBatchSize <= 0){
		cerr <<"[ERROR]: max batchsize must be more than 0! \n";
		return false;
	}
	if (!CheckFileIfExist(enginePath)){
		cerr <<"[ERROR]: "<< enginePath << " not found! \n";
		return false;
	}
	return true;
}


bool Convert(int argc, char** argv){
	/*
	Convert onnx engine to tensorrt engine.
	Input:
		argv[2]: enginePath 	<string>: path to onnx engine.
		argv[3]: maxBatchSize 	<unsigned int>: max inference's batchsize.
		(agrv[4]): fp16 		<bool>: export to FP16 fast mode engine.
	Return:
		<bool>: Success check. 
	 */
	string enginePath = string(argv[2]);
	unsigned maxBatchSize = stoi(argv[3]);
	bool fp16 = (argc == 5 && string(argv[4]) == "fp16") ? true : false;
	if (!Check(maxBatchSize, enginePath)) return false;
	
	if (enginePath.substr(enginePath.find_last_of(".") + 1) == "onnx"){
		if (ExportOnnx2Trt(enginePath, maxBatchSize, fp16)){
			cout << "[INFO]: Export to TensorRT Success! \n"; 
			return true;
		}
		else{
			cerr << "[ERROR]: Export Failed! \n"; 
			return false;
		}
	}
}

bool ConvertWithDynamicShapeInput(int argc, char **argv){
	/*
	Convert onnx engine to tensorrt engine with dynamic shape input.
	Input:
		argv[2]: enginePath 		<string>: path to onnx engine.
		argv[3]: maxBatchSize 		<unsigned int>: max inference's batchsize.
		argv[4]: input_tensor_name 	<string>: network input tensor's name.
		argv[5-6-7]: dimension		<array(int)> dimension of input tensor.
		(agrv[8]): fp16 			<bool>: export to FP16 fast mode engine.
	Return:
		<bool>: Success check. 
	 */
	string enginePath = string(argv[2]);
	unsigned maxBatchSize = stoi(argv[3]);
	string input_tensor_name = string(argv[4]);
	bool fp16 = (argc == 9 && (string(argv[8]) == "fp16")) ? true : false;
	if (!Check(maxBatchSize, enginePath)) return false;

	vector<unsigned> dimension;
	for (unsigned i = 5; i < 8; i++){
		if (stoi(argv[i]) > 0){
			dimension.emplace_back(stoi(argv[i]));
		}
		else{
			cerr << "[ERROR]: Dimension must be more than 0 \n";
			return false;
		}
	}
	
	if (ExportOnnx2Trt(enginePath, maxBatchSize, fp16, input_tensor_name, dimension, true)){
		cout << "[INFO]: Export to TensorRT Success! \n";
		dimension.clear();
		return true;
	}
	else{
		cerr << "[ERROR]: Export Failed! \n";
		dimension.clear();
		return false;
	}
}

bool TRT_Inference(int argc, char **argv){
	/*
	TensorRT Engine Inference.
	Input:
		argv[2]: enginePath <string>: path to tensorrt engine.
		argv[3]: folderPath <string>: path to inference images's folder.
		argv[4]: batchSize 	<int>: infernce batchsize (must smaller than max batchsize of trt engine)
		argv[5]: softmax 	<bool>: add softmax to last layer of engine.
	Return:
		<bool>: Success checking.
	 */
	string enginePath = string(argv[2]);
	string folderPath = string(argv[3]);
	unsigned batchSize = stoi(argv[4]);
	bool softmax = (argc == 6 && (string(argv[5]) == "softmax")) ? true : false;
	if (!Check(batchSize, enginePath)) return -1;
	
	if (folderPath[folderPath.length() - 1] != '/' && folderPath[folderPath.length() -1] != '\\') {
		folderPath = folderPath + '/';
	}
	cv::Mat image;
	vector<string> fileNames;
	vector<cv::Mat> images;
	TRTParser engine;
	unsigned nrofInferIamges = 0;
	unsigned i = 0;

	//Get images form folder
	if (!ReadFilesInDir(folderPath.c_str(), fileNames)) {
        cout << "[ERROR]: Could not read files from"<< folderPath << endl;
        return false;
    }
 	//Initialize engine
	if (!engine.Init(enginePath)){
		cerr << "[ERROR]: Could not parse tensorRT engine! \n";
		return false;
	}
	for (unsigned f = 0; f < (unsigned)fileNames.size(); f += i) {
		//Prepare inference batch
		unsigned index = 0;
		for (i = 0; index < batchSize && (f + i) < (unsigned)fileNames.size(); i++) {
			string fileExtension = fileNames[f + i].substr(fileNames[f + i].find_last_of(".") + 1);
			if (fileExtension == "bmp" || fileExtension == "png" || fileExtension == "jpeg" || fileExtension == "jpg") {
				cout << fileNames[f + i] << endl;
				cv::Mat image = cv::imread(folderPath + fileNames[f + i]);
				images.emplace_back(image);
				index++;
				nrofInferIamges++;
			}
			else{
				cout << fileNames[f + i] << " is not an image! \n";
			}
		}
		if (images.size() == 0) {
			continue; //Skip if got a of non-image files stack.
		}

		//Inference
		auto start = chrono::system_clock::now();
		if (!engine.Inference(images, softmax)){
			cerr << "[ERROR]: Inference error! \n";
			return false;
		}
		auto end = chrono::system_clock::now();
		cout << chrono::duration_cast<chrono::milliseconds>(end - start).count() << "ms. \n";

		//Clear image temp memories
		for (unsigned j = 0; j < images.size(); j++){
			images.at(j).release();
		}
		images.clear();
	}
	cout << "[INFO]: Total inferenced images: " << nrofInferIamges << endl;
	fileNames.clear();
	return true;
}