/*
Convert Onnx engine to TensorRT Engine And Infer.

Author: phatnt
Date modified: 2021-10-06
 */

#include <iostream>
#include <chrono>
#include "utils.h"
#include "TRTParser.h"
using namespace std;

#define DEFAULT_MAX_WORKSPACE_SIZE (1048576 * 1300)
#define DEFAULT_MAX_BATCHSIZE 1
#define DEFAULT_USE_FP16 false

#define DEFAULT_INFER_BATCHSIZE 1
#define DEFAULT_USE_SOFTMAX false

static bool GetExportConfig(int argc, char ** argv, ExportConfig& config);
static bool TRT_Inference(int argc, char **argv);

int main(int argc,char** argv) {
	/*
	Args:
	[mode]
			export: Export onnx => TensorRT engine.		
			
			infer: Infer TensorRT engine.
	*/
	
	if (string(argv[1]) == "export") {
		ExportConfig config;
		if (!GetExportConfig(argc, argv, config)) {
			cerr << "[ERROR] Get Arguments error!\n";
			return -1;
		}
		if (!ExportOnnx2Trt(config)){
			cerr << "[ERROR] Export Failed! \n";
			return -1;
		}
		cout << "[INFO] Export Successed! \n";
		return 0;
	}

	else if (string(argv[1]) == "infer") {
		if (TRT_Inference(argc, argv)) {
			cout << "[INFO] Inference Successed! Done!\n";
			return 0;
		}
		return -1;
	}
	else {
		cout << "[ERROR] Undefined mode: '" << string(argv[1]) << "'. \n";
		cout <<	"[export]: to export Onnx Engine => TensorRT Engine. \n\n";
		cout << "[infer]: TensorRT engine inference. \n";
		return -1;
	}
	
	return 0;
}


bool GetExportConfig(int argc, char ** argv, ExportConfig& config) {
	/*
	Get config from arguments use for tensorrt export.
	Args:
		--weight <string>: 			path to onnx engine.
		--fp16 <bool>:				use FP16 fast mode (x2 inference time).
		--maxbatchsize <unsigned>:	inference max batchsize.
		--workspace <unsigned>:		max workspace size(MB)
		--tensor <string>:			input tensor's name.
		--dims <array(unsigned)>:	input tensor's dimension. 
		config <ExportConfig>:  	rerturned config for export.
	Return:
		<bool> Success checking.
	 */
	vector<string> required_args = {"--weight"};
	vector<string> non_req_args = {"--fp16", "--maxbatchsize", "--workspace", "--tensor", "--dims", "--gpu"};
	
	vector<string> arguments = {};
	unsigned argsIndex = 1;
	string enginePath = "";
	bool useFP16 = DEFAULT_USE_FP16;
	size_t workspaceSize = DEFAULT_MAX_WORKSPACE_SIZE;
	unsigned maxBatchSize = DEFAULT_MAX_BATCHSIZE;
	bool useDynamicShape = false;
	string inputTensorName = "";
	vector<unsigned> tensorDims = {};
	
	//Get Arguments from argv and Check condition for required args.
	for (unsigned i = 2; i < argc; i++) {
		if (string(argv[i]).rfind("--") != -1) {
			arguments.emplace_back(string(argv[i]));
		}
	}
	if (!CheckRequiredArguments(required_args, arguments)) return false;
	
	//Get value from arguments and value validable checking.
	for (unsigned i = 0; i < arguments.size(); i++) {
		if (arguments.at(i) == "--weight") {
			argsIndex += 2;
			if (argsIndex >= argc) {
				cerr << "[ERROR] None value for [--weight]! \n";
				return false;
			}
			if (!CheckValidValue(string(argv[argsIndex]), "file")){
				cerr << "[ERROR] Invalid value for [--weight]: '" << argv[argsIndex] <<"'! \n";
				return false;
			}
			enginePath = string(argv[argsIndex]);
		}
		else if (arguments.at(i) == "--fp16") {
			argsIndex += 1;
			useFP16 = true;
		}
		else if (arguments.at(i) == "--maxbatchsize") {
			argsIndex += 2;
			if (argsIndex >= argc) {
				cerr << "[ERROR] None value for [--maxbatchsize]! \n";
				return false;
			}
			if (!CheckValidValue(string(argv[argsIndex]), "unsigned")){
				cerr << "[ERROR] Invalid value for [--maxbatchsize]: '" << argv[argsIndex] <<"'! \n";
				return false;
			}
			maxBatchSize = stoi(argv[argsIndex]);
			
		}
		else if (arguments.at(i) == "--workspace") {
			argsIndex += 2;
			if (argsIndex >= argc) {
				cerr << "[ERROR] None value for [--workspace]! \n";
				return false;
			}
			if (!CheckValidValue(string(argv[argsIndex]), "unsigned")){
				cerr << "[ERROR] Invalid value for [--workspace]: '" << argv[argsIndex] <<"'! \n";
				return false;
			}
			workspaceSize = stoi(argv[argsIndex]) * 1048576;
		}
		else if (arguments.at(i) == "--tensor") {
			useDynamicShape = true;
			argsIndex += 2;
			if (argsIndex >= argc) {
				cerr << "[ERROR] None value for [--tensor]! \n";
				return false;
			}
			inputTensorName = string(argv[argsIndex]);
		}
		else if (arguments.at(i) == "--dims") {
			useDynamicShape = true;
			if (argsIndex + 4 >= argc) {
				cerr << "[ERROR] Not enough values for [--dims]! \n";
				return false;
			}
			try {
				tensorDims.emplace_back(stoi(argv[argsIndex+2]));
				tensorDims.emplace_back(stoi(argv[argsIndex+3]));
				tensorDims.emplace_back(stoi(argv[argsIndex+4]));
			}
			catch (exception &err) {
				cerr << "[ERROR] Invalid value for '--dims': " << argv[argsIndex+2] << " " << argv[argsIndex+3] << " " << argv[argsIndex+4] << ". Value must be unsigned interger array!\n";
				return false;
			}
			argsIndex += 4;
		}
		else if (arguments.at(i) == "--gpu") {
			argsIndex += 2;
			if (argsIndex >= argc) {
				cerr << "[ERROR] Not enough values for [--gpu]! \n";
				return false;
			}
			if (!CheckValidValue(string(argv[argsIndex]), "unsigned")){
				cerr << "[ERROR] Invalid value for [--gpu]: '" << argv[argsIndex] <<"'! \n";
				return false;
			}
			unsigned gpuNum = stoi(argv[argsIndex]);
			int deviceCount = 0;
			cudaError_t err = cudaSuccess;
			err = cudaGetDeviceCount(&deviceCount);
			if (err != cudaSuccess) {
				cerr << "[ERROR] " << cudaGetErrorString(err) << endl; 
				return false;
			}
			else {
				cout << "[INFO] Device Count: " << deviceCount << endl;
			}
			if (gpuNum >= deviceCount){
				cout << "[ERROR] Gpu num must smaller than '" << deviceCount <<"'. \n";
				return false;
			}
			err = cudaSetDevice(gpuNum);
			if (err == cudaSuccess) {
	  			cout << "[INFO] Switched to GPU:" << gpuNum << " success!\n";
	  		}
	  		else{
	  			cout << "[ERROR] Set CUDA device failed!\n";
	  			return false;
	  		}
		}
		else {
			cerr << "[ERROR] Invalid arguments :[" << arguments.at(i) << "]. \n";
			return false;
		}
		if (argsIndex < argc-1) {
			if (!CheckValidArgument(required_args, non_req_args, string(argv[argsIndex+1]))) return false;
		}
	}
	size_t totalDevMem, freeDevMem;
	cudaMemGetInfo(&freeDevMem, &totalDevMem);
	if (workspaceSize > freeDevMem) {
		cerr << "[ERROR] Not enough Gpu Memory! Model's WorkspaceSize: " << workspaceSize/1048576 << "MB. Free memory left: " << freeDevMem/1048576 <<"MB. \nReduce workspacesize to continue.\n";
		return false;
	}
	//Update config.
	if (!useDynamicShape) {
		return config.Update(enginePath, maxBatchSize, workspaceSize, useFP16);
	}
	else {
		return config.Update(enginePath, maxBatchSize, workspaceSize, useFP16, inputTensorName, tensorDims);
	}
}


bool TRT_Inference(int argc, char **argv) {
	/*
	TensorRT Engine Inference.
	Args:
		--weight <string>		: path to tensorrt engine.
		--data <string>			: path to inference images's folder.
		--batchSize <unsigned>	: infernce batchsize (must smaller than max batchsize of trt engine)
		--softmax <bool>		: add softmax to last layer of engine.
	Return:
		<bool>: Success checking.
	 */
	vector<string> required_args = {"--weight", "--data"};
	vector<string> non_req_args = {"--batchsize", "--softmax", "--gpu"};

	unsigned argsIndex = 1;
	vector<string> arguments = {};
	string enginePath = "";
	string dataPath = "";
	unsigned batchsize = DEFAULT_INFER_BATCHSIZE;
	bool useSofmax = DEFAULT_USE_SOFTMAX;

	for (unsigned i = 2; i < argc; i++){
		if (string(argv[i]).rfind("--") != -1){
			arguments.emplace_back(string(argv[i]));
		}
	}
	if (!CheckRequiredArguments(required_args, arguments)) return false;

	for (unsigned i = 0; i < arguments.size(); i++) {
		if (arguments.at(i) == "--weight"){
			argsIndex += 2;
			if (argsIndex >= argc) {
				cerr << "[ERROR] None value for [--weight]! \n";
				return false;
			}
			if (!CheckValidValue(string(argv[argsIndex]), "file")){
				cerr << "[ERROR] Invalid value for [--weight]: '" << argv[argsIndex] <<"'! \n";
				return false;
			}
			enginePath = string(argv[argsIndex]);
		}
		else if (arguments.at(i) == "--data"){
			argsIndex += 2;
			if (argsIndex >= argc) {
				cerr << "[ERROR] None value for [--data]! \n";
				return false;
			}
			if (!CheckValidValue(string(argv[argsIndex]), "folder")){
				cerr << "[ERROR] Invalid value for [--data]: '" << argv[argsIndex] <<"'! \n";
				return false;
			}
			dataPath = string(argv[argsIndex]);
		}
		else if (arguments.at(i) == "--batchsize") {
			argsIndex += 2;
			if (argsIndex >= argc) {
				cerr << "[ERROR] None value for [--batchsize]! \n";
				return false;
			}
			if (!CheckValidValue(string(argv[argsIndex]), "unsigned")){
				cerr << "[ERROR] Invalid value for [--batchsize]: '" << argv[argsIndex] <<"'! \n";
				return false;
			}
			batchsize = stoi(argv[argsIndex]);
		}
		else if (arguments.at(i) == "--softmax"){
			argsIndex += 1;
			useSofmax = true;
		}
		else if (arguments.at(i) == "--gpu") {
			argsIndex += 2;
			if (argsIndex >= argc) {
				cerr << "[ERROR] Not enough values for [--gpu]! \n";
				return false;
			}
			if (!CheckValidValue(string(argv[argsIndex]), "unsigned")){
				cerr << "[ERROR] Invalid value for [--gpu]: '" << argv[argsIndex] <<"'! \n";
				return false;
			}
			unsigned gpuNum = stoi(argv[argsIndex]);
			int deviceCount = 0;
			cudaError_t err = cudaSuccess;
			err = cudaGetDeviceCount(&deviceCount);
			if (err != cudaSuccess) {
				cerr << "[ERROR] " << cudaGetErrorString(err) << endl; 
				return false;
			}
			else {
				cout << "[INFO] Device Count: " << deviceCount << endl;
			}
			if (gpuNum >= deviceCount){
				cout << "[ERROR] Gpu num must smaller than '" << deviceCount <<"'. \n";
				return false;
			}
			err = cudaSetDevice(gpuNum);
			if (err == cudaSuccess) {
	  			cout << "[INFO] Switched to GPU:" << gpuNum << " success!\n";
	  		}
	  		else{
	  			cout << "[ERROR] Set CUDA device failed!\n";
	  			return false;
	  		}
		}
		else{
			cerr << "[ERROR] Invalid arguments :[" << arguments.at(i) << "]. \n";
			return false;
		}
		if (argsIndex < argc-1){
			if (!CheckValidArgument(required_args, non_req_args, string(argv[argsIndex+1]))) return false;
		}
	}

	
	if (dataPath[dataPath.length() - 1] != '/' && dataPath[dataPath.length() -1] != '\\') {
		dataPath = dataPath + '/';
	}
	cv::Mat image;
	vector<string> fileNames;
	vector<cv::Mat> images;
	TRTParser engine;
	unsigned nrofInferIamges = 0;
	
 	//Initialize engine
	if (engine.Init(enginePath)){
		cout << "[INFO] Load '" << enginePath << "' success!. Inferencing... \n";
	}
	else{
		cerr << "[ERROR] Could not parse tensorRT engine! \n";
		return false;
	}
	//Get images form folder
	if (ReadFilesInDir(dataPath.c_str(), fileNames)) {
        cout << "[INFO] Load data from '" << dataPath << "' success! Total " << fileNames.size() << " files. \n";
    }
    else{
    	cout << "[ERROR] Could not read files from"<< dataPath << endl;
        return false;
    }
    unsigned i = 0;
	for (unsigned f = 0; f < fileNames.size(); f += i) {
		//Prepare inference batch
		unsigned batchIndex = 0;
		for (i = 0; batchIndex < batchsize && (f + i) < fileNames.size(); i++) {
			string fileExtension = fileNames[f + i].substr(fileNames[f + i].find_last_of(".") + 1);
			if (fileExtension == "bmp" || fileExtension == "png" || fileExtension == "jpeg" || fileExtension == "jpg") {
				cout << fileNames[f + i] << endl;
				cv::Mat image = cv::imread(dataPath + fileNames[f + i]);
				images.emplace_back(image);
				batchIndex++;
				nrofInferIamges++;
			}
			else{
				cout << "[WARNING] '" << fileNames[f + i] << "' not an image! \n";
			}
		}
		if (images.size() == 0) {
			continue; //Skip if got a non-image files stack.
		}
		//Inference
		auto start = chrono::system_clock::now();
		if (!engine.Inference(images, useSofmax)){
			cerr << "[ERROR] Inference error! \n";
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
	cout << "[INFO] Total inferenced images: " << nrofInferIamges << endl;
	return true;
}