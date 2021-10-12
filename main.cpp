/*
Convert Onnx engine to TensorRT Engine And Infer.

Author: phatnt
Date modified: 2021-10-07
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
			cerr << "[ERROR] Update ExportConfig Error !\n";
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
		cout << "[ERROR] Undefined Mode: '" << string(argv[1]) << "'. \n\n";
		cout << "[HELP] Mode:\n";
		cout <<	"\t+ (export): Export Onnx Engine => TensorRT Engine. \n";
		cout << "\t+ (infer): TensorRT engine inference. \n";
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
	unsigned argsIndex = 1;
	try {
		for (unsigned i = 0; i < arguments.size(); i++) {
			//Check next argument vailable
			if (argsIndex < argc-1) { 
				if (!CheckValidArgument(required_args, non_req_args, string(argv[argsIndex+1]))) return false;
			}
			//Get value from arguments
			if (arguments.at(i) == "--weight") {
				enginePath = GetArgumentsValue(argc, argv, argsIndex, "file");
			}
			else if (arguments.at(i) == "--fp16") {
				GetArgumentsValue(argc, argv, argsIndex, "store_true");
				useFP16 = true;
			}
			else if (arguments.at(i) == "--maxbatchsize") {
				maxBatchSize = stoi(GetArgumentsValue(argc, argv, argsIndex, "int"));
			}
			else if (arguments.at(i) == "--workspace") {
				workspaceSize = stoi(GetArgumentsValue(argc, argv, argsIndex, "int")) * 1048576;
			}
			else if (arguments.at(i) == "--tensor") {
				useDynamicShape = true;
				inputTensorName = GetArgumentsValue(argc, argv, argsIndex, "string");
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
				unsigned gpuNum = stoi(GetArgumentsValue(argc, argv, argsIndex, "int"));
				if (!SetPrimaryCudaDevice(gpuNum)){
					cout << "[ERROR] switch primary CUDA device failed!\n";
					return false;
				}
			}
			else {
				cerr << "[ERROR] Invalid arguments :[" << arguments.at(i) << "]. \n";
				return false;
			}
		}
		//Check next argument existance
		if (argsIndex < argc-1 && !CheckValidArgument(required_args, non_req_args, string(argv[argsIndex+1]))) return false;
	}
	catch (exception& err) {
		cerr << "[ERROR] Get arguments error!\n";
		cerr << err.what();
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

	try{
		for (unsigned i = 0; i < arguments.size(); i++) {
			//Check next argument vailable
			if (argsIndex + 1 <= argc){
				if (!CheckValidArgument(required_args, non_req_args, string(argv[argsIndex+1]))) return false;
			}
			//Get value from arguments
			if (arguments.at(i) == "--weight"){
				enginePath = GetArgumentsValue(argc, argv, argsIndex, "file");
			}
			else if (arguments.at(i) == "--data"){
				dataPath = GetArgumentsValue(argc, argv, argsIndex, "folder");
			}
			else if (arguments.at(i) == "--batchsize") {
				batchsize = stoi(GetArgumentsValue(argc, argv, argsIndex, "int"));
			}
			else if (arguments.at(i) == "--softmax"){
				GetArgumentsValue(argc, argv, argsIndex, "store_true");
				useSofmax = true;
			}
			else if (arguments.at(i) == "--gpu") {
				unsigned gpuNum = stoi(GetArgumentsValue(argc, argv, argsIndex, "int"));
				if (!SetPrimaryCudaDevice(gpuNum)){
					cout << "[ERROR] switch primary CUDA device failed!\n";
					return false;
				}
			}
			else{
				cerr << "[ERROR] Invalid arguments :[" << arguments.at(i) << "]. \n";
				return false;
			}
		}
		//Check next argument existance
		if (argsIndex < argc-1 && !CheckValidArgument(required_args, non_req_args, string(argv[argsIndex+1]))) return false;
	}
	catch (exception& err) {
		cerr << err.what();
		return false; 
	}
	

	TRTParser engine;
	unsigned nrofInferIamges = 0;
	vector< vector< cv::Mat >> batchedImages;
	
 	//Initialize engine
	if (engine.Init(enginePath)){
		cout << "[INFO] Load '" << enginePath << "' success!. Inferencing... \n";
	}
	else{
		cerr << "[ERROR] Could not parse tensorRT engine! \n";
		return false;
	}

	//Prepare data
	try {
		batchedImages = PrepareImageBatch(dataPath, batchsize);
	}
	catch (exception& err){
		cerr << err.what();
		return false; 
	}
    
    for (unsigned i = 0; i < batchedImages.size(); i++){
    	auto start = chrono::system_clock::now();
		if (!engine.Inference(batchedImages.at(i), useSofmax)){
			cerr << "[ERROR] Inference error! \n";
			return false;
		}
		auto end = chrono::system_clock::now();
		cout << chrono::duration_cast<chrono::milliseconds>(end - start).count() << "ms. \n";
		nrofInferIamges += batchedImages.at(i).size();
    }
    batchedImages.clear();
	cout << "[INFO] Total inferenced images: " << nrofInferIamges << endl;
	return true;
}