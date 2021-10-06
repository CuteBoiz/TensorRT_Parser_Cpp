#include "utils.h"


bool CheckFileIfExist(const string filePath) {
	ifstream f(filePath, ios::binary);
	if (!f.good()) {
		f.close();
		return false;
	}
	f.close();
	return true;
}

bool ReadFilesInDir(const char *p_dir_name, vector<string> &file_names) {
    DIR *p_dir = opendir(p_dir_name);
    if (p_dir == nullptr) {
        return false;
    }
    struct dirent* p_file = nullptr;
    while ((p_file = readdir(p_dir)) != nullptr) {
        if (strcmp(p_file->d_name, ".") != 0 &&
            strcmp(p_file->d_name, "..") != 0) {
            string cur_file_name(p_file->d_name);
            file_names.push_back(cur_file_name);
        }
    }
    closedir(p_dir);
    return true;
}

bool CheckRequiredArguments(const vector<string> required_args, const vector<string> args){
	vector<string> missed_args;
	string str_args = "";
	for (unsigned i = 0; i < args.size(); i++) {
		str_args += args.at(i);
	}
	for (unsigned i = 0; i < required_args.size(); i++) {
		if (str_args.rfind(required_args.at(i)) == -1) {
			missed_args.emplace_back(required_args.at(i));
		}
	}
	if (missed_args.size() > 0){
		cerr << "[ERROR] Missed Arguments: ";
		for (unsigned i = 0; i < missed_args.size(); i++) {
			cerr << "[" << missed_args.at(i) << "] ";
		}
		cout << endl;
		return false;
	}
	return true;
}

bool CheckValidValue(string value, string type) {
	if (type == "file") {
		if (CheckFileIfExist(value.c_str())) {
			return true;
		}
		else{
			cerr << "[ERROR]: '" << value <<"' is not exist! \n";
			return false;
		}
	}
	else if (type == "folder") {
		DIR *dir = opendir(value.c_str());
		bool result;
		if (dir){
			result =  true;
		}
		else if (ENOENT == errno){
			cerr << "[ERROR]: Folder '" << value <<"' is not exist! \n";
			result = false;
		}
		else{
			cerr << "[ERROR]: Could not open '" << value <<"'!\n";
			result = false;
		}
		closedir(dir);
		return result;
	} 
	else if (type == "unsigned") {
		try  {
			stoi(value);
		}
		catch (exception &err) {
			return false;
		}
		return true;
	} 
	else if (type == "double") {
		try  {
			stod(value);
		}
		catch (exception &err) {
			return false;
		}
		return true;
	} 
	else {
		cerr << "[ERROR] Undefined value type '"<< type <<"'! \n";
		return false;
	}
}

bool CheckValidArgument(const vector<string> required_args, const vector<string> valid_args, const string args) {
	vector<string> invalid_args;
	string str_valid_args = "";
	for (unsigned i = 0; i < required_args.size(); i++) {
		str_valid_args += required_args.at(i);
	}
	for (unsigned i = 0; i < valid_args.size(); i++) {
		str_valid_args += valid_args.at(i);
	}
	if (str_valid_args.rfind(args) == -1) {
		cerr << "[ERROR] Invalid Argument [" << args << "]. \n";
		return false;
	}
	return true;
}


nvinfer1::ICudaEngine* LoadOnnxEngine(const ExportConfig exportConfig) {
	string onnxEnginePath = exportConfig.onnxEnginePath;
	unsigned maxWorkspaceSize = exportConfig.maxWorkspaceSize;
	unsigned maxBatchsize = exportConfig.maxBatchsize;
    bool useDynamicShape = exportConfig.useDynamicShape;
    bool useFP16 = exportConfig.useFP16;
    cout << "[INFO] Enigne info: \n";
    cout << "\t - Engine Path: " << onnxEnginePath << endl;
	cout << "\t - Max inference batchsize: " << maxBatchsize << endl;
	cout << "\t - Max workspace size: " << maxWorkspaceSize/1e6 << " MB" << endl;
	cout << "\t - Use Dynamic shape: " << (useDynamicShape ? "True":"False") << endl;

	//Parse engine (Check onnx support operators if parse wasn't success)
	nvinfer1::IBuilder* builder{ nvinfer1::createInferBuilder(gLogger) };
	const auto flag = 1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
	nvinfer1::INetworkDefinition* network = builder->createNetworkV2(flag);
	TRTUniquePtr<nvonnxparser::IParser> parser{ nvonnxparser::createParser(*network, gLogger) };
	if (!parser->parseFromFile(onnxEnginePath.c_str(), static_cast<int>(nvinfer1::ILogger::Severity::kWARNING))){
		cerr << "[ERROR] Could not parse onnx engine \n";
		return nullptr;
	}

	//Building enigne
	TRTUniquePtr<nvinfer1::IBuilderConfig> config{ builder->createBuilderConfig() };
	config->setMaxWorkspaceSize(maxWorkspaceSize);
	
	builder->setMaxBatchSize(maxBatchsize);

	if (useDynamicShape){
		string inputTensorName = exportConfig.inputTensorName;
		vector<unsigned> tensorDims = exportConfig.tensorDims;
		cout << "\t - Input tensor: '" << inputTensorName << "': BactchSize ";
		for (unsigned i = 0; i < tensorDims.size(); i++) {
			cout << tensorDims.at(i) << " ";
		}
		cout << endl;
		auto profile = builder->createOptimizationProfile();
		profile->setDimensions(inputTensorName.c_str(), nvinfer1::OptProfileSelector::kMIN, nvinfer1::Dims4{1, tensorDims.at(0), tensorDims.at(1), tensorDims.at(2)});
		profile->setDimensions(inputTensorName.c_str(), nvinfer1::OptProfileSelector::kOPT, nvinfer1::Dims4{max(int(maxBatchsize/2),1), tensorDims.at(0), tensorDims.at(1), tensorDims.at(2)});
		profile->setDimensions(inputTensorName.c_str(), nvinfer1::OptProfileSelector::kMAX, nvinfer1::Dims4{maxBatchsize, tensorDims.at(0), tensorDims.at(1), tensorDims.at(2)});
		config->addOptimizationProfile(profile);
	}

	if (useFP16){
		if (builder->platformHasFastFp16()) {
			cout << "[INFO] Model exporting in FP16 Fast Mode\n";
			config->setFlag(nvinfer1::BuilderFlag::kFP16);
		}
		else{
			cout << "[INFO] This system does not support FP16 fast mode. Exporting model in FP32 Mode.\n";
		}
	}
	else{
		cout << "[INFO] Model exporting in FP32 Mode.\n";
	}

	return builder->buildEngineWithConfig(*network, *config);
}


bool ExportOnnx2Trt(const ExportConfig exportConfig) {	
	string onnxEnginePath = exportConfig.onnxEnginePath;
	//Check condition
	if (!CheckFileIfExist(onnxEnginePath)) {
		cout << "[ERROR] '" << onnxEnginePath << "' not found! \n";
		return false;
	}
	else {
		cout << "[INFO] Weight: '" << onnxEnginePath << "' found!, Converting to TensorRT. \n";
	}
	size_t lastindex = onnxEnginePath.find_last_of(".");
	string TRTFilename = onnxEnginePath.substr(0, lastindex) + ".trt";
	if (CheckFileIfExist(TRTFilename)) {
		cout << "[ERROR] '" << TRTFilename << "' already exist! \n";
		return false;
	}

	//Duplicate onnx eninge in order to avoid losing onnx engine. 
	char buf[BUFSIZ];
	size_t size;
	FILE* source = fopen(onnxEnginePath.c_str(), "rb");
	FILE* dest = fopen(TRTFilename.c_str(), "wb");
	while (size = fread(buf, 1, BUFSIZ, source)) {
		fwrite(buf, 1, size, dest);
	}
	fclose(source);
	fclose(dest);

	//Read onnx engine
	ofstream engineFile(TRTFilename, ios::binary);
	if (!engineFile){
		cerr << "[ERROR] Could not open engine file: " << TRTFilename << endl;
		remove(TRTFilename.c_str());
		return false;
	}
	nvinfer1::ICudaEngine* engine = LoadOnnxEngine(exportConfig);
	if (engine == nullptr) {
		cerr << "[ERROR] Could not load onnx engine" << endl;
		remove(TRTFilename.c_str());
		return false;
	}

	//Seialize tensorrt engine.
	TRTUniquePtr<nvinfer1::IHostMemory> serializedEngine{ engine->serialize() };
	if (serializedEngine == nullptr) {
		remove(TRTFilename.c_str());
		cerr << "[ERROR] Could not serialize engine \n";
		return false;
	}
	engineFile.write(static_cast<char*>(serializedEngine->data()), serializedEngine->size());
	cout << "[INFO] '"<< TRTFilename <<"' Created! \n";
	engine->destroy();
	return !engineFile.fail();
}


