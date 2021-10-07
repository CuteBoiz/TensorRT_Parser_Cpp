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

bool SetPrimaryCudaDevice(const unsigned gpuNum){
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
    	size_t totalDevMem, freeDevMem;
		cudaMemGetInfo(&freeDevMem, &totalDevMem);
        cout << "[INFO] Switched to GPU:" << gpuNum << " success! Free memory: " << freeDevMem/1048576 <<"MB. \n";
        return true;
    }
    else{
        return false;
    }
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

string GetArgumentsValue(const int argc, char** argv, unsigned& argsIndex, const string type){
    if (type == "store_true"){
        argsIndex += 1;
        return "";
    }
    else{
        argsIndex += 2;
    }
    if (argsIndex >= argc) {
        throw std::invalid_argument("[ERROR] None value for argument!\n");
    }    
    string value = string(argv[argsIndex]);
    if (type == "string"){
        //do nothing.
    }
    else if (type == "file") {
        if (!CheckFileIfExist(value.c_str())) {
            throw std::invalid_argument("[ERROR]: '" + value + "' does not exist! \n");
        }
    }
    else if (type == "folder") {
        DIR *dir = opendir(value.c_str());
        bool temp;
        if (dir) temp =  true;
        else if (ENOENT == errno) temp = false;
        else temp = false;
        closedir(dir);
        if (!temp){
            throw std::invalid_argument("[ERROR]: Folder '" + value + "' does not exist or Could not open! \n");
        }
    } 
    else if (type == "int") {
        try  {
           stoi(value);
        }
        catch (exception &err) {
            throw std::invalid_argument("[ERROR]: Could not cast '" + value + "' to int!\n");
        }
    } 
    else if (type == "float") {
        try  {
           stod(value);
        }
        catch (exception &err) {
            throw std::invalid_argument("[ERROR]: Could not cast '" + value + "' to float!\n");
        }
    } 
    else {
        throw std::invalid_argument("[ERROR] Undefined value type '" + type + "'! \n");
    }
    return value;
}


ExportConfig::ExportConfig(){
    onnxEnginePath = "";
    maxBatchsize = 0;
    maxWorkspaceSize = 0;
    useFP16 = false;
    useDynamicShape = false;
    inputTensorName = "";
    tensorDims = {};
}

bool ExportConfig::Update(const string enginePath, const unsigned i_maxbatchsize, const size_t workspaceSize, const bool fp16){
    onnxEnginePath = enginePath;
    maxBatchsize = i_maxbatchsize;
    maxWorkspaceSize = workspaceSize;
    useFP16 = fp16;
    useDynamicShape = false;
    inputTensorName = "";
    tensorDims = {};
    if (maxBatchsize <= 0){
        cerr <<"[ERROR] Max batchsize must be more than 0! \n";
        return false;
    }
    if (!CheckFileIfExist(onnxEnginePath)){
        cerr <<"[ERROR] '"<< onnxEnginePath << "' not found! \n";
        return false;
    }
    size_t totalDevMem, freeDevMem;
    cudaMemGetInfo(&freeDevMem, &totalDevMem);
    if (maxWorkspaceSize > freeDevMem) {
        cerr << "[ERROR] Not enough Gpu Memory! Model's WorkspaceSize: " << maxWorkspaceSize/1048576 << "MB. Free memory left: " << freeDevMem/1048576 <<"MB. \nTry decreasing workspacesize to continue.\n";
        return false;
    }
    return true;
}

bool ExportConfig::Update(const string enginePath, const unsigned i_maxbatchsize, const size_t workspaceSize, const bool fp16, const string tensorName, const vector<unsigned> dims){
    onnxEnginePath = enginePath;
    maxBatchsize = i_maxbatchsize;
    maxWorkspaceSize = workspaceSize;
    useFP16 = fp16;
    useDynamicShape = true;
    inputTensorName = tensorName;
    for (unsigned i = 0; i < dims.size(); i++){
        tensorDims.emplace_back(dims.at(i));
    }
    if (maxBatchsize <= 0){
        cerr <<"[ERROR] Max batchsize must be more than 0! \n";
        return false;
    }
    if (!CheckFileIfExist(onnxEnginePath)){
        cerr <<"[ERROR] '"<< onnxEnginePath << "' not found! \n";
        return false;
    }
    size_t totalDevMem, freeDevMem;
    cudaMemGetInfo(&freeDevMem, &totalDevMem);
    if (maxWorkspaceSize > freeDevMem) {
        cerr << "[ERROR] Not enough Gpu Memory! Model's WorkspaceSize: " << maxWorkspaceSize/1048576 << "MB. Free memory left: " << freeDevMem/1048576 <<"MB. \nTry decreasing workspacesize to continue.\n";
        return false;
    }
    if (inputTensorName == ""){
        cerr << "[ERROR] Input tensor name is empty! \n";
        return false;
    }
    if (tensorDims.size() != 3){
        cerr << "[ERROR] Dimension of dynamic shape must be 3! \n";
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
	cout << "\t - Max workspace size: " << maxWorkspaceSize/1048576 << " MB" << endl;
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


