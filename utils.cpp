#include "utils.h"
using namespace nvinfer1;


vector<string> splitString(string s, string delimiter){
    size_t pos_start = 0, pos_end, delim_len = delimiter.length();
    string token;
    vector<string> res;

    while ((pos_end = s.find (delimiter, pos_start)) != string::npos) {
        token = s.substr (pos_start, pos_end - pos_start);
        pos_start = pos_end + delim_len;
        res.push_back (token);
    }
    res.push_back (s.substr (pos_start));
    return res;
}


bool CudaCheck(cudaError_t status){                                                                       
    if (status != cudaSuccess){                                                   
        cout << "[ERROR] [CUDA Failure] " << cudaGetErrorString(status) 
             << " in file "<< __FILE__                                  
             << " at line " << __LINE__ << endl;                        
        return false;                                                    
    }
    return true;                                                   
}

bool CheckFileIfExist(const string filePath) {
	ifstream f(filePath, ios::binary);
	if (!f.good()) {
		f.close();
		return false;
	}
	f.close();
	return true;
}


bool SetPrimaryCudaDevice(const unsigned gpuNum){
    int deviceCount = 0;

    if (!CudaCheck(cudaGetDeviceCount(&deviceCount))) return false;

    else {
        cout << "[INFO] Device Count: " << deviceCount << endl;
    }
    if (gpuNum >= deviceCount){
        cout << "[ERROR] Gpu num must smaller than '" << deviceCount <<"'. \n";
        return false;
    }
    if (!CudaCheck(cudaSetDevice(gpuNum))) return false;

	size_t totalDevMem, freeDevMem;
	if (!CudaCheck(cudaMemGetInfo(&freeDevMem, &totalDevMem))) return false;
    cout << "[INFO] Switched to GPU:" << gpuNum << " success! Free memory: " << freeDevMem/1048576 <<"MB. \n";
    return true;

}

bool CheckRequiredArguments(const vector<string> required_args, const vector<string> args){
	vector<string> missed_args;
	string str_args = "";
	for (unsigned i = 0; i < args.size(); i++) {
		str_args += args.at(i);
	}
    for (unsigned i = 0; i < args.size(); i++) {
        string check_arg = args.at(i);
        unsigned count = 0;
        for (unsigned j = 0; j < args.size(); j++){
            if (check_arg == args.at(j)){
                count++;
                if (count >= 2){
                    cerr << "[ERROR] Duplicated Arguments [" << check_arg <<"]!\n";
                    return false;
                }
            }
        }
        
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
        throw std::invalid_argument("[ERROR] Got None value for [" + string(argv[argsIndex-1]) + "]!\n");
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
    else if (type == "int") {
        try {
           stoi(value);
        }
        catch (exception &err) {
            throw std::invalid_argument("[ERROR]: Could not cast '" + value + "' to int!\n");
        }
    } 
    else if (type == "float") {
        try {
           stod(value);
        }
        catch (exception &err) {
            throw std::invalid_argument("[ERROR]: Could not cast '" + value + "' to float!\n");
        }
    }
    else if (type == "array") {
        while ((argsIndex+1) < argc && string(argv[argsIndex+1]).rfind("--") == -1){
            argsIndex++;
            value += ("/" + string(argv[argsIndex]));
        }
    } 
    else {
        throw std::invalid_argument("[ERROR] Undefined value type '" + type + "'! \n");
    }
    return value;
}


ExportConfig::ExportConfig(){
    this->onnxEnginePath = "";
    this->maxBatchsize = 0;
    this->maxWorkspaceSize = 0;
    this->useFP16 = false;
    this->useDynamicShape = false;
    this->tensorNames = {};
    this->tensorDims = {};
}

bool ExportConfig::Update(const string onnxEnginePath, const unsigned maxBatchsize, const size_t maxWorkspaceSize, const bool fp16){
    this->onnxEnginePath = onnxEnginePath;
    this->maxBatchsize = maxBatchsize;
    this->maxWorkspaceSize = maxWorkspaceSize;
    this->useFP16 = fp16;
    this->useDynamicShape = false;
    this->tensorNames = {};
    this->tensorDims = {};
    if (maxBatchsize <= 0){
        cerr <<"[ERROR] Max batchsize must be more than 0! \n";
        return false;
    }
    if (!CheckFileIfExist(this->onnxEnginePath)){
        cerr <<"[ERROR] '"<< this->onnxEnginePath << "' not found! \n";
        return false;
    }
    size_t totalDevMem, freeDevMem;
    if (!CudaCheck(cudaMemGetInfo(&freeDevMem, &totalDevMem))) return false;
    if (this->maxWorkspaceSize > freeDevMem) {
        cerr << "[ERROR] Not enough Gpu Memory! Model's WorkspaceSize: " << this->maxWorkspaceSize/1048576 << "MB. Free memory left: " 
             << freeDevMem/1048576 <<"MB. \nTry decreasing workspacesize to continue.\n";
        return false;
    }
    return true;
}

bool ExportConfig::Update(const string onnxEnginePath, const unsigned maxBatchsize, const size_t maxWorkspaceSize, const bool fp16, vector<string> tensorNames, vector<vector<unsigned>> tensorDims){   
    this->onnxEnginePath = onnxEnginePath;
    this->maxBatchsize = maxBatchsize;
    this->maxWorkspaceSize = maxWorkspaceSize;
    this->useFP16 = fp16;
    this->useDynamicShape = true;
    this->tensorNames = tensorNames;
    this->tensorDims = tensorDims;

    if (this->maxBatchsize <= 0){
        cerr <<"[ERROR] Max batchsize must be more than 0! \n";
        return false;
    }
    if (!CheckFileIfExist(this->onnxEnginePath)){
        cerr <<"[ERROR] '"<< this->onnxEnginePath << "' not found! \n";
        return false;
    }
    size_t totalDevMem, freeDevMem;
    if (!CudaCheck(cudaMemGetInfo(&freeDevMem, &totalDevMem))) return false;
    if (this->maxWorkspaceSize > freeDevMem) {
        cerr << "[ERROR] Not enough Gpu Memory! Model's WorkspaceSize: " << this->maxWorkspaceSize/1048576 << "MB. Free memory left: " 
             << freeDevMem/1048576 << "MB. \nTry decreasing workspacesize to continue.\n";
        return false;
    }
    if (this->tensorNames.size() != this->tensorDims.size()){
        cerr << "[ERROR] UnMatched size between tensorNames and tensorDims!\n";
        return false;
    }
    for (auto& tensorName : this->tensorNames){
        if (tensorName == ""){
            cerr << "[ERROR] Input tensor name is empty! \n";
            return false;
        }
    }
    for (auto& tensorDim : this->tensorDims){
        if (tensorDims.size() < 1 || tensorDims.size() > 3){
            cerr << "[ERROR] Dimension of tensorDims must >= 1 and <= 3! \n";
            return false;
        }
    }
    return true;
}

Tensor::Tensor(){
	throw ("[ERROR] Un-Initialize tensor!\n");
	abort();
}

Tensor::Tensor(nvinfer1::ICudaEngine* engine, const unsigned bindingIndex){
    if (bindingIndex >= engine->getNbBindings()){
        throw ("[ERROR] bindingIndex is more than engine's binding index");
        abort();
    }
    this->tensorName = engine->getBindingName(bindingIndex);
    this->dims = engine->getBindingDimensions(bindingIndex);
    this->type = engine->getBindingDataType(bindingIndex);
    this->format = engine->getBindingFormat(bindingIndex);

    if (this->type == nvinfer1::DataType::kFLOAT) {
        this->tensorSize = sizeof(float);
    }
    else if (this->type == nvinfer1::DataType::kHALF) {
        this->tensorSize = sizeof(float)/2;
    }
    else if (this->type == nvinfer1::DataType::kINT8) {
        this->tensorSize = sizeof(int8_t);
    }
    else if (this->type == nvinfer1::DataType::kINT32) {
        this->tensorSize = sizeof(int32_t);
    }
    else if (this->type == nvinfer1::DataType::kBOOL) {
        this->tensorSize = sizeof(bool);
    }

    if (this->format == TensorFormat::kLINEAR || this->format == TensorFormat::kCHW2 || this->format == TensorFormat::kCHW4 || this->format == TensorFormat::kCHW16 
        || this->format == TensorFormat::kCHW32|| this->format ==  TensorFormat::kCDHW32|| this->format == TensorFormat::kDLA_LINEAR){
        this->isCHW = true;
    }
    else if (this->format == TensorFormat::kHWC8 || this->format == TensorFormat::kDHWC8 || this->format == TensorFormat::kHWC || this->format == TensorFormat::kDLA_HWC4){
        this->isCHW = false;
    }
    else{
        throw ("[ERROR] Unsupported TensorFormat! Check TensorRT Document 'TensorFormat' to add new format!\n");
        abort();
    }
}

ostream& operator << (ostream& os, const Tensor& x) {
    //Tensor name
    os << "'" << x.tensorName << "' :batchsize";
    //Tensor dims
    for (unsigned i = 1; i < x.dims.nbDims; i++) {
        os << " x " << x.dims.d[i];
    }
    //Tensor type
    if (x.type == nvinfer1::DataType::kFLOAT) {
        os << " (kFLOAT/";
    }
    else if (x.type == nvinfer1::DataType::kHALF) {
        os << " (kHALF/";
    }
    else if (x.type == nvinfer1::DataType::kINT8) {
        os << " (kINT8/";
    }
    else if (x.type == nvinfer1::DataType::kINT32) {
        os << " (kINT32/";
    }
    else if (x.type == nvinfer1::DataType::kBOOL) {
        os << " (kBOOL/";
    }
    //TensorFormat
    if (x.format == TensorFormat::kLINEAR) {
        os << "kLINEAR)";
    }
    else if (x.format == TensorFormat::kCHW2){
        os << "kCHW2)";
    }
    else if (x.format == TensorFormat::kHWC8){
        os << "kHWC8)";
    }
    else if (x.format == TensorFormat::kCHW4){
        os << "kCHW4)";
    }
    else if (x.format == TensorFormat::kCHW16){
        os << "kCHW16)";
    }
    else if (x.format == TensorFormat::kCHW32){
        os << "kCHW32)";
    }
    else if (x.format == TensorFormat::kDHWC8){
        os << "kDHWC8)";
    }
    else if (x.format == TensorFormat::kCDHW32){
        os << "kCDHW32)";
    }
    else if (x.format == TensorFormat::kHWC){
        os << "kHWC)";
    }
    else if (x.format == TensorFormat::kDLA_LINEAR){
        os << "kDLA_LINEAR)";
    }
    else if (x.format == TensorFormat::kDLA_HWC4){
        os << "kDLA_HWC4)";
    }

    return os;
}

nvinfer1::ICudaEngine* LoadOnnxEngine(const ExportConfig exportConfig) {
	string onnxEnginePath = exportConfig.onnxEnginePath;
	unsigned maxWorkspaceSize = exportConfig.maxWorkspaceSize;
	unsigned maxBatchsize = exportConfig.maxBatchsize;
    bool useDynamicShape = exportConfig.useDynamicShape;
    bool useFP16 = exportConfig.useFP16;
    cout << "[INFO] Export info: \n";
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
        auto profile = builder->createOptimizationProfile();
        for (unsigned i = 0; i < exportConfig.tensorNames.size(); i++){
            string inputTensorName = exportConfig.tensorNames.at(i);
            vector<unsigned> tensorDims = exportConfig.tensorDims.at(i);
            
            cout << "\t - Input tensor: '" << inputTensorName << "': BactchSize";
            for (unsigned i = 0; i < tensorDims.size(); i++){
                cout <<" x " << tensorDims.at(i);
            }
            cout << endl;
            
            if (tensorDims.size() == 1){
                profile->setDimensions(inputTensorName.c_str(), nvinfer1::OptProfileSelector::kMIN, nvinfer1::Dims2{1, tensorDims.at(0)});
                profile->setDimensions(inputTensorName.c_str(), nvinfer1::OptProfileSelector::kOPT, nvinfer1::Dims2{max(int(maxBatchsize/2),1), tensorDims.at(0)});
                profile->setDimensions(inputTensorName.c_str(), nvinfer1::OptProfileSelector::kMAX, nvinfer1::Dims2{maxBatchsize, tensorDims.at(0)});
            }
            else if (tensorDims.size() == 2){
                profile->setDimensions(inputTensorName.c_str(), nvinfer1::OptProfileSelector::kMIN, nvinfer1::Dims3{1, tensorDims.at(0), tensorDims.at(1)});
                profile->setDimensions(inputTensorName.c_str(), nvinfer1::OptProfileSelector::kOPT, nvinfer1::Dims3{max(int(maxBatchsize/2),1), tensorDims.at(0), tensorDims.at(1)});
                profile->setDimensions(inputTensorName.c_str(), nvinfer1::OptProfileSelector::kMAX, nvinfer1::Dims3{maxBatchsize, tensorDims.at(0), tensorDims.at(1)});
            }
            else if (tensorDims.size() == 3){
                profile->setDimensions(inputTensorName.c_str(), nvinfer1::OptProfileSelector::kMIN, nvinfer1::Dims4{1, tensorDims.at(0), tensorDims.at(1), tensorDims.at(2)});
                profile->setDimensions(inputTensorName.c_str(), nvinfer1::OptProfileSelector::kOPT, nvinfer1::Dims4{max(int(maxBatchsize/2),1), tensorDims.at(0), tensorDims.at(1), tensorDims.at(2)});
                profile->setDimensions(inputTensorName.c_str(), nvinfer1::OptProfileSelector::kMAX, nvinfer1::Dims4{maxBatchsize, tensorDims.at(0), tensorDims.at(1), tensorDims.at(2)});
            }
        }
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

bool ShowEngineInfo(nvinfer1::ICudaEngine* engine){
	if (engine == nullptr){
		cout << "[ERROR] ICuda Engine is null! \n";
		return false;
	}
	cout << "[INFO] TensorRT Engine Info: \n";
	cout << "\t - Max batchSize: " << engine->getMaxBatchSize() << endl;
	cout << "\t - Engine size: " << engine->getDeviceMemorySize()/(1048576) << " MB (GPU Mem)" << endl; 
	cout << "\t - Tensors: \n";
	for (unsigned i = 0; i < engine->getNbBindings(); i++) {
		Tensor x(engine, i);
		if (engine->bindingIsInput(i)) {
			cout << "\t\t + (Input) " << x << endl;
		}
		else{
			cout << "\t\t + (Output) " << x << endl;
		}
	}
	return true;
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
	if (!ShowEngineInfo(engine)){
		return false;
	}
	engine->destroy();
	return !engineFile.fail();
}


