#include "utils.h"


bool CheckFileIfExist(const string filePath){
	ifstream f(filePath, ios::binary);
	if (!f.good()){
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

nvinfer1::ICudaEngine* LoadOnnxEngine(const string onnxPath, const unsigned max_batchsize,
									const bool fp16, const string input_tensor_name,
									const vector<unsigned> dimension,
									const bool dynamic_shape) {
	//Parse engine (Check onnx support operators if parse wasn't success)
	nvinfer1::IBuilder* builder{ nvinfer1::createInferBuilder(gLogger) };
	const auto flag = 1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
	nvinfer1::INetworkDefinition* network = builder->createNetworkV2(flag);
	TRTUniquePtr<nvonnxparser::IParser> parser{ nvonnxparser::createParser(*network, gLogger) };
	if (!parser->parseFromFile(onnxPath.c_str(), static_cast<int>(nvinfer1::ILogger::Severity::kWARNING))){
		cerr << "[ERROR]: Could not parse onnx engine \n";
		return nullptr;
	}

	//Building enigne
	TRTUniquePtr<nvinfer1::IBuilderConfig> config{ builder->createBuilderConfig() };
	config->setMaxWorkspaceSize(MAX_WORKSPACE_SIZE);
	if (fp16){
		if (builder->platformHasFastFp16()){
			cout << "[INFO]: Model exporting in FP16 Fast Mode\n";
			config->setFlag(nvinfer1::BuilderFlag::kFP16);
		}
		else{
			cout << "[INFO]: This system does not support FP16 fast mode.\nExporting model in FP32 Mode.\n";
		}
	}
	else{
		cout << "[INFO]: Model exporting in FP32 Mode.\n";
	}
	cout << "[INFO]: Max inference batchsize " << max_batchsize << endl;
	builder->setMaxBatchSize(max_batchsize);

	if (dynamic_shape){
		if (input_tensor_name == ""){
			cerr << "[ERROR]: Input tensor name is empty \n";
			return nullptr;
		}
		if (dimension.size() != 3){
			cerr << "[ERROR]: Dimension of dynamic shape must be 3 \n";
			return nullptr;
		}
		auto profile = builder->createOptimizationProfile();
		profile->setDimensions(input_tensor_name.c_str(), nvinfer1::OptProfileSelector::kMIN, nvinfer1::Dims4{1, dimension.at(0), dimension.at(1), dimension.at(2)});
		profile->setDimensions(input_tensor_name.c_str(), nvinfer1::OptProfileSelector::kOPT, nvinfer1::Dims4{max(int(max_batchsize/2),1), dimension.at(0), dimension.at(1), dimension.at(2)});
		profile->setDimensions(input_tensor_name.c_str(), nvinfer1::OptProfileSelector::kMAX, nvinfer1::Dims4{max_batchsize, dimension.at(0), dimension.at(1), dimension.at(2)});
		config->addOptimizationProfile(profile);
	}
	return builder->buildEngineWithConfig(*network, *config);
}


bool ExportOnnx2Trt(const string onnxEnginePath, const unsigned max_batchsize, 
					const bool fp16, const string input_tensor_name, 
					const vector<unsigned> dimension, const bool dynamic_shape)
{
	//Check condition
	if (!CheckFileIfExist(onnxEnginePath)) {
		cout << "[ERROR]: " << onnxEnginePath << " not found! \n";
		return false;
	}
	else {
		cout << "[INFO]: " << onnxEnginePath << " found!, Converting to TensorRT Engine \n";
	}
	size_t lastindex = onnxEnginePath.find_last_of(".");
	string TRTFilename = onnxEnginePath.substr(0, lastindex) + ".trt";
	if (CheckFileIfExist(TRTFilename)) {
		cout << "[ERROR]: " << TRTFilename << " is already exist! \n";
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
		cerr << "[ERROR]: Could not open engine file: " << TRTFilename << endl;
		remove(TRTFilename.c_str());
		return false;
	}
	nvinfer1::ICudaEngine* engine = LoadOnnxEngine(onnxEnginePath, max_batchsize, fp16, input_tensor_name, dimension, dynamic_shape);
	if (engine == nullptr) {
		cerr << "[ERROR]: Could not get onnx engine" << endl;
		remove(TRTFilename.c_str());
		return false;
	}

	//Seialize tensorrt engine.
	TRTUniquePtr<nvinfer1::IHostMemory> serializedEngine{ engine->serialize() };
	if (serializedEngine == nullptr)	{
		remove(TRTFilename.c_str());
		cerr << "[ERROR]: Could not serialized engine \n";
		return false;
	}
	engineFile.write(static_cast<char*>(serializedEngine->data()), serializedEngine->size());
	engine->destroy();
	return !engineFile.fail();
}


