#include "TRTParser.h"


TRTParser::TRTParser() {
	engine = nullptr;
	context = nullptr;
}

bool TRTParser::init(const string enginePath) {
	this->engine = this->loadTRTEngine(enginePath);
	if (this->engine == nullptr) {
		return false;
	}
	this->context = this->engine->createExecutionContext();
	return true;
}

TRTParser::~TRTParser() {
	this->engine->destroy();
	this->context->destroy();
}

size_t TRTParser::getSizeByDim(const nvinfer1::Dims& dims)
{
	size_t size = 1;
	for (size_t i = 0; i < dims.nbDims; ++i)	{
		size *= dims.d[i];
	}
	return size;
}

nvinfer1::ICudaEngine* TRTParser::loadTRTEngine(const string enginePath) {
	vector<char> trtModelStream_;
	size_t size{ 0 };

	ifstream file(enginePath, ios::binary);
	if (file.good()){
		file.seekg(0, file.end);
		size = file.tellg();
		file.seekg(0, file.beg);
		trtModelStream_.resize(size);
		file.read(trtModelStream_.data(), size);
		file.close();
	}
	else{
		cerr << "ERROR: Could not read engine! \n";
		file.close();
		return nullptr;
	}
	nvinfer1::IRuntime* runtime = nvinfer1::createInferRuntime(gLogger);
	if (runtime == nullptr) {
		cerr << "ERROR: Could not create InferRuntime! \n";
		return nullptr;
	}
	return runtime->deserializeCudaEngine(trtModelStream_.data(), size, nullptr);
}

void TRTParser::preprocessImage(vector<cv::Mat> frame, float* gpu_input, const nvinfer1::Dims& dims) {
	uint32_t input_width, input_height, channels;
	if (dims.d[3] == 3 || dims.d[3] == 1) { //chanel last
		input_width = dims.d[1];
		input_height = dims.d[2];
		channels = dims.d[3];
	}
	else if (dims.d[1] == 3 || dims.d[1] == 1) { //chanel first
		input_width = dims.d[2];
		input_height = dims.d[3];
		channels = dims.d[1];
	}
	else {
		cerr << "Input shape not valid!\n";
		exit(-1);
	}
	auto input_size = cv::Size(input_width, input_height);
	
	for (int i = 0; i < frame.size(); i++) {
		if (frame[i].empty()) {
			cerr << "ERROR: Could not load Input image!! \n";
			return;
		}
		cv::cuda::GpuMat gpu_frame;
		gpu_frame.upload(frame[i]);
		
		//Resize
		cv::cuda::GpuMat resized;
		cv::cuda::resize(gpu_frame, resized, input_size, 0, 0, cv::INTER_NEAREST);
		//Normalize
		cv::cuda::GpuMat flt_image;
		resized.convertTo(flt_image, CV_32FC3, 1.f / 255.f);
		cv::cuda::subtract(flt_image, cv::Scalar(0.485f, 0.456f, 0.406f), flt_image, cv::noArray(), -1);
		cv::cuda::divide(flt_image, cv::Scalar(0.229f, 0.224f, 0.225f), flt_image, 1, -1);
		//Allocate
		if (channels == 3){
			vector< cv::cuda::GpuMat > chw;
			for (unsigned j = 0; j < channels; j++){
				chw.emplace_back(cv::cuda::GpuMat(input_size, CV_32FC1, gpu_input + (i * channels + j) * input_width * input_height));
			}
			cv::cuda::split(flt_image, chw);
		}
		else if (channels == 1){
			cudaMemcpyAsync(gpu_input, flt_image.ptr<float>(), flt_image.rows*flt_image.step, cudaMemcpyDeviceToDevice);
		}

	}
}
vector<float> TRTParser::postprocessResult(float *gpu_output, const unsigned batch_size, const unsigned output_size,const bool softMax) {
	vector< float > cpu_output(output_size * batch_size);
	cudaMemcpy(cpu_output.data(), gpu_output, cpu_output.size() * sizeof(float), cudaMemcpyDeviceToHost);
	if (softMax){
		std::transform(cpu_output.begin(), cpu_output.end(), cpu_output.begin(), [](float val) {return std::exp(val);});
    	for (unsigned i = 0; i < batch_size; i++){
			float sum = 0;
			for (int j = 0; j < output_size; j++){
				sum += cpu_output.at(i*output_size + j);
			}
			for (int k = 0; k < output_size; k++){
				cpu_output.at(i*output_size + k) /=  sum;
			}
		}
	}
	
	return cpu_output;
}

bool TRTParser::inference(vector<cv::Mat> images, bool softMax) {
	if (images.size() > this->engine->getMaxBatchSize()){
		cerr << "Batch size must be smaller or equal " << this->engine->getMaxBatchSize() << endl;
		return false;
	}
	vector< nvinfer1::Dims > input_dims;
	vector< nvinfer1::Dims > output_dims;
	bool is_input[this->engine->getNbBindings()];
	unsigned nrof_inputs = 0;
	vector< void* > buffers(this->engine->getNbBindings());

	for (unsigned i = 0; i < this->engine->getNbBindings(); ++i)
	{
		auto binding_size = getSizeByDim(this->engine->getBindingDimensions(i)) * images.size() * sizeof(float);
		cudaMalloc(&buffers[i], binding_size);
		if (this->engine->bindingIsInput(i)){
			input_dims.emplace_back(this->engine->getBindingDimensions(i));
			is_input[i] = true;
			nrof_inputs++;
		}
		else{
			output_dims.emplace_back(this->engine->getBindingDimensions(i));
			is_input[i] = false;
		}
	}
	if (input_dims.empty() || output_dims.empty()){
		cerr << "ERROR: Expect at least one input and one output for network \n";
		return false;
	}

	/*
	If has more than 1 input duplicate the below preprocessImage 
	with (float*)buffers[1], (float*)buffers[2], ....
	coresponding with number of your network inputs.
	*/
	if (nrof_inputs > 1){
		cerr << "Your network has more than 1 input\nAdd modify preprocessImage() script then delete this condition code\n";
		return false;
	}
	this->preprocessImage(images, (float*)buffers[0], input_dims[0]); 

	this->context->enqueueV2(buffers.data(), 0, nullptr);

	for (unsigned i = nrof_inputs; i < this->engine->getNbBindings(); i++){
		vector<float> result;
		unsigned output_size = output_dims[i-nrof_inputs].d[1];
		unsigned batch_size = images.size();

		result = this->postprocessResult((float *)buffers[i], batch_size, output_size, softMax);

		cout << "Result: \n";
		for (unsigned j = 0; j < batch_size; j++){
			for (unsigned k = 0; k < output_size; k++){
				cout << result.at(j*output_size + k) << ' ';
			}
			cout << endl;
		}
	}

	input_dims.clear();
	output_dims.clear();
	for (void* buf : buffers)
	{
		cudaFree(buf);
	}
	return true;
}

nvinfer1::ICudaEngine* loadOnnxEngine(const string onnxPath, const unsigned max_batchsize, bool fp16, string input_tensor_name, vector<unsigned> dimension, bool dynamic_shape) {
	nvinfer1::IBuilder*builder{ nvinfer1::createInferBuilder(gLogger) };
	const auto explicitBatch = 1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
	nvinfer1::INetworkDefinition* network{ builder->createNetworkV2(explicitBatch) };

	TRTUniquePtr<nvonnxparser::IParser> parser{ nvonnxparser::createParser(*network, gLogger) };
	TRTUniquePtr<nvinfer1::IBuilderConfig> config{ builder->createBuilderConfig() };
	if (!parser->parseFromFile(onnxPath.c_str(), static_cast<int>(nvinfer1::ILogger::Severity::kINFO))){
		cerr << "ERROR: Could not parse the engine from " << onnxPath << endl;
		return nullptr;
	}
	config->setMaxWorkspaceSize(MAX_WORKSPACE_SIZE);
	
	if (fp16){
		if (builder->platformHasFastFp16()){
			cout << "Exporting model in FP16 Fast Mode\n";
			config->setFlag(nvinfer1::BuilderFlag::kFP16);
		}
		else{
			cout << "This system does not support FP16 fast mode\nExporting model in FP32 Mode\n";
		}
	}
	else{
		cout << "Exporting model in FP32 Mode\n";
	}
	builder->setMaxBatchSize(max_batchsize);

	if (dynamic_shape){
		if (input_tensor_name == ""){
			cerr << "ERROR: Input tensor name is empty \n";
			return nullptr;
		}
		if (dimension.size() != 3){
			cerr << "ERROR: Dimension of dynamic shape must be 3 \n";
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


bool convertOnnx2Trt(const string onnxEnginePath, const unsigned max_batchsize, bool fp16, string input_tensor_name, vector<unsigned> dimension, bool dynamic_shape) {
	if (!checkFileIfExist(onnxEnginePath)) {
		cout << "ERROR: " << onnxEnginePath << " not found! \n";
		return false;
	}
	else {
		cout << onnxEnginePath << " found!, Converting to TensorRT Engine \n";
	}
	size_t lastindex = onnxEnginePath.find_last_of(".");
	string TRTFilename = onnxEnginePath.substr(0, lastindex) + ".trt";
	if (checkFileIfExist(TRTFilename)) {
		cout << TRTFilename << " is already exist! \n";
		return true;
	}

	char buf[BUFSIZ];
	size_t size;
	FILE* source = fopen(onnxEnginePath.c_str(), "rb");
	FILE* dest = fopen(TRTFilename.c_str(), "wb");

	while (size = fread(buf, 1, BUFSIZ, source)) {
		fwrite(buf, 1, size, dest);
	}

	fclose(source);
	fclose(dest);

	std::ofstream engineFile(TRTFilename, std::ios::binary);
	if (!engineFile){
		cerr << "ERROR: Could not open engine file: " << TRTFilename << endl;
		remove(TRTFilename.c_str());
		return false;
	}
	nvinfer1::ICudaEngine* engine = loadOnnxEngine(onnxEnginePath, max_batchsize, fp16, input_tensor_name, dimension, dynamic_shape);
	if (engine == nullptr) {
		cerr << "ERROR: Could not get onnx engine" << endl;
		remove(TRTFilename.c_str());
		return false;
	}
	TRTUniquePtr<nvinfer1::IHostMemory> serializedEngine{ engine->serialize() };
	if (serializedEngine == nullptr)	{
		remove(TRTFilename.c_str());
		cerr << "ERROR: Could not serialized engine \n";
		return false;
	}
	engineFile.write(static_cast<char*>(serializedEngine->data()), serializedEngine->size());
	engine->destroy();
	return !engineFile.fail();
}
