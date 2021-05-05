#include "TRTParser.h"

class Logger : public nvinfer1::ILogger
{
public:
	void log(Severity severity, const char* msg) override {
		if ((severity == Severity::kERROR) || (severity == Severity::kINTERNAL_ERROR)) {
			cout << msg << endl;
		}
	}
	nvinfer1::ILogger& getTRTLogger()
	{
		return *this;
	}
} gLogger;

TRTParser::TRTParser() {
	enginePath = " ";
	engine = nullptr;
	context = nullptr;
}

bool TRTParser::init(string path) {
	this->enginePath = path;

	this->engine = this->getTRTEngine();
	this->context = this->engine->createExecutionContext();
	if (this->engine == nullptr || this->context == nullptr) {
		return false;
	}

	return true;
}

TRTParser::~TRTParser() {
	this->engine->destroy();
	this->context->destroy();
}

size_t TRTParser::getSizeByDim(const nvinfer1::Dims& dims)
{
	size_t size = 1;
	for (size_t i = 0; i < dims.nbDims; ++i)
	{
		size *= dims.d[i];
	}
	return size;
}

nvinfer1::ICudaEngine* getOnnxEngine(string onnxPath) {
	nvinfer1::IBuilder*builder{ nvinfer1::createInferBuilder(gLogger) };
	const auto explicitBatch = 1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
	nvinfer1::INetworkDefinition* network{ builder->createNetworkV2(explicitBatch) };

	TRTUniquePtr<nvonnxparser::IParser> parser{ nvonnxparser::createParser(*network, gLogger) };
	TRTUniquePtr<nvinfer1::IBuilderConfig> config{ builder->createBuilderConfig() };
	if (!parser->parseFromFile(onnxPath.c_str(), static_cast<int>(nvinfer1::ILogger::Severity::kINFO)))
	{
		cerr << "ERROR: Could not parse the engine from " << onnxPath << endl;
		return nullptr;
	}
	config->setMaxWorkspaceSize(MAX_WORKSPACE);
	if (builder->platformHasFastFp16())
	{
		config->setFlag(nvinfer1::BuilderFlag::kFP16);
	}
	builder->setMaxBatchSize(MAX_BATCHSIZE);
	return builder->buildEngineWithConfig(*network, *config);
}

nvinfer1::ICudaEngine* TRTParser::getTRTEngine() {
	string fileExtention = this->enginePath.substr(this->enginePath.find_last_of(".") + 1);
	if (fileExtention == "trt") {
		vector<char> trtModelStream_;
		size_t size{ 0 };

		ifstream file(this->enginePath, ios::binary);
		if (file.good())
		{
			file.seekg(0, file.end);
			size = file.tellg();
			file.seekg(0, file.beg);
			trtModelStream_.resize(size);
			file.read(trtModelStream_.data(), size);
			file.close();
		}
		nvinfer1::IRuntime* runtime = nvinfer1::createInferRuntime(gLogger);
		if (runtime == nullptr) {
			cerr << "ERROR: Could not create InferRuntime! \n";
			return nullptr;
		}
		return runtime->deserializeCudaEngine(trtModelStream_.data(), size, nullptr);
	}
	else {
		cerr << "ERROR: Engine file extension must be .trt \n";
		return nullptr;
	}
}

void TRTParser::preprocessImage(vector<cv::Mat> frame, float* gpu_input, const nvinfer1::Dims& dims) {
	for (int i = 0; i < frame.size(); i++) {
		if (frame[i].empty()) {
			cerr << "ERROR: Could not load Input image!! \n";
			return;
		}
		cv::cuda::GpuMat gpu_frame;
		gpu_frame.upload(frame[i]);
		uint32_t input_width, input_height, channels;
		if (dims.d[3] == 3) { //chanel last
			input_width = dims.d[1];
			input_height = dims.d[2];
			channels = dims.d[3];
		}
		else if (dims.d[1] == 3) { //chanel first
			input_width = dims.d[2];
			input_height = dims.d[3];
			channels = dims.d[1];
		}
		auto input_size = cv::Size(input_width, input_height);
		// resize
		cv::cuda::GpuMat resized;
		cv::cuda::resize(gpu_frame, resized, input_size, 0, 0, cv::INTER_NEAREST);
		//normalize
		cv::cuda::GpuMat flt_image;
		resized.convertTo(flt_image, CV_32FC3, 1.f / 255.f);
		cv::cuda::subtract(flt_image, cv::Scalar(0.485f, 0.456f, 0.406f), flt_image, cv::noArray(), -1);
		cv::cuda::divide(flt_image, cv::Scalar(0.229f, 0.224f, 0.225f), flt_image, 1, -1);
		vector< cv::cuda::GpuMat > chw;
		for (size_t j = 0; j < channels; ++j)
		{
			chw.emplace_back(cv::cuda::GpuMat(input_size, CV_32FC1, gpu_input + (i * 3 + j) * input_width * input_height));
		}
		cv::cuda::split(flt_image, chw);
	}
}
void TRTParser::postprocessResult(float *gpu_output, int size, const nvinfer1::Dims &dims) {
	vector< float > cpu_output(dims.d[1] * size);

	cudaMemcpy(cpu_output.data(), gpu_output, cpu_output.size() * sizeof(float), cudaMemcpyDeviceToHost);

	for (unsigned i = 0; i < cpu_output.size(); i++)
		cout << cpu_output.at(i) << ' ';
	cout << endl;
}

void TRTParser::inference(vector<cv::Mat> images) {

	vector< nvinfer1::Dims > input_dims;
	vector< nvinfer1::Dims > output_dims;
	vector< void* > buffers(this->engine->getNbBindings());

	for (size_t i = 0; i < this->engine->getNbBindings(); ++i)
	{
		auto binding_size = getSizeByDim(this->engine->getBindingDimensions(i)) * images.size() * sizeof(float);
		cudaMalloc(&buffers[i], binding_size);
		if (this->engine->bindingIsInput(i))
			input_dims.emplace_back(this->engine->getBindingDimensions(i));
		else
			output_dims.emplace_back(this->engine->getBindingDimensions(i));
	}
	if (input_dims.empty() || output_dims.empty())
	{
		cerr << "ERROR: Expect at least one input and one output for network \n";
		return {};
	}

	this->preprocessImage(images, (float*)buffers[0], input_dims[0]);
	this->context->enqueue(images.size(), buffers.data(), 0, nullptr);

	this->postprocessResult((float *)buffers[1], images.size(), output_dims[0]);

	input_dims.clear();
	output_dims.clear();
	for (void* buf : buffers)
	{
		cudaFree(buf);
	}
}

bool saveTRTEngine(string onnxEnginePath) {
	ifstream onnxFile(onnxEnginePath, ios::binary);
	if (!onnxFile.good()) {
		cout << "ERROR: " << onnxEnginePath << " not found! \n";
		return false;
	}
	else {
		cout << onnxEnginePath << " found!, Exporting TensorRT Engine \n";
	}
	size_t lastindex = onnxEnginePath.find_last_of(".");
	string TRTFilename = onnxEnginePath.substr(0, lastindex) + ".trt";
	ifstream file(TRTFilename, ios::binary);
	if (file.good()) {
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
	if (!engineFile)
	{
		cerr << "ERROR: Could not open engine file: " << TRTFilename << endl;
		return false;
	}
	nvinfer1::ICudaEngine* engine = getOnnxEngine(onnxEnginePath);
	if (engine == nullptr) {
		cerr << "ERROR: Could not get onnx engine" << endl;
		return false;
	}
	TRTUniquePtr<nvinfer1::IHostMemory> serializedEngine{ engine->serialize() };
	if (serializedEngine == nullptr)
	{
		cerr << "ERROR: Could not serialized engine \n";
		return false;
	}
	engineFile.write(static_cast<char*>(serializedEngine->data()), serializedEngine->size());
	engine->destroy();
	return !engineFile.fail();
}