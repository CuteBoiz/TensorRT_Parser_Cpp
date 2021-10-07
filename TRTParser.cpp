#include "TRTParser.h"

TRTParser::TRTParser() {
	this->imgH = 0;
	this->imgW = 0;
	this->imgC = 0;
	this->engineSize = 0;
	this->maxBatchSize = 0;
	this->isCHW = false;
	this->engine = nullptr;
	this->context = nullptr;
}

TRTParser::~TRTParser() {
	this->inputDims.clear();
	this->outputDims.clear();
	this->engine->destroy();
	this->context->destroy();
}

size_t TRTParser::GetSizeByDim(const nvinfer1::Dims& dims)
{
	size_t size = 1;
	for (unsigned i = 0; i < dims.nbDims; i++) {
		size *= dims.d[i];
	}
	return size;
}

nvinfer1::ICudaEngine* TRTParser::LoadTRTEngine(const string enginePath) {
	ifstream gieModelStream(enginePath, ios::binary);
	if (!gieModelStream.good()) {
		cerr << "[ERROR] Could not read engine! \n";
		gieModelStream.close();
		return nullptr;
	}
	gieModelStream.seekg(0, ios::end);
	size_t modelSize = gieModelStream.tellg();
	gieModelStream.seekg(0, ios::beg);

	void* modelData = malloc(modelSize);
	if(!modelData) {
		cerr << "[ERROR] Could not allocate memory for onnx engine! \n";
		gieModelStream.close();
		return nullptr;
	}
	gieModelStream.read((char*)modelData, modelSize);
	gieModelStream.close();

	nvinfer1::IRuntime* runtime = nvinfer1::createInferRuntime(gLogger);
	if (runtime == nullptr) {
		cerr << "[ERROR] Could not create InferRuntime! \n";
		return nullptr;
	}
	return runtime->deserializeCudaEngine(modelData, modelSize, nullptr);
}


bool TRTParser::Init(const string enginePath) {
	this->engine = this->LoadTRTEngine(enginePath);
	if (this->engine == nullptr) {
		return false;
	}
	else{
		this->maxBatchSize = this->engine->getMaxBatchSize();
		this->context = this->engine->createExecutionContext();
		this->engineSize = this->engine->getDeviceMemorySize();
		size_t totalDevMem, freeDevMem;
		cudaMemGetInfo(&freeDevMem, &totalDevMem);
		if (this->engineSize > freeDevMem) {
			cerr << "[ERROR] Not enough Gpu Memory! Model's WorkspaceSize: " << this->engineSize/1048576 << "MB. Free memory left: " << freeDevMem/1048576 <<"MB. \nReduce workspacesize to continue.\n";
			return false;
		}

		cout << "[INFO] TensorRT Engine Info: \n";
		cout << "\t - Max batchSize: " << this->maxBatchSize << endl;
		cout << "\t - Engine size: " << this->engine->getDeviceMemorySize()/(1048576) << " MB (GPU Mem)" << endl; 
		cout << "\t - Tensors: \n";
		for (unsigned i = 0; i < this->engine->getNbBindings(); i++) {
			string tensorName;
			auto dims = this->engine->getBindingDimensions(i);
			if (this->engine->bindingIsInput(i)) {
				if (dims.d[3] == 3 || dims.d[3] == 1) {
					this->imgH = dims.d[1];
					this->imgW = dims.d[2];
					this->imgC = dims.d[3];
					this->isCHW = false;
				}
				else if (dims.d[1] == 3 || dims.d[1] == 1) { 
					this->imgH = dims.d[2];
					this->imgW = dims.d[3];
					this->imgC = dims.d[1];
					this->isCHW = true;
				}
				else {
					cerr << "[ERROR] Input shape not valid! (If you used an non-image input, remove this condition! \n";
					return false;
				}
				cout << "\t\t + (Input) '" << this->engine->getBindingName(i) << "': batchSize";
				this->inputDims.emplace_back(dims);
			}
			else {
				cout << "\t\t + (Output) '" << this->engine->getBindingName(i) << "': batchSize";
				this->outputDims.emplace_back(dims);
			}
			for (unsigned j = 1; j < dims.nbDims; j++) {
				cout << " x " << dims.d[j];
			}
			cout << endl;	
		}
		if (this->inputDims.empty() || this->outputDims.empty()) {
			cerr << "[ERROR] Expect at least one input and one output for network \n";
			return false;
		}
		if (this->inputDims.size() > 1) {
			cerr << "[ERROR] [Unsupported mutiple-inputs] Your must use CudaMalloc() for other inputs then remove this condition to continue\n";
			return false;
		}
		return true;
	}
}


void TRTParser::PreprocessImage(vector<cv::Mat> images, float* gpu_input) {
	auto imageSize = cv::Size(this->imgW, this->imgH);
	for (unsigned i = 0; i < images.size(); i++){
		//Upload images to GPU
		cv::Mat image = images[i];
		if (image.empty()) {
			cerr << "[ERROR] Could not load Input image!! \n";
			return;
		}
		cv::cuda::GpuMat gpu_frame;
		gpu_frame.upload(image);
		//Resize
		cv::cuda::GpuMat resized;
		cv::cuda::resize(gpu_frame, resized, imageSize, 0, 0, cv::INTER_AREA);
		//Normalize
		cv::cuda::GpuMat flt_image;
		resized.convertTo(flt_image, CV_32FC3, 1.f / 255.f);
		cv::cuda::subtract(flt_image, cv::Scalar(0.485f, 0.456f, 0.406f), flt_image, cv::noArray(), -1);
		cv::cuda::divide(flt_image, cv::Scalar(0.229f, 0.224f, 0.225f), flt_image, 1, -1);
		//Allocate
		if (this->imgC == 3){
			if (this->isCHW){
				vector< cv::cuda::GpuMat > chw;
				for (unsigned j = 0; j < this->imgC; j++){
					chw.emplace_back(cv::cuda::GpuMat(imageSize, CV_32FC1, gpu_input + (i*this->imgC+j)*this->imgW*this->imgH));
				}
				cv::cuda::split(flt_image, chw);
			}
			else{
				cout << "[ERROR] Does not support channels last yet!";
				exit(-1);
			}
		}
		else if (this->imgC == 1){
			cudaMemcpyAsync(gpu_input, flt_image.ptr<float>(), flt_image.rows*flt_image.step, cudaMemcpyDeviceToDevice);
		}
	}
}

vector<float> TRTParser::PostprocessResult(float *gpu_output, const unsigned batchSize, const unsigned outputSize, const bool softMax) {
	vector< float > cpu_output(outputSize * batchSize);
	cudaMemcpyAsync(cpu_output.data(), gpu_output, cpu_output.size() * sizeof(float), cudaMemcpyDeviceToHost);
	if (softMax){
		transform(cpu_output.begin(), cpu_output.end(), cpu_output.begin(), [](float val) {return exp(val);});
    	for (unsigned i = 0; i < batchSize; i++){
			float sum = 0;
			for (unsigned j = 0; j < outputSize; j++){
				sum += cpu_output.at(i*outputSize + j);
			}
			for (unsigned k = 0; k < outputSize; k++){
				cpu_output.at(i*outputSize + k) /=  sum;
			}
		}
	}
	return cpu_output;
}

bool TRTParser::Inference(vector<cv::Mat> images, const bool softMax) {
	unsigned batchSize = images.size();
	unsigned nrofInputs = this->inputDims.size();
	if (batchSize > this->maxBatchSize){
		cerr << "[ERROR] Batch size must be smaller or equal " << this->maxBatchSize << endl;
		return false;
	}
	vector< void* > buffers(this->engine->getNbBindings());

	for (unsigned i = 0; i < this->engine->getNbBindings(); i++) {
		auto dims = this->engine->getBindingDimensions(i);
		auto binding_size = this->GetSizeByDim(dims) * sizeof(float);
		cudaMalloc(&buffers[i], binding_size);
	}
	
	this->PreprocessImage(images, (float*)buffers[0]);

	this->context->enqueueV2(buffers.data(), 0, nullptr);

	for (unsigned i = 0; i < this->outputDims.size(); i++){
		vector<float> result;
		unsigned outputSize = this->GetSizeByDim(outputDims[i])/outputDims[i].d[0];
		result = this->PostprocessResult((float *)buffers[i+nrofInputs], batchSize, outputSize, softMax);

		cout << "Result: \n";
		for (unsigned j = 0; j < batchSize; j++){
			for (unsigned k = 0; k < outputSize; k++){
				cout << result.at(j*outputSize + k) << ' ';
			}
			cout << endl;
		}
	}
	for (void* buf : buffers) {
		cudaFree(buf);
	}
	return true;
}
