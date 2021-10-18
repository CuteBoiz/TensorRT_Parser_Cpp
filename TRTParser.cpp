#include "TRTParser.h"

TRTParser::TRTParser() {
	this->engineSize = 0;
	this->maxBatchSize = 0;
	this->engine = nullptr;
	this->context = nullptr;
}

TRTParser::~TRTParser() {
	this->inputTensors.clear();
	this->outputTensors.clear();
	this->context->destroy();
	this->engine->destroy();
}

size_t TRTParser::GetDimensionSize(const nvinfer1::Dims& dims) {	
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
		if (!CudaCheck(cudaMemGetInfo(&freeDevMem, &totalDevMem))) return false;
		if (this->engineSize > freeDevMem) {
			cerr << "[ERROR] Not enough Gpu Memory! Model's WorkspaceSize: " << this->engineSize/1048576 << "MB. Free memory left: " << freeDevMem/1048576 <<"MB. \nReduce workspacesize to continue.\n";
			return false;
		}
		for (unsigned i = 0; i < this->engine->getNbBindings(); i++) {
			Tensor tensor(this->engine, i);
			if (this->engine->bindingIsInput(i)) {
				this->inputTensors.emplace_back(tensor);
			}
			else {
				this->outputTensors.emplace_back(tensor);
			}	
		}
		if (this->inputTensors.empty() || this->outputTensors.empty()) {
			cerr << "[ERROR] Expect at least one input and one output for network \n";
			return false;
		}
		if (!ShowEngineInfo(engine)){
			return false;
		}
		return true;
	}
}


bool TRTParser::AllocateImageInput(vector<cv::Mat> images, float* gpuInputBuffer, const unsigned inputIndex) {
	if (inputIndex >= this->inputTensors.size()) {
		cerr << "[ERROR] inputIndex is greater than number of inputTensor's index!\n";
		return false;
	}
	unsigned imgH, imgW, imgC;
	if (!inputTensors.at(inputIndex).isCHW){
		imgH = this->inputTensors.at(inputIndex).dims.d[1];
		imgW = this->inputTensors.at(inputIndex).dims.d[2];
		imgC = this->inputTensors.at(inputIndex).dims.d[3];
	}
	else { 
		imgH = this->inputTensors.at(inputIndex).dims.d[2];
		imgW = this->inputTensors.at(inputIndex).dims.d[3];
		imgC = this->inputTensors.at(inputIndex).dims.d[1];
	}
	auto imageSize = cv::Size(imgW, imgH);
	for (unsigned i = 0; i < images.size(); i++) {
		//Upload images to GPU
		cv::Mat image = images.at(i);
		if (image.empty()) {
			cerr << "[ERROR] Could not load Input image!! \n";
			return false;
		}
		cv::cuda::GpuMat gpuImage;
		gpuImage.upload(image);
		//Resize
		cv::cuda::GpuMat gpuResized, gpuImageFloat;
		cv::cuda::resize(gpuImage, gpuResized, imageSize, 0, 0, cv::INTER_AREA);
		//Normalize
		gpuResized.convertTo(gpuImageFloat, CV_32FC3, 1.f / 255.f);
		cv::cuda::subtract(gpuImageFloat, cv::Scalar(0.485f, 0.456f, 0.406f), gpuImageFloat, cv::noArray(), -1);
		cv::cuda::divide(gpuImageFloat, cv::Scalar(0.229f, 0.224f, 0.225f), gpuImageFloat, 1, -1);
		//Allocate
		if (imgC == 3){
			if (this->inputTensors.at(inputIndex).isCHW){
				try {
					cv::Mat test(gpuImageFloat);
					vector< cv::cuda::GpuMat > chw;
					for (unsigned j = 0; j < imgC; j++) {
						chw.emplace_back(cv::cuda::GpuMat(imageSize, CV_32FC1, gpuInputBuffer + (i*imgC+j)*imgW*imgH));
					}
					cv::cuda::split(gpuImageFloat, chw);
				}
				catch (cv::Exception& e) {
    				cout << "[ERROR] [OpenCV] Exception caught: " << e.what();
    				return false;
				}
			}
			else {
				size_t inputBufferSize = this->GetDimensionSize(this->inputTensors.at(inputIndex).dims);
				if (!CudaCheck(cudaMemcpyAsync(gpuInputBuffer, gpuImageFloat.ptr<float>(), inputBufferSize*sizeof(float), cudaMemcpyDeviceToDevice))) return false;
			}
		}
		else if (imgC == 1) {
			if (!CudaCheck(cudaMemcpyAsync(gpuInputBuffer, gpuImageFloat.ptr<float>(), gpuImageFloat.rows*gpuImageFloat.step, cudaMemcpyDeviceToDevice))) return false;
		}
		else {
			cerr << "[ERROR] Undefined image channel!\n";
			return false;
		}
	}
	return true;
}

bool TRTParser::AllocateNonImageInput(void *pData, float* gpuInputBuffer, const unsigned inputIndex){
	if (inputIndex >= this->inputTensors.size()){
		cerr << "[ERROR] inputIndex is greater than number of inputTensor's index!\n";
		return false;
	}
	size_t inputBufferSize = this->GetDimensionSize(this->inputTensors.at(inputIndex).dims);
	if (!CudaCheck(cudaMemcpyAsync(gpuInputBuffer, pData, inputBufferSize * this->inputTensors.at(inputIndex).tensorSize, cudaMemcpyHostToDevice))) return false;
	return true;
}


vector<float> TRTParser::PostprocessResult(float *gpuOutputBuffer, const unsigned batchSize, const unsigned outputIndex, const bool softMax) {
	if (outputIndex >= this->outputTensors.size()){
		throw std::overflow_error("[ERROR] outputIndex is greater than number of outputTensor's index!\n");
	}
	//Create CPU buffer.
	size_t outputSize = this->GetDimensionSize(this->outputTensors.at(outputIndex).dims)/this->outputTensors.at(outputIndex).dims.d[0];
	vector< float > cpu_output(outputSize * batchSize);

	//Transfer data from GPU buffer to CPU buffer.
	if (!CudaCheck(cudaMemcpyAsync(cpu_output.data(), gpuOutputBuffer, cpu_output.size() * this->outputTensors.at(outputIndex).tensorSize, cudaMemcpyDeviceToHost))) {
		throw std::overflow_error("[ERROR] Get data from device to host failure!\n");
		abort();
	}
	//Preform a softmax to classifier output.
	if (softMax){
		//Apply Exponentialfunction to result
		transform(cpu_output.begin(), cpu_output.end(), cpu_output.begin(), [](float val) {return exp(val);});

		unsigned nrofDims = this->outputTensors.at(outputIndex).dims.nbDims;

		//2D: batchsize x K
		if (nrofDims == 2) { 
			unsigned K = this->outputTensors.at(outputIndex).dims.d[1];
	    	for (unsigned k = 0; k < batchSize; k++){
	    		float sum = 0;
	    		for (unsigned n = 0; n < K; n++) {
	    			sum += cpu_output.at(k*K + n);
	    		}
	    		for (unsigned n = 0; n < K; n++) {
	 				cpu_output.at(k*K + n) /=  sum;
	 			}
			}
		}
		// 3D: batchsize x L x K
		else if (nrofDims == 3) { 
			unsigned L =  this->outputTensors.at(outputIndex).dims.d[1];
			unsigned K = this->outputTensors.at(outputIndex).dims.d[2];
			for (unsigned l = 0; l < batchSize; l++){
				for (unsigned k = 0; k < L; k++){
		    		float sum = 0;
		    		for (unsigned n = 0; n < K; n++) {
		    			sum += cpu_output.at(l*(K*L) + k*K + n);
		    		}
		    		for (unsigned n = 0; n < K; n++) {
		 				cpu_output.at(l*(K*L) +  k*K + n) /=  sum;
		 			}
		 		}
			}
		}
		// 4D: batchsize x M x L x K
		else if (nrofDims == 4) { 
			unsigned M = this->outputTensors.at(outputIndex).dims.d[1];
			unsigned L =  this->outputTensors.at(outputIndex).dims.d[2];
			unsigned K = this->outputTensors.at(outputIndex).dims.d[3];
			for (unsigned m = 0; m < batchSize; m++) {
				for (unsigned l = 0; l < M; l++) {
					for (unsigned k = 0; k < L; k++){
			    		float sum = 0;
			    		for (unsigned n = 0; n < K; n++) {
			    			sum += cpu_output.at(m*(M*L*M) + l*(K*L) + k*K + n);
			    		}
			    		for (unsigned n = 0; n < K; n++) {
			 				cpu_output.at(m*(M*L*M) + l*(K*L) +  k*K + n) /=  sum;
			 			}
			 		}
				}
			}
		}
		else {
			throw std::overflow_error("[ERROR] Unsupported softmax for "+ to_string(nrofDims) +"D output!\n");
			abort();
		}
	}

	return cpu_output;
}


bool TRTParser::Inference(vector<cv::Mat> images, const bool softMax) {
	unsigned batchSize = images.size();
	unsigned nrofInputs = this->inputTensors.size();
	if (batchSize > this->maxBatchSize){
		cerr << "[ERROR] Batch size must be smaller or equal " << this->maxBatchSize << endl;
		return false;
	}

	//Create buffer on GPU device
	vector< void* > buffers(this->engine->getNbBindings());
	for (unsigned i = 0; i < this->engine->getNbBindings(); i++) {
		auto dims = this->engine->getBindingDimensions(i);
		size_t bindingSize;
		if (this->engine->bindingIsInput(i)){
			bindingSize = this->GetDimensionSize(dims) * this->inputTensors.at(i).tensorSize;
		}
		else{
			bindingSize = this->GetDimensionSize(dims) * this->outputTensors.at(i - nrofInputs).tensorSize;
		}
		if (!CudaCheck(cudaMalloc(&buffers[i], bindingSize))) return false;
	}

	//Allocate data to GPU. 
	//If you have multiple inputs add AllocateImageInput or AllocateNonImageInput with coresponding inputIndex
	if (!this->AllocateImageInput(images, (float*)buffers[0], 0)){
		cerr << "[ERROR] Allocate Input error!\n";
		return false;
	}

	if (nrofInputs > 1) {
		cerr << "[ERROR] Your must add AllocateImageInput or AllocateNonImageInput with coresponding inputIndex for other inputs above / add data for Inference()'s arguments then remove this condition at " << __FILE__ << ":" << __LINE__<< " to continue!\n";
		return false;
	}
	//Model Inference on GPU
	this->context->enqueueV2(buffers.data(), 0, nullptr);

	//Transfer result from GPU to CPU
	for (unsigned i = 0; i < this->outputTensors.size(); i++) {
		vector<float> result;
		unsigned lastLayerSize = this->outputTensors.at(i).dims.d[int(this->outputTensors.at(i).dims.nbDims-1)];

		cout << "'"<<this->outputTensors.at(i).tensorName << "':\n";
		try {
			result = this->PostprocessResult((float *)buffers[i+nrofInputs], batchSize, i, softMax);
		}
		catch (exception& err) {
			cerr << err.what();
			return false;
		}

		unsigned nrofDims = this->outputTensors.at(i).dims.nbDims;
		//2D: batchsize x K
		if (nrofDims == 2) { 
			unsigned K = this->outputTensors.at(i).dims.d[1];
			for (unsigned k = 0; k < batchSize; k++){
		    	for (unsigned n = 0; n < K; n++){
		    		cout << result.at(k*K + n) << " ";
				}
				cout << endl;
			}
		}
		// 3D: batchsize x L x K
		else if (nrofDims == 3) { 
			unsigned L = this->outputTensors.at(i).dims.d[1];
			unsigned K =  this->outputTensors.at(i).dims.d[2];
			for (unsigned l = 0; l < batchSize; l++){
				for (unsigned k = 0; k < L; k++){
					for (unsigned n = 0; n < K; n++){
						cout << result.at(l*(K*L) +  k*K + n) << " ";
			 		}
			 		cout << endl;
				}
				cout << endl;
			}
		}
		// 4D: batchsize x M x L x K
		else if (nrofDims == 4) { 
			unsigned M = this->outputTensors.at(i).dims.d[1];
			unsigned L =  this->outputTensors.at(i).dims.d[2];
			unsigned K = this->outputTensors.at(i).dims.d[3];
			for (unsigned m = 0; m < batchSize; m++) {
				for (unsigned l = 0; l < M; l++) {
					for (unsigned k = 0; k < L; k++){
			    		for (unsigned n = 0; n < K; n++) {
			    			cout << result.at(m*(M*L*M) + l*(K*L) + k*K + n) << " ";
			    		}
			    		cout << endl;
			 		}
			 		cout << endl;
				}
				cout << endl;
			}
		}
		else{
			for (unsigned j = 0; j < result.size(); j++){
				cout <<result.at(j) << " ";
			}
			cout << endl;
		}
		result.clear();
	}

	//Deallocate memory to avoid memory leak
	for (void* buf : buffers) {
		if (!CudaCheck(cudaFree(buf))) return false;
	}
	return true;
}
