#include "TRTParser.h"

size_t TRTParser::GetSizeByDim(const nvinfer1::Dims& dims)
{
	size_t size = 1;
	for (unsigned i = 1; i < dims.nbDims; i++){
		size *= dims.d[i];
	}
	return size;
}

nvinfer1::ICudaEngine* TRTParser::LoadTRTEngine(const string enginePath) {
	ifstream gieModelStream(enginePath, ios::binary);
	if (!gieModelStream.good()){
		cerr << "[ERROR] Could not read engine! \n";
		gieModelStream.close();
		return nullptr;
	}
	gieModelStream.seekg(0, ios::end);
	size_t modelSize = gieModelStream.tellg();
	gieModelStream.seekg(0, ios::beg);

	void* modelData = malloc(modelSize);
	if(!modelData)
	{
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


TRTParser::TRTParser() {
	engine = nullptr;
	context = nullptr;
}

bool TRTParser::Init(const string enginePath) {
	this->engine = this->LoadTRTEngine(enginePath);
	if (this->engine == nullptr) {
		return false;
	}
	else{
		for (unsigned i = 0; i < this->engine->getNbBindings(); i++)
		{
			auto dims = this->engine->getBindingDimensions(i);
			if (this->engine->bindingIsInput(i))
			{
				this->input_dims.emplace_back(dims);
				if (dims.d[3] == 3 || dims.d[3] == 1) {
					this->imgH = dims.d[1];
					this->imgW = dims.d[2];
					this->imgC = dims.d[3];
					this->is_channel_first = false;
				}
				else if (dims.d[1] == 3 || dims.d[1] == 1) { 
					this->imgH = dims.d[2];
					this->imgW = dims.d[3];
					this->imgC = dims.d[1];
					this->is_channel_first = true;
				}
				else {
					cerr << "[ERROR] Input shape not valid!\n";
					return false;
				}
			}
			else{
				output_dims.emplace_back(dims);
			}
		}
		if (input_dims.empty() || output_dims.empty()){
			cerr << "[ERROR] Expect at least one input and one output for network \n";
			return false;
		}
			/*
		If has more than 1 input duplicate the below preprocessImage 
		with (float*)buffers[1], (float*)buffers[2], ....
		coresponding with number of your network inputs.
		*/
		if (this->input_dims.size() > 1){
			cerr << "[ERROR] [Unsupported mutiple-inputs] Your must use CudaMalloc() for other inputs then delete this condition to continue\n";
			return false;
		}
		this->maxBatchSize = this->engine->getMaxBatchSize();
		this->context = this->engine->createExecutionContext();
		return true;
	}
}

TRTParser::~TRTParser() {
	this->input_dims.clear();
	this->output_dims.clear();
	this->engine->destroy();
	this->context->destroy();
}


void TRTParser::PreprocessImage(vector<cv::Mat> images, float* gpu_input) {
	
	auto input_size = cv::Size(this->imgW, this->imgH);
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
		cv::cuda::resize(gpu_frame, resized, input_size, 0, 0, cv::INTER_AREA);
		//Normalize
		cv::cuda::GpuMat flt_image;
		resized.convertTo(flt_image, CV_32FC3, 1.f / 255.f);
		cv::cuda::subtract(flt_image, cv::Scalar(0.485f, 0.456f, 0.406f), flt_image, cv::noArray(), -1);
		cv::cuda::divide(flt_image, cv::Scalar(0.229f, 0.224f, 0.225f), flt_image, 1, -1);
		//Allocate
		if (this->imgC == 3){
			if (is_channel_first){
				vector< cv::cuda::GpuMat > chw;
				for (unsigned j = 0; j < this->imgC; j++){
					chw.emplace_back(cv::cuda::GpuMat(input_size, CV_32FC1, gpu_input + (i*this->imgC+j)*this->imgW*this->imgH));
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

vector<float> TRTParser::PostprocessResult(float *gpu_output, const unsigned batch_size, const unsigned output_size, const bool softMax) {
	vector< float > cpu_output(output_size * batch_size);
	cudaMemcpy(cpu_output.data(), gpu_output, cpu_output.size() * sizeof(float), cudaMemcpyDeviceToHost);
	if (softMax){
		transform(cpu_output.begin(), cpu_output.end(), cpu_output.begin(), [](float val) {return exp(val);});
    	for (unsigned i = 0; i < batch_size; i++){
			float sum = 0;
			for (unsigned j = 0; j < output_size; j++){
				sum += cpu_output.at(i*output_size + j);
			}
			for (unsigned k = 0; k < output_size; k++){
				cpu_output.at(i*output_size + k) /=  sum;
			}
		}
	}
	return cpu_output;
}

bool TRTParser::Inference(vector<cv::Mat> images, const bool softMax) {
	unsigned batch_size = images.size();
	unsigned nrof_input = this->input_dims.size();
	if (batch_size > this->maxBatchSize){
		cerr << "[ERROR] Batch size must be smaller or equal " << this->engine->getMaxBatchSize() << endl;
		return false;
	}
	vector< void* > buffers(this->engine->getNbBindings());

	for (unsigned i = 0; i < this->engine->getNbBindings(); i++)
	{
		auto dims = this->engine->getBindingDimensions(i);
		auto binding_size = this->GetSizeByDim(dims) * batch_size * sizeof(float);
		cudaMalloc(&buffers[i], binding_size);
	}
	
	this->PreprocessImage(images, (float*)buffers[0]);

	this->context->enqueueV2(buffers.data(), 0, nullptr);

	for (unsigned i = 0; i < this->output_dims.size(); i++){
		vector<float> result;
		unsigned output_size = this->GetSizeByDim(output_dims[i]);
		cout << output_size << endl;
		
		result = this->PostprocessResult((float *)buffers[i+nrof_input], batch_size, output_size, softMax);

		cout << "[INFO] Result: \n";
		for (unsigned j = 0; j < batch_size; j++){
			for (unsigned k = 0; k < output_size; k++){
				cout << result.at(j*output_size + k) << ' ';
			}
			cout << endl;
		}
	}
	for (void* buf : buffers)
	{
		cudaFree(buf);
	}
	return true;
}
