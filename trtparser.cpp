#include "trtparser.h"

using namespace nvinfer1;


#ifndef LOGGER
#define LOGGER

class Logger1 : public nvinfer1::ILogger
{
public:
    void log(Severity severity, const char* msg) override {
        if ((severity == Severity::kERROR) || (severity == Severity::kINTERNAL_ERROR)) {
            std::cout << msg << "n";
        }
    }
    nvinfer1::ILogger& getTRTLogger()
    {
        return *this;
    }
} gLogger1;
#endif 

size_t TRTParser::getSizeByDim(const nvinfer1::Dims& dims)
{
    size_t size = 1;
    for (size_t i = 0; i < dims.nbDims; ++i)
    {
        size *= dims.d[i];
    }
    return size;
}


TRTParser::TRTParser(string path, int batch_sz = 1){
	this->model_path = path;
	this->batch_size = batch_sz;
	if (this->model_path.substr(this->model_path.find_last_of(".") + 1) == "trt"){
		std::stringstream gieModelStream; 
		gieModelStream.seekg(0, gieModelStream.beg); 
		std::ifstream cache(this->model_path); 
		gieModelStream << cache.rdbuf();
		cache.close(); 
		IRuntime* runtime = createInferRuntime(gLogger1); 
		assert(runtime != nullptr); 
		gieModelStream.seekg(0, std::ios::end);
		const int modelSize = gieModelStream.tellg(); 
		gieModelStream.seekg(0, std::ios::beg);
		void* modelMem = malloc(modelSize); 
		gieModelStream.read((char*)modelMem, modelSize);
		this->engine = runtime->deserializeCudaEngine(modelMem, modelSize, NULL); assert(engine != nullptr);
		this->context = this->engine->createExecutionContext();
		free(modelMem);
	}
	else{
		cerr << "Cannot read " << this->model_path << endl;
		exit(0);
	}

}

TRTParser::~TRTParser(){}

void TRTParser::inference(cv::Mat image){
	//create buffer
	std::vector< nvinfer1::Dims > input_dims; // we expect only one input
	std::vector< nvinfer1::Dims > output_dims; // and one output
	std::vector< void* > buffers(engine->getNbBindings()); // buffers for input and output data
	for (size_t i = 0; i < engine->getNbBindings(); ++i)
	{
	    auto binding_size = getSizeByDim(engine->getBindingDimensions(i)) * batch_size * sizeof(float);
	    cudaMalloc(&buffers[i], binding_size);
	    if (engine->bindingIsInput(i))
	    {
	        input_dims.emplace_back(engine->getBindingDimensions(i));
	    }
	    else
	    {
	        output_dims.emplace_back(engine->getBindingDimensions(i));
	    }
	}
	if (input_dims.empty() || output_dims.empty())
	{
	    std::cerr << "Expect at least one input and one output for network \n";
	    return;
	}
	this->preprocessImage(image, (float*)buffers[0], input_dims[0]);
	this->context->enqueue(batch_size, buffers.data(), 0, nullptr);
	this->postprocessResults((float *) buffers[1], output_dims[0]);
	for (void* buf : buffers)
    {
        cudaFree(buf);
    }
}
void TRTParser::preprocessImage(cv::Mat frame, float* gpu_input, const nvinfer1::Dims& dims){
	if (frame.empty()){
		std::cerr << "Cannot load Input image!! \n";
        return;
	}
	cv::cuda::GpuMat gpu_frame;
	gpu_frame.upload(frame);

	auto input_width = 128;
	auto input_height = 128;
	auto channels = 3;
	auto input_size = cv::Size(input_width, input_height);
	// resize
	cv::cuda::GpuMat resized;
	cv::cuda::resize(gpu_frame, resized, input_size, 0, 0, cv::INTER_NEAREST);

	//normalize
	cv::cuda::GpuMat flt_image;
	resized.convertTo(flt_image, CV_32FC3, 1.f / 255.f);
	cv::cuda::subtract(flt_image, cv::Scalar(0.485f, 0.456f, 0.406f), flt_image, cv::noArray(), -1);
	cv::cuda::divide(flt_image, cv::Scalar(0.229f, 0.224f, 0.225f), flt_image, 1, -1);

	std::vector< cv::cuda::GpuMat > chw;
    for (size_t i = 0; i < channels; ++i)
    {
        chw.emplace_back(cv::cuda::GpuMat(input_size, CV_32FC1, gpu_input + i * input_width * input_height));
    }
    cv::cuda::split(flt_image, chw);
}
void TRTParser::postprocessResults(float *gpu_output, const nvinfer1::Dims &dims){
	// copy results from GPU to CPU
    std::vector< float > cpu_output(getSizeByDim(dims) * this->batch_size);
    cudaMemcpy(cpu_output.data(), gpu_output, cpu_output.size() * sizeof(float), cudaMemcpyDeviceToHost);

    for (int i = 0; i < cpu_output.size(); i ++)
    	cout << cpu_output.at(i) << ' ';
    cout << endl;
}