#include "onnxparser.h"

class Logger : public nvinfer1::ILogger
{
public:
    void log(Severity severity, const char* msg) override {
        if ((severity == Severity::kERROR) || (severity == Severity::kINTERNAL_ERROR)) {
            std::cout << msg << "n";
        }
    }
} gLogger;



size_t getSizeByDim(const nvinfer1::Dims& dims)
{
    size_t size = 1;
    for (size_t i = 0; i < dims.nbDims; ++i)
    {
        size *= dims.d[i];
    }
    return size;
}

OnnxParser::OnnxParser(string model_path, int batch_sz = 1){
	this->batch_size = batch_sz;
	nvinfer1::IBuilder*builder{nvinfer1::createInferBuilder(gLogger)};
    const auto explicitBatch = 1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    nvinfer1::INetworkDefinition* network{builder->createNetworkV2(explicitBatch)};
 
    TRTUniquePtr<nvonnxparser::IParser> parser{nvonnxparser::createParser(*network, gLogger)};
    TRTUniquePtr<nvinfer1::IBuilderConfig> config{builder->createBuilderConfig()};
    // parse ONNX
    if (!parser->parseFromFile(model_path.c_str(), static_cast< int >(nvinfer1::ILogger::Severity::kINFO)))
    {
        std::cerr << "ERROR: could not parse the model.\n";
        return;
    }
	// allow TensorRT to use up to 1GB of GPU memory for tactic selection.
	config->setMaxWorkspaceSize(1ULL << 30);
	// use FP16 mode if possible
	if (builder->platformHasFastFp16())
	{
	    config->setFlag(nvinfer1::BuilderFlag::kFP16);
	}
	// we have only one image in batch
	builder->setMaxBatchSize(1);
    this->engine.reset(builder->buildEngineWithConfig(*network, *config));
    this->context.reset(this->engine->createExecutionContext());
}
OnnxParser::~OnnxParser(){

}
void OnnxParser::inference(cv::Mat image){
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
void OnnxParser::preprocessImage(cv::Mat frame, float* gpu_input, const nvinfer1::Dims& dims){
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
void OnnxParser::postprocessResults(float *gpu_output, const nvinfer1::Dims &dims){
	// copy results from GPU to CPU
    std::vector< float > cpu_output(getSizeByDim(dims) * this->batch_size);
    cudaMemcpy(cpu_output.data(), gpu_output, cpu_output.size() * sizeof(float), cudaMemcpyDeviceToHost);

    for (int i = 0; i < cpu_output.size(); i ++)
    	cout << cpu_output.at(i) << ' ';
    cout << endl;
}