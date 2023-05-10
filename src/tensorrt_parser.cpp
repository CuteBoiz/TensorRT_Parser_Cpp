#include "tensorrt_parser.h"

TensorRT::TensorRT(){
    this->m_maxBatchSize = 0;
    this->m_engineSize = 0;
    this->m_context = nullptr;
    this->m_engine = nullptr;
}

TensorRT::~TensorRT(){
    this->m_inputTensors.clear();
    this->m_outputTensors.clear();
    delete this->m_context;
    delete this->m_engine;    
    for (void* buf : this->m_buffers) {
        if (!utils::cudaCheck(cudaFree(buf))){
            std::cout << "\033[1;33m[WARNING] Could not deallocate buffer!\033[0m\n";
        }
    }
}

size_t f_getDimsSize(const nvinfer1::Dims dims){
    size_t size = 1;
    for (unsigned i = 0; i < dims.nbDims; i++){
        size *= dims.d[i];
    }
    return size;
}

nvinfer1::ICudaEngine* f_parseTensorRTEngine(const std::string enginePath){
    // Check engine path exist
    if (!utils::checkIsFileAndExist(enginePath)){
        std::cerr << "\033[1;31m[ERROR] '"<< enginePath <<"' not exist!\033[0m\n";
        return nullptr;
    }
    // Load engine binary
    std::ifstream gieModelStream(enginePath, std::ios::binary);
    if (!gieModelStream.good()){
        std::cerr << "\033[1;31m[ERROR] Could not parse engine!\033[0m\n";
        gieModelStream.close();
        return nullptr;
    }
    // Get length of engine binary
    gieModelStream.seekg(0, std::ios::end);
    size_t engineSize = gieModelStream.tellg();
    gieModelStream.seekg(0, std::ios::beg);
    // Allocate memory
    void* engineData = malloc(engineSize);
    if (!engineData){
        std::cerr << "\033[1;31m[ERROR] Could not allocate memory for engine '"<<
            enginePath << "'!\033[0m\n";
        gieModelStream.close();
        return nullptr;
    }
    gieModelStream.read((char*)engineData, engineSize);
    gieModelStream.close();
    // Create tensorrt runtime
    nvinfer1::IRuntime* runtime = nvinfer1::createInferRuntime(gLogger);
    if (runtime == nullptr){
        std::cerr << "\033[1;31m[ERROR] Could not create TensorRT runtime!\033[0m\n";
        return nullptr;
    }
    return runtime->deserializeCudaEngine(engineData, engineSize);
}

bool TensorRT::init(const std::string enginePath){
    // Parse TensorRT engine
    this->m_engine = f_parseTensorRTEngine(enginePath);
    if (this->m_engine == nullptr){
        std::cerr << "\033[1;31m[ERROR] Could not parse '"<< enginePath << "' as TensorRT engine!\033[0m\n";
        return false;
    }
    this->m_maxBatchSize = this->m_engine->getMaxBatchSize();
    this->m_context = this->m_engine->createExecutionContext();
    this->m_engineSize = this->m_engine->getDeviceMemorySize();
    // Check cuda memory
    size_t totalMem=0, freeMem=0;
    if (!utils::cudaCheck(cudaMemGetInfo(&totalMem, &freeMem))){
        std::cerr << "\033[1;31m[ERROR] Could not get cuda memory infomation!\033[0m\n";
        return false;
    }
    if (this->m_engineSize > freeMem){
        std::cerr << "\033[1;31m[ERROR] Not enough Gpu Memory! Engine's workspace size: " <<
             this->m_engineSize/1048576 << "MB. Free memory left: " << freeMem/1048576 <<"MB. \nReduce workspace size to continue.\033[0m\n";
		return false;
    }
    // Get engine bindings
    for (unsigned i =0; i < this->m_engine->getNbBindings(); i++){
        tensor::TensorAttribute tensor(this->m_engine, i);
        if (this->m_engine->bindingIsInput(i)){
            this->m_inputTensors.emplace_back(tensor);
        }
        else{
            this->m_outputTensors.emplace_back(tensor);
        }
    }
    if (this->m_inputTensors.empty() || this->m_outputTensors.empty()){
        std::cerr << "\033[1;31m[ERROR] Expect at least one input and one output for network \033[0m\n";
		return false;
    }
    // Create buffers on GPU
    std::vector< void*> buffers(this->m_engine->getNbBindings());
    for (unsigned i = 0; i < this->m_engine->getNbBindings(); i++) {
        auto dims = this->m_engine->getBindingDimensions(i);
        size_t bindingSize = 0;
        if (this->m_engine->bindingIsInput(i)){
            bindingSize = f_getDimsSize(dims)*this->m_inputTensors[i].size;
        }
        else{
            bindingSize = f_getDimsSize(dims)*this->m_outputTensors[i-this->m_inputTensors.size()].size;
        }
        if (!utils::cudaCheck(cudaMalloc(&buffers[i], bindingSize))){
            throw std::invalid_argument("Could not allocate buffer for '" + std::string(this->m_engine->getBindingName(i)) + "' !");
        }
    }
    this->m_buffers = buffers;
    // Show engine info
    try{
        std::cout << this->m_engine;
    }
    catch (std::exception& e){
        std::cerr << "\033[1;31m[ERROR] " << e.what() << "\033[0m\n";
        return false;
    }
    
    return true;
}

bool TensorRT::allocateImage(std::vector<cv::Mat> images, float* buffer, const unsigned index){
    unsigned imgH=0, imgW=0, imgC=0;
    if (this->m_inputTensors[index].isCHW){
        imgC = this->m_inputTensors[index].shape[1];
        imgH = this->m_inputTensors[index].shape[2];
        imgW = this->m_inputTensors[index].shape[3];
    }
    else{
        imgH = this->m_inputTensors[index].shape[1];
        imgW = this->m_inputTensors[index].shape[2];
        imgC = this->m_inputTensors[index].shape[3];
    }
    auto imgsz = cv::Size(imgW, imgH);
    for (unsigned i = 0; i < images.size(); i++){
        // Upload to gpu
        if (images[i].empty()){
            std::cerr << "\033[1;31m[ERROR] Input image is empty! \033[0m\n";
            return false;
        }
        cv::cuda::GpuMat gImage;
        gImage.upload(images[i]);
        // Resize
        cv::cuda::resize(gImage, gImage, imgsz, 0, 0, cv::INTER_AREA);
        if (imgC == 3){
            // Normalize
            gImage.convertTo(gImage, CV_32FC3, 1.f/255.f);
            cv::cuda::subtract(gImage, cv::Scalar(0.485f, 0.456f, 0.406f), gImage, cv::noArray(), -1);
            cv::cuda::divide(gImage, cv::Scalar(0.229f, 0.224f, 0.225f), gImage, 1, -1);
            // Allocate
            if (this->m_inputTensors[index].isCHW){
                std::vector<cv::cuda::GpuMat> chw;
                try {
                    for (unsigned j = 0; j < imgC; j++){
                        chw.emplace_back(cv::cuda::GpuMat(imgsz, CV_32FC1, buffer + (i*imgC+j)*imgW*imgH));
                    }
                    cv::cuda::split(gImage, chw);
                }
                catch (cv::Exception& e) {
    				std::cerr << "\033[1;31m[ERROR] [OpenCV] Exception caught: " << e.what() << "\033[0m\n";
    				return false;
				}
            }
            else {
                size_t bufferSize = utils::getShapeSize(this->m_inputTensors[index].shape);
				if (!utils::cudaCheck(cudaMemcpyAsync(buffer, gImage.ptr<float>(), bufferSize*sizeof(float), cudaMemcpyDeviceToDevice))){
                    return false;
                }
            }
        }
        else if (imgC == 1){
            // Normalize
            gImage.convertTo(gImage, CV_32FC1, 1.f/255.f);
            // Allocate
            if (!utils::cudaCheck(cudaMemcpyAsync(buffer, gImage.ptr<float>(), gImage.rows*gImage.step, cudaMemcpyDeviceToDevice))){
                return false;
            }
        }
        else {
            std::cerr << "\033[1;31m[ERROR] Undefined image channel! \033[0m\n";
			return false;
        }
    }
    return true;
}

bool TensorRT::allocateNonImage(void *pData, float* buffer, const unsigned index){
	if (index >= this->m_inputTensors.size()){
		std::cerr << "\033[1;31m[ERROR] inputIndex is greater than number of inputTensor's index!\033[0m\n";
		return false;
	}
	size_t bufferSize = utils::getShapeSize(this->m_inputTensors[index].shape);
	if (!utils::cudaCheck(cudaMemcpyAsync(buffer, pData, bufferSize * this->m_inputTensors[index].size, cudaMemcpyHostToDevice))){
        return false;
    }
	return true;
}

std::vector<float> TensorRT::postProcessing(float *buffer, const unsigned size, const unsigned index){
    if (index >= this->m_outputTensors.size()){
		throw std::invalid_argument("outputIndex is greater than number of outputTensor's index!\n");
	}
    // Create CPU buffer
    std::vector<float> result(size);

    // Transfer data from GPU to CPU
    if (!utils::cudaCheck(cudaMemcpyAsync(result.data(), buffer, result.size() * this->m_outputTensors[index].size, cudaMemcpyDeviceToHost))) {
		throw std::invalid_argument("Get data from device to host failure!");
	}
    return result;

}

std::vector<std::vector<cv::Mat>> TensorRT::batchSplit(std::vector<cv::Mat> images){
    unsigned i = 0;
    std::vector<std::vector<cv::Mat>> batchedImages;
    std::vector<cv::Mat> batch;
    while (i < images.size()){
        batch.emplace_back(images[i]);
        if (batch.size() == this->m_maxBatchSize || i == images.size()-1){
            batchedImages.emplace_back(batch);
            batch.clear();
        }
        if (batch.size() >= this->m_maxBatchSize || i >= images.size()){
            throw std::invalid_argument("Unidentify error: batchsize > maxBatchSize!");
        }
        i++;
    }
    return batchedImages;
}

std::vector<tensor::TensorValue> TensorRT::inference(std::vector<cv::Mat> images){
    std::vector<tensor::TensorValue> result;
    unsigned noInputs = this->m_inputTensors.size();
    if (noInputs != 1){
        throw std::invalid_argument("Your must add  allocateInput function with coresponding inputIndex for other inputs above / add data for Inference()'s arguments then remove this condition to continue!");
    }
    
    std::vector<std::vector<cv::Mat>> batchedImages = TensorRT::batchSplit(images);
    for (auto& batch : batchedImages){
        if (batch.size() < 1 || batch.size() > this->m_maxBatchSize){
            throw std::invalid_argument("Batchsize must be '> 1' and 'batchSize <= " + std::to_string(this->m_maxBatchSize) + "'");
        }
        if (!this->allocateImage(batch, (float*)this->m_buffers[0], 0)){
            throw std::invalid_argument("Could not allocate input images!");
        }
        this->m_context->enqueueV2(this->m_buffers.data(), 0, nullptr);

        for (unsigned i = 0; i < this->m_outputTensors.size(); i++) {
            std::string name = this->m_outputTensors[i].name;
            // Get shape size without batchsize 
            std::vector<unsigned> shape(this->m_outputTensors[i].shape.begin()+1, this->m_outputTensors[i].shape.end());
            size_t tensorSize = utils::getShapeSize(shape);
            std::vector<float> preds;
            try {
                preds = this->postProcessing((float *)this->m_buffers[i+noInputs], batch.size()*tensorSize, i);
            }
            catch (std::exception& err) {
                throw std::invalid_argument(err.what());
            }
            // Divide output by tensorSize 
            for (unsigned j = 0; j < batch.size(); j++){
                std::vector<float> pred(preds.begin() + j*tensorSize, preds.begin() + (j+1)*tensorSize);
                result.emplace_back(name, shape, pred);
                pred.clear();
            }
            preds.clear();
        }
    }
    batchedImages.clear();
	return result;
}
