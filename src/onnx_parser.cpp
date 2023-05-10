#include "onnx_parser.h"

static auto StreamDeleter = [](cudaStream_t* pStream){
    if (pStream){
        cudaStreamDestroy(*pStream);
        delete pStream;
    }
};

inline std::unique_ptr<cudaStream_t, decltype(StreamDeleter)> makeCudaStream(){
    std::unique_ptr<cudaStream_t, decltype(StreamDeleter)> pStream(new cudaStream_t, StreamDeleter);
    if (cudaStreamCreateWithFlags(pStream.get(), cudaStreamNonBlocking) != cudaSuccess)
    {
        pStream.reset(nullptr);
    }

    return pStream;
}

nvinfer1::IHostMemory* f_parseOnnxEngine(YAML::Node& config, std::string configPath){
    // Load config 
    std::string onnxPath="";
    bool isDynamicShape=false, useFP16=false;
    int batchSize=0, workspaceSize=0;
    try{
        config = YAML::LoadFile(configPath);
        onnxPath        = config["onnx_path"].as<std::string>();
        isDynamicShape  = config["use_dynamic"].as<bool>();
        useFP16         = config["use_fp16"].as<bool>();
        batchSize       = config["batch_size"].as<unsigned>();
        workspaceSize   = config["workspace_size"].as<unsigned>();
    }
    catch (std::exception& e){
        std::cerr << "\033[1;31m[ERROR] Could not parse '"<< configPath << "' as yaml file!\033[0m\n";
        std::cerr << "\033[1;31m[ERROR] Please check '"<< configPath << "' is a correct yaml file or not!\033[0m\n";
        return nullptr;
    }
    // Print param read from yaml file:
    std::cout << "Onnx Path: '" << onnxPath << "'\n";
    std::cout << "\t- Max batch size: '" << batchSize << "'\n";
    std::cout << "\t- Max workspace size: '" << workspaceSize << "'\n";
    std::cout << "\t- useFP16: '" << useFP16 << "'\n";
    std::cout << "\t- isDynamicShape: '" << isDynamicShape << "'\n";
    if (!utils::checkIsFileAndExist(onnxPath)){
        std::cerr << "\033[1;31m[ERROR] 'onnx_path: "<< onnxPath <<"' in '"<< configPath << "' is an unavailable path!\033[0m\n";
        return nullptr;
    }
    // Create Builder, Network, Config
    auto builder = TRTUniquePtr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(gLogger));
    builder->setMaxBatchSize(batchSize);
    const auto flag = 1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    auto network = TRTUniquePtr<nvinfer1::INetworkDefinition>(builder->createNetworkV2(flag));
    auto engineConfig = TRTUniquePtr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
    if (!network || !engineConfig){
        std::cerr << "\033[1;31m[ERROR] Could not create network or config from builder'! \033[0m\n";
        return nullptr;
    }
    // Configure onnx engine
    engineConfig->setFlag(nvinfer1::BuilderFlag::kGPU_FALLBACK);
    auto profileStream = makeCudaStream();
    engineConfig->setProfileStream(*profileStream);
    engineConfig->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, workspaceSize);

    if (isDynamicShape){
        auto profile = builder->createOptimizationProfile();
        int id = 0, minBatch = 1, halfBatch = std::max(int(batchSize/2),1);
        std::cout << "[INFO]  Dynamic tensor input: \n";  
        while (1) {
            id++;
            std::string name = "";
            std::vector<int> dims;
            try {
                std::string tensorID = "tensor"+ std::to_string(id);
                name = config[tensorID]["name"].as<std::string>();
                dims = config[tensorID]["dims"].as<std::vector<int>>();
            }
            catch (std::exception& e){
                break;
            }
            std::cout << "\t+ Input: '" << name << "': batch_size x ";
            for (int i = 0; i < dims.size()-1; i ++){
                std::cout <<  dims[i] << " x ";
            }
            std::cout <<  dims[dims.size()-1] << "\n";
            if (dims.size() == 1){
                profile->setDimensions(name.c_str(), nvinfer1::OptProfileSelector::kMIN, nvinfer1::Dims2{minBatch, dims[0]});
                profile->setDimensions(name.c_str(), nvinfer1::OptProfileSelector::kOPT, nvinfer1::Dims2{halfBatch, dims[0]});
                profile->setDimensions(name.c_str(), nvinfer1::OptProfileSelector::kMAX, nvinfer1::Dims2{batchSize, dims[0]});
            }
            else if (dims.size() == 2){
                profile->setDimensions(name.c_str(), nvinfer1::OptProfileSelector::kMIN, nvinfer1::Dims3{minBatch, dims[0], dims[1]});
                profile->setDimensions(name.c_str(), nvinfer1::OptProfileSelector::kOPT, nvinfer1::Dims3{halfBatch, dims[0], dims[1]});
                profile->setDimensions(name.c_str(), nvinfer1::OptProfileSelector::kMAX, nvinfer1::Dims3{batchSize, dims[0], dims[1]});
            }
            else if (dims.size() == 3){
                profile->setDimensions(name.c_str(), nvinfer1::OptProfileSelector::kMIN, nvinfer1::Dims4{minBatch, dims[0], dims[1], dims[2]});
                profile->setDimensions(name.c_str(), nvinfer1::OptProfileSelector::kOPT, nvinfer1::Dims4{halfBatch, dims[0], dims[1], dims[2]});
                profile->setDimensions(name.c_str(), nvinfer1::OptProfileSelector::kMAX, nvinfer1::Dims4{batchSize, dims[0], dims[1], dims[2]});
            }
        }
        engineConfig->addOptimizationProfile(profile);
        
    }
    if (useFP16){
        if (builder->platformHasFastFp16()){
            std::cout << "[INFO] Exporting model as FP16 ....\n";
            engineConfig->setFlag(nvinfer1::BuilderFlag::kFP16);
        }
        else {
            std::cout << "\033[1;33m[WARNING] This system does not support FP16. Export model in FP32.\033[0m\n";
        }
    }
    else {
        std::cout << "[INFO] Exporting model as FP32 ....\n";
    }
    // Load network and weight from onnxPath
    auto parser = TRTUniquePtr<nvonnxparser::IParser>(nvonnxparser::createParser(*network, gLogger));
    if (!parser->parseFromFile(onnxPath.c_str(), static_cast<int>(nvinfer1::ILogger::Severity::kWARNING))){
        std::cerr << "\033[1;31m[ERROR] Could not parse '"<< onnxPath <<"'! \033[0m\n";
        return nullptr;
    }
    return builder->buildSerializedNetwork(*network, *engineConfig);
}

bool onnx::convertEngine(const std::string configPath){
    // Check config path exist
    if (!utils::checkIsFileAndExist(configPath)){
        std::cerr << "\033[1;31m[ERROR] '"<< configPath <<"' not exist!\033[0m\n";
        return false;
    }
    // Load engine    
    YAML::Node config;
    //Config will be loaded in this f_parseOnnxEngine
    nvinfer1::IHostMemory* serializedEngine = f_parseOnnxEngine(config, configPath);
    std::string onnxPath = config["onnx_path"].as<std::string>(); 
    std::string trtPath = onnxPath.substr(0, onnxPath.find_last_of(".")) + ".trt";
    if (serializedEngine == nullptr) {
        std::cerr << "\033[1;31m[ERROR] Could not serialize engine '"<< onnxPath << "'!\033[0m\n";
		return false;
	}
    // Duplicate onnx engine in order to use 1 of them to convert and. 
    if (!utils::copyBinaryFile(onnxPath, trtPath)){
        std::cerr << "\033[1;31m[ERROR] Could not duplicate '"<< onnxPath <<"'!\033[0m\n";
        return false;
    }
    // Check engine can open or not
    std::ofstream engineFile(trtPath, std::ios::binary);
    if (!engineFile){
        std::cerr << "\033[1;31m[ERROR] Could not open engine file: '" << trtPath << "'!\033[0m\n";
        engineFile.close();
        remove(trtPath.c_str());
        return false;
    }

    // Write serialized TensorRT engine into file
    engineFile.write(static_cast<char*>(serializedEngine->data()), serializedEngine->size());
    if (engineFile.fail()){
        std::cerr << "\033[1;31m[ERROR] Could not write serialized engine to '" << trtPath << "'!\033[0m\n";
        engineFile.close();
        remove(trtPath.c_str());
        return false;
    }
    std::cout << "\033[1;32m[INFO] '"<< trtPath <<"' created! \033[0m\n";
    
    // nvinfer1::IRuntime* runtime = nvinfer1::createInferRuntime(gLogger);
    // nvinfer1::ICudaEngine* engine = runtime->deserializeCudaEngine(serializedEngine->data(), serializedEngine->size());
    // try {
    //     std::cout << engine;
    // }
    // catch (std::exception& e){
    //     std::cerr << "\033[1;31m[ERROR] " << e.what() << "\033[0m\n";
    //     return false;
    // }
    // delete runtime;
    // delete engine;
    
    delete serializedEngine;
    engineFile.close();
    return true;
}