#include "tensor.h"

tensor::TensorAttribute::TensorAttribute(const nvinfer1::ICudaEngine* engine, const unsigned index){
    if (index >= engine->getNbBindings()){
        throw std::invalid_argument("Index is more than engine's binding index");
        abort();
    }
    this->name = engine->getBindingName(index);
    this->isInput = engine->bindingIsInput(index);
    // Get tensor size & type
    nvinfer1::DataType type = engine->getBindingDataType(index);
    if (type == nvinfer1::DataType::kFLOAT) {
        this->size = sizeof(float);
        this->type = "kFLOAT";
    }
    else if (type == nvinfer1::DataType::kHALF) {
        this->size = sizeof(float)/2;
        this->type = "kHALF";
    }
    else if (type == nvinfer1::DataType::kINT8) {
        this->size = sizeof(int8_t);
        this->type = "kINT8";
    }
    else if (type == nvinfer1::DataType::kINT32) {
        this->size = sizeof(int32_t);
        this->type = "kINT32";
    }
    else if (type == nvinfer1::DataType::kBOOL) {
        this->size = sizeof(bool);
        this->type = "kBOOL";
    }
    else{
        throw std::invalid_argument("Unsupported DataType! Please check 'https://docs.nvidia.com/deeplearning/tensorrt/api/c_api/namespacenvinfer1.html' to add new type!\n");
        abort();
    }
    // Get tensor shape
    nvinfer1::Dims32 dims = engine->getBindingDimensions(index);
    for (unsigned i = 0; i < dims.nbDims; i++){
        shape.emplace_back(dims.d[i]);
    }
    this->isImage = false;
    if (shape.size() == 4){
        this->isImage = true;
    }
    isCHW = false;
    if (isImage && (shape[1] == 1 || shape[1] == 3)){
        isCHW = true;
    }
}

tensor::TensorValue::TensorValue(const std::string t_name, const std::vector<unsigned> t_shape, const std::vector<float> t_value){
    this->name = t_name;
    this->shape = t_shape;
    this->value = t_value;
}

std::ostream& tensor::operator << (std::ostream& os, const TensorAttribute& x) {
    os << "'" << x.name << "': ";
    for (unsigned i = 0; i < x.shape.size()-1; i++) {
        os << x.shape[i] << " x ";
    }
    os << x.shape[x.shape.size()-1];
    os << " (" << x.type << ") ";
    if (x.isInput && x.isImage){
        os << "(Image - channel " << (x.isCHW ? "first":"last") << ")";
    }
    return os;
}

std::ostream& tensor::operator << (std::ostream& os, const TensorValue& x) {
    os << "'" << x.name << "'(1x";
    unsigned noDims = x.shape.size();
    unsigned shapeSize = utils::getShapeSize(x.shape);
    for (unsigned i = 0; i < noDims-1; i++){
        os << x.shape[i] << "x";
    }
    os << x.shape[noDims-1] << "): ";
    os << "\n[";
    if (noDims == 1){
        //1D: i
        unsigned I = x.shape[0];
        if (shapeSize < MAXSIZE){
            for (unsigned i = 0; i < I; i++){
                os << x.value[i] << " ";
            }
        }
        else{
            os << x.value[0] << " ... " << x.value[I-1];
        }
    }
    else if (noDims == 2){
        //2D: r x c
        unsigned R = x.shape[0];
        unsigned C = x.shape[1];
        if (shapeSize < MAXSIZE){
            for (unsigned r = 0; r < R; r++){
                os << "\n [";
                for (unsigned c = 0; c < C; c++){
                    os << x.value[r*C + c] << " ";
                }
                os << " ]\n";
            }
        }
        else{
            os << "\n [";
            for (unsigned c = 0; c < C; c++){
                os << x.value[c] << " ";
            }
            os << " ]\n";
            os << " ...";
            os << "\n [";
            for (unsigned c = 0; c < C; c++){
                os << x.value[(R-1)*C + c] << " ";
            }
            os << " ]\n";
        }
    }
    else if (noDims == 3){
        // 3D: z x r x c
        unsigned Z = x.shape[0];
        unsigned R = x.shape[1];
        unsigned C = x.shape[2];
        if (shapeSize < MAXSIZE){
            for (unsigned z = 0; z < Z; z++){
                os << "\n [";
                for (unsigned r = 0; r < R; r++){
                    os << "\n  [";
                    for (unsigned c = 0; c < C; c++){
                        os << x.value[z*R*C + r*C + c] << " ";
                    }
                    os << "  ]\n";
                }
                os << " ]\n";
            }
        }
        else{
            os << "\n [";
            os << "\n  [";
            for (unsigned c = 0; c < C; c++){
                os << x.value[c] << " ";
            }
            os << "  ]\n";
            os << "  ...";
            os << "\n  [";
            for (unsigned c = 0; c < C; c++){
                os << x.value[(Z-1)*R*C + (R-1)*C + c] << " ";
            }
            os << "  ]\n";
            os << " ]\n";
            os << " ...\n";

        }
    }
    else if (noDims == 4){
        // 4D: y x z x r x c
        unsigned Y = x.shape[0];
        unsigned Z = x.shape[1];
        unsigned R = x.shape[2];
        unsigned C = x.shape[3];
        if (shapeSize < MAXSIZE){
            for (unsigned y = 0; y < Y; y++){
                os << "\n [";
                for (unsigned z = 0; z < Z; z++){
                    os << "\n  [";
                    for (unsigned r = 0; r < R; r++){
                        os << "\n   [";
                        for (unsigned c = 0; c < C; c++){
                            os << x.value[y*Z*R*C + z*R*C + r*C + c] << " ";
                        }
                        os << "   ]\n";
                    }
                    os << "  ]\n";
                }
                os << " ]\n";
            }
        }
        else{
            os << "\n [";
            os << "\n  [";
            os << "\n   ["; 
            for (unsigned c = 0; c < C; c++){
                os << x.value[c] << " ";
            }
            os << "   ]\n"; 
            os << "   ...";
            os << "\n   ["; 
            for (unsigned c = 0; c < C; c++){
                os << x.value[(Y-1)*Z*R*C + (Z-1)*R*C + (R-1)*C + c] << " ";
            }
            os << "   ]\n";
            os << "  ]\n";
            os << "  ...\n";
            os << " ]\n";
            os << " ...\n";
        }
    }
    else {
        throw std::invalid_argument("Unidentify dimensions of tensor");
    }
    os << "]\n";
    return os;
}

void tensor::softmax(TensorValue& x){
    std::transform(x.value.begin(), x.value.end(), x.value.begin(), [](float val) {return exp(val);});
    unsigned noDims = x.shape.size();
    if (noDims == 1){
        //1D: i
        unsigned I = x.shape[0];
        float sum = 0;
        for (unsigned i = 0; i < I; i++){
            sum += x.value[i];
        }
        for (unsigned i = 0; i < I; i++){
            x.value[i] /= sum;
        }
    }
    else if (noDims == 2){
        //2D: r x c
        unsigned R = x.shape[0];
        unsigned C = x.shape[1];
        for (unsigned r = 0; r < R; r++){
            float sum = 0;
            for (unsigned c = 0; c < C; c++){
                sum += x.value[r*C + c];
            }
            for (unsigned c = 0; c < C; c++){
                x.value[r*C + c] /= sum;
            } 
        }

    }
    else if (noDims == 3){
        // 3D: z x r x c
        unsigned Z = x.shape[0];
        unsigned R = x.shape[1];
        unsigned C = x.shape[2];
        for (unsigned z = 0; z < Z; z++){
            for (unsigned r = 0; r < R; r++){
                float sum = 0;
                for (unsigned c = 0; c < C; c++){
                    sum += x.value[z*R*C + r*C + c];
                }
                for (unsigned c = 0; c < C; c++){
                    x.value[z*R*C + r*C + c] /= sum;
                }
            }
        }
    }
    else if (noDims == 4){
        // 4D: y x z x r x c
        unsigned Y = x.shape[0];
        unsigned Z = x.shape[1];
        unsigned R = x.shape[2];
        unsigned C = x.shape[3];
        for (unsigned y = 0; y < Y; y++){
            for (unsigned z = 0; z < Z; z++){
                for (unsigned r = 0; r < R; r++){
                    float sum = 0;
                    for (unsigned c = 0; c < C; c++){
                        sum += x.value[y*Z*R*C + z*R*C + r*C + c];
                    }
                    for (unsigned c = 0; c < C; c++){
                        x.value[y*Z*R*C + z*R*C + r*C + c] /= sum;
                    }
                }
            }
        }
    }
    else {
        throw std::invalid_argument("Unidentify dimensions of tensor");
    }
}

std::ostream& operator << (std::ostream& os, const nvinfer1::ICudaEngine* x){
    if (x == nullptr){
        throw std::invalid_argument("ShowEngineinfo: ICudaEngine is null! \n");
        abort();
    }
    std::cout << "[INFO] TensorRT engine info: \n";
    std::cout << "\t - Max batchSize: " << x->getMaxBatchSize() << std::endl;
	std::cout << "\t - Engine size: " << x->getDeviceMemorySize()/(1048576) << " MB (GPU Mem)" << std::endl; 
	std::cout << "\t - Tensors: \n";
    for (unsigned i = 0; i < x->getNbBindings(); i++) {
		tensor::TensorAttribute tensor(x, i);
		if (x->bindingIsInput(i)) {
			std::cout << "\t\t + (Input) " << tensor << std::endl;
		}
		else{
			std::cout << "\t\t + (Output) " << tensor << std::endl;
		}
	}
    return os;
}


