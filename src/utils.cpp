#include "utils.h"

int utils::checkPathAttribute(const std::string path) {
    struct stat s;
    if( stat(path.c_str(), &s) == 0 ){
        if( s.st_mode & S_IFDIR )
        {
            return 2;
        }
        else if( s.st_mode & S_IFREG ){
            return 1;
        }
        else{
            std::cerr << "\033[1;31m[ERROR] '" << path << "' not readable!\n";
            return 0;
        }
    }
    else{
        return 0;
    }
}

bool utils::checkIsFileAndExist(const std::string filePath){
    if (!utils::checkPathAttribute(filePath)){
        std::cerr << "\033[1;31m[ERROR] '"<< filePath <<"' does not exist!\033[0m\n";
        return false;
    }
    else if (utils::checkPathAttribute(filePath) == 2){
        std::cerr << "\033[1;31m[ERROR] '"<< filePath <<"' is a folder !\033[0m\n";
        return false;
    }
    return true;
}


bool utils::checkIsFolderAndExist(const std::string folderPath){
    if (!utils::checkPathAttribute(folderPath)){
        std::cerr << "\033[1;31m[ERROR] '"<< folderPath <<"' does not exist!\033[0m\n";
        return false;
    }
    else if (utils::checkPathAttribute(folderPath) == 1){
        std::cerr << "\033[1;31m[ERROR] '"<< folderPath <<"' is a file !\033[0m\n";
        return false;
    }
    return true;
}

bool utils::copyBinaryFile(const std::string src, const std::string dst){
    if (utils::checkPathAttribute(src) != 1){
        std::cerr << "\033[1;31m[ERROR] '"<< src <<"' not exist!\033[0m\n";
        return false;
    }
    char buf[BUFSIZ];
	size_t size;
	FILE* source = fopen(src.c_str(), "rb");
	FILE* dest = fopen(dst.c_str(), "wb");
	while (size = fread(buf, 1, BUFSIZ, source)) {
		fwrite(buf, 1, size, dest);
	}
	fclose(source);
	fclose(dest);
    return true;
}

bool utils::cudaCheck(cudaError_t status){
    if (status != cudaSuccess){                                                   
        std::cerr << "\033[1;31m[ERROR] [CUDA Failure] " << cudaGetErrorString(status) << "\033[0m\n";                        
        return false;                                                    
    }
    return true; 
}

unsigned utils::getShapeSize(const std::vector<unsigned> shape){
    unsigned size = 1;
    for (unsigned i = 0; i < shape.size(); i++){
        size *= shape[i];
    }
    return size;
}

bool utils::setCudaNum(const unsigned gpuNum){
    int deviceCount = 0;
    if (!utils::cudaCheck(cudaGetDeviceCount(&deviceCount))) {
        return false;
    }

    else {
        std::cout << "[INFO] Device Count: " << deviceCount << std::endl;
    }
    if (gpuNum >= deviceCount){
        std::cerr << "\033[1;31m[ERROR] Gpunum must smaller than '" << deviceCount <<"'. Got '" << gpuNum << "' !\033[0m\n";
        return false;
    }
    if (!utils::cudaCheck(cudaSetDevice(gpuNum))){
        return false;
    }

	size_t totalDevMem, freeDevMem;
	if (!cudaCheck(cudaMemGetInfo(&freeDevMem, &totalDevMem))){
        return false;
    }
    std::cout << "[INFO] Switched to GPU:" << gpuNum << " success! Free memory: " << freeDevMem/1048576 <<"MB. \n";
    return true;

}