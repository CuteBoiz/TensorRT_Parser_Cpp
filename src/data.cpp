#include "data.h"

bool data::readFilesInDir(const char *dataPath, std::vector<std::string> &filesName) {
    DIR *pDir = opendir(dataPath);
    if (pDir == nullptr) {
        return false;
    }
    struct dirent* pFile = nullptr;
    while ((pFile = readdir(pDir)) != nullptr) {
        if (strcmp(pFile->d_name, ".") != 0 && strcmp(pFile->d_name, "..") != 0) {
            std::string fileName(pFile->d_name);
            filesName.emplace_back(fileName);
        }
    }
    closedir(pDir);
    std::sort(filesName.begin(), filesName.end());
    return true;
}


data::InputData data::prepareData(std::string dataPath){
    InputData data;
    int status = utils::checkPathAttribute(dataPath);
    if (status == 0){
        throw std::invalid_argument("'"+ dataPath + "'is not exist!");
    }
    else if (status == 1){
        std::string exts = dataPath.substr(dataPath.find_last_of(".") + 1);
        std::transform(exts.begin(), exts.end(), exts.begin(), ::tolower);
        bool isImage = std::find(std::begin(data::imageExts), std::end(data::imageExts), exts) != std::end(data::imageExts);
        bool isVideo = std::find(std::begin(data::videoExts), std::end(data::videoExts), exts) != std::end(data::videoExts);
        if ((!isImage && !isVideo) || (isImage && isVideo)){
            throw std::invalid_argument("Unsupported '" + exts + "' extenstion!");
        }
        if (isImage){
            cv::Mat image = cv::imread(dataPath);
            data.imagePaths.emplace_back(dataPath);
            data.images.emplace_back(image);
        }
        if (isVideo){
            cv::VideoCapture cap(dataPath);
            if (!cap.isOpened()){
                throw std::invalid_argument("Could not open video '" + dataPath +"' !");
            }
            cv::Mat frame;
            unsigned id = 0;
            while (1){
                cap >> frame;
                if (frame.empty()){
                    break;
                }
                std::string name = dataPath.insert(dataPath.find_last_of("."), "_frame_" + std::to_string(id));
                data.imagePaths.emplace_back(name);
                data.images.emplace_back(frame);
                id++;
            }
        }
    }
    else if (status == 2){
        std::vector<std::string> filesName;
        if (dataPath[dataPath.length() - 1] != '/' && dataPath[dataPath.length() -1] != '\\') {
            dataPath = dataPath + '/';
        }
        if (!data::readFilesInDir(dataPath.c_str(), filesName)) {
            throw std::invalid_argument("Could not read files from '" + dataPath + "'!\n");
        }
        for (auto & fileName : filesName){
            std::string exts = fileName.substr(fileName.find_last_of(".") + 1);
            bool isImage = std::find(std::begin(data::imageExts), std::end(data::imageExts), exts) != std::end(data::imageExts);
            if (isImage){
                cv::Mat image = cv::imread(dataPath + fileName);
                data.imagePaths.emplace_back(dataPath + fileName);
                data.images.emplace_back(image); 
            }
        }
    }
    else {
        throw std::invalid_argument("DataPath attribute: Unidentify error!");
    }
    return data;    
}
