#ifndef _UTILS_HPP_
#define _UTILS_HPP_

#include <iostream>
#include <dirent.h>
using namespace std;

static bool checkFileIfExist(string filePath){
	ifstream f(filePath, ios::binary);
	if (!f.good()){
		f.close();
		return false;
	}
	f.close();
	return true;
}

static inline int readFilesInDir(const char *p_dir_name, std::vector<std::string> &file_names) {
    DIR *p_dir = opendir(p_dir_name);
    if (p_dir == nullptr) {
        return -1;
    }

    struct dirent* p_file = nullptr;
    while ((p_file = readdir(p_dir)) != nullptr) {
        if (strcmp(p_file->d_name, ".") != 0 &&
            strcmp(p_file->d_name, "..") != 0) {
            std::string cur_file_name(p_file->d_name);
            file_names.push_back(cur_file_name);
        }
    }
    closedir(p_dir);
    return 0;
}

#endif