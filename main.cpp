#include <iostream>
#include <onnxparser.h>

int main(int argc,char** argv){
	string model_path = "../20210108_0049_0925.onnx";
	string image_path = "../0.bmp";
	cv::Mat image = cv::imread(image_path);
	OnnxParser model(model_path, 1);
	model.inference(image);
	return 0;
}