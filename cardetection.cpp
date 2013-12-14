#include <opencv2/core/core.hpp> 
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <vector>
#include <sstream>
#include <string>
#include <fstream>
#include "segmenter.hpp"
using namespace std;
using namespace cv;


vector<CarImage> loadDataset() {
	cout << "Loading Images..." << endl;
	std::vector<CarImage> dataSet;
	ifstream info("cars_info.txt");
	string line;
	CarImage tempImage;
	bool firstImage=true;
	while (getline(info,line)) {
		size_t offset = line.find_first_of(':');
		if (offset != string::npos) {
			line = line.substr(0,offset);
			if (line[0]==' ') { // MASK
				line = line.substr(line.find_last_of(' ')+1);
				tempImage.masks.push_back(imread("cars/"+line));
			} else {
				if (!firstImage) {
					dataSet.push_back(tempImage);
				}
				else {firstImage=false;}
				tempImage.masks.clear();
				tempImage.image = imread("cars/"+line);
			}
		}
	}
	cout << "DONE" << endl;
	return dataSet;
}

int main() {

	initModule_nonfree();

	vector<CarImage> dataSet=loadDataset();
	vector<CarImage> trainSet;
	vector<CarImage> testSet;
	for (size_t i=0;i<dataSet.size();++i) {
		if (i<dataSet.size()*0.1) {trainSet.push_back(dataSet[i]);}
		else {testSet.push_back(dataSet[i]);}
	}

	//SIFTKNNSegmenter siftknn;
	//siftknn.train(trainSet);

	//SURFKNNSegmenter surfknn;
	//surfknn.train(trainSet);

	SIFTBOWSegmenter siftbow;
	siftbow.train(trainSet);

	for (int i=0;i<testSet.size();++i) {
		Mat original = testSet[i].image;
		Mat mask = siftbow.apply(original);
		Mat extracted;
		original.copyTo(extracted,mask);
		imshow("Original", original);
		imshow("Mask", mask);
		imshow("extracted", extracted);
		waitKey(0);
	}

	waitKey(0);

	return 0;
}