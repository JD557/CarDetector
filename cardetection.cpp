#include <opencv2/core/core.hpp> 
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <vector>
#include <sstream>
#include <string>
#include <fstream>
#include "segmenter.hpp"
#include "ensemble.hpp"
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
		if (i<0.5*dataSet.size()) {trainSet.push_back(dataSet[i]);}
		else {testSet.push_back(dataSet[i]);}
	}

	SIFTBOWSegmenter siftbow;
	siftbow.train(trainSet);

	SURFBOWSegmenter surfbow;
	surfbow.train(trainSet);

	FASTBOWSegmenter fastbow;
	fastbow.train(trainSet);

	vector<Segmenter*> models;
	models.push_back(&siftbow);
	models.push_back(&surfbow);
	models.push_back(&fastbow);

	MajorityVoter voter(models);

	for (int i=0;i<testSet.size();++i) {
		stringstream id;
		id << "results/" << i;
		Mat original = testSet[i].image;
		imshow("Original", original);
		imwrite(id.str()+" Original.png",original);

		Mat maskSift = siftbow.apply(original);
		Mat extractedSift;
		original.copyTo(extractedSift,maskSift);
		imshow("Extracted - SIFT", extractedSift);
		imwrite(id.str()+" SIFT.png",extractedSift);

		Mat maskSurf = surfbow.apply(original);
		Mat extractedSurf;
		original.copyTo(extractedSurf,maskSurf);
		imshow("Extracted - SURF", extractedSurf);
		imwrite(id.str()+" SURF.png",extractedSurf);

		Mat maskFast = fastbow.apply(original);
		Mat extractedFast;
		original.copyTo(extractedFast,maskFast);
		imshow("Extracted - FAST", extractedFast);
		imwrite(id.str()+" FAST.png",extractedFast);

		Mat maskMaj = voter.apply(original);
		Mat extractedMaj;
		original.copyTo(extractedMaj,maskMaj);
		imshow("Extracted - Majority", extractedMaj);
		imwrite(id.str()+" Majority.png",extractedMaj);
		waitKey(1000);
	}

	waitKey(0);

	return 0;
}