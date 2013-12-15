#include <opencv2/core/core.hpp> 
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <vector>
#include <sstream>
#include <string>
#include <fstream>
#include <algorithm>
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
				tempImage.imageName = line;
			}
		}
	}
	cout << "DONE" << endl;
	return dataSet;
}

double getModelPrecision(CarImage img,Mat result) { // TP/(TP+FP)
	double truePositive;
	double positive;
	for (size_t y=0;y<result.rows;++y) {
		for (size_t x=0;x<result.cols;++x) {
			if (result.at<unsigned char>(y,x)==255) {
				positive++;
				for (size_t i=0;i<img.masks.size();++i) {
					Vec3b pixel=img.masks[i].at<Vec3b>(y,x);
					if (pixel[0]==255 || pixel[1]==255 || pixel[2]==255) {
						truePositive++;
						break;
					}
				}
			}
		}
	}
	if (positive==0) {return INFINITY;}
	return truePositive/positive;
}

double getModelRecall(CarImage img,Mat result) { // TP/(TP+FN)
	double truePositive;
	double falseNegative;
	for (size_t y=0;y<result.rows;++y) {
		for (size_t x=0;x<result.cols;++x) {
			for (size_t i=0;i<img.masks.size();++i) {
				Vec3b pixel=img.masks[i].at<Vec3b>(y,x);
				if (pixel[0]==255 || pixel[1]==255 || pixel[2]==255) {
					if (result.at<unsigned char>(y,x)==255) {truePositive++;}
					else {falseNegative++;}
					break;
				}
			}
		}
	}
	if (truePositive+falseNegative==0) {return INFINITY;}
	return truePositive/(truePositive+falseNegative);
}

string getModelStats(CarImage img,Mat result) {
	stringstream output;
	double precision = getModelPrecision(img,result);
	double recall = getModelRecall(img,result);
	output << "\n\tPrecision: ";
	if (precision>1) {output << "INF";}
	else {output << precision;}
	output << "\n\tRecall:     ";
	if (recall>1) {output << "INF";}
	else {output << recall;}
	return output.str();
}


int main() {

	ofstream resultsFile("results/results.txt");

	initModule_nonfree();

	vector<CarImage> dataSet=loadDataset();
	vector<CarImage> trainSet;
	vector<CarImage> testSet;

	string temp;

	vector<string> trainFiles;
	ifstream carsTrain("cars_train.txt");
	while (getline(carsTrain,temp)) {
		trainFiles.push_back(temp+".image.png");;
	}
	sort(trainFiles.begin(),trainFiles.end());


	vector<string> testFiles;
	ifstream carsTest("cars_test.txt");
	while (getline(carsTest,temp)) {
		testFiles.push_back(temp+".image.png");
	}
	sort(testFiles.begin(),testFiles.end());


	for (size_t i=0;i<dataSet.size();++i) {
		if (binary_search(trainFiles.begin(),trainFiles.end(),dataSet[i].imageName)) {
			trainSet.push_back(dataSet[i]);
		}
		if (binary_search(testFiles.begin(),testFiles.end(),dataSet[i].imageName)) {
			testSet.push_back(dataSet[i]);
		}
	}

	cout << "DataSet: " << dataSet.size() << " " << trainSet.size() << " " << testSet.size() << endl;

	SIFTBOWSegmenter siftbow;
	siftbow.train(trainSet);

	SURFBOWSegmenter surfbow;
	surfbow.train(trainSet);

	FASTBOWSegmenter fastbow;
	fastbow.train(trainSet);

	HarrisBOWSegmenter harrisbow;
	harrisbow.train(trainSet);

	STARBOWSegmenter starbow;
	starbow.train(trainSet);

	vector<Segmenter*> models;
	models.push_back(&siftbow);
	models.push_back(&surfbow);
	models.push_back(&fastbow);
	models.push_back(&harrisbow);
	models.push_back(&starbow);

	MajorityVoter voter(models);

	for (int i=0;i<testSet.size();++i) {
		resultsFile << i << "(" << testSet[i].imageName << ")" << endl;
		stringstream id;
		id << "results/" << i;
		Mat original = testSet[i].image;
		//imshow("Original", original);
		imwrite(id.str()+" Original.png",original);

		Mat maskSift = siftbow.apply(original);
		Mat extractedSift;
		original.copyTo(extractedSift,maskSift);
		//mshow("Extracted - SIFT", extractedSift);
		imwrite(id.str()+" SIFT.png",extractedSift);
		resultsFile << "SIFT" << getModelStats(testSet[i],maskSift) << endl;

		Mat maskSurf = surfbow.apply(original);
		Mat extractedSurf;
		original.copyTo(extractedSurf,maskSurf);
		//imshow("Extracted - SURF", extractedSurf);
		imwrite(id.str()+" SURF.png",extractedSurf);
		resultsFile << "SURF" << getModelStats(testSet[i],maskSurf) << endl;

		Mat maskFast = fastbow.apply(original);
		Mat extractedFast;
		original.copyTo(extractedFast,maskFast);
		//imshow("Extracted - FAST", extractedFast);
		imwrite(id.str()+" FAST.png",extractedFast);
		resultsFile << "FAST" << getModelStats(testSet[i],maskFast) << endl;

		Mat maskHarris = harrisbow.apply(original);
		Mat extractedHarris;
		original.copyTo(extractedHarris,maskHarris);
		//imshow("Extracted - Harris", extractedHarris);
		imwrite(id.str()+" Harris.png",extractedHarris);
		resultsFile << "Harris" << getModelStats(testSet[i],maskHarris) << endl;

		Mat maskStar = starbow.apply(original);
		Mat extractedStar;
		original.copyTo(extractedStar,maskStar);
		//imshow("Extracted - STAR", extractedStar);
		imwrite(id.str()+" STAR.png",extractedStar);
		resultsFile << "STAR" << getModelStats(testSet[i],maskStar) << endl;

		Mat maskMaj = voter.apply(original);
		Mat extractedMaj;
		original.copyTo(extractedMaj,maskMaj);
		//imshow("Extracted - Majority", extractedMaj);
		imwrite(id.str()+" Majority.png",extractedMaj);
		resultsFile << "Majority" << getModelStats(testSet[i],maskMaj) << endl;
		waitKey(1000);
		resultsFile << endl;
	}

	waitKey(0);

	return 0;
}