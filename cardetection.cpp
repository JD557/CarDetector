#include <opencv2/core/core.hpp> 
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/contrib/contrib.hpp>
#include <opencv2/nonfree/nonfree.hpp>
#include <opencv2/nonfree/features2d.hpp>
#include <iostream>
#include <vector>
#include <sstream>
#include <string>
#include <fstream>
using namespace std;
using namespace cv;

struct CarImage {
	Mat image;
	vector<Mat> masks;
	vector<KeyPoint> keypoints;
	Mat keypointDesc;
	vector<bool> isKeypointCar;
};


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

void findKeyPoints(std::vector<CarImage> &dataSet, Ptr<FeatureDetector> detector, Ptr<DescriptorExtractor> extractor) {
	cout << "Finding KeyPoints..." << endl;
	for (size_t i=0;i<dataSet.size();++i) {
		detector->detect(dataSet[i].image,dataSet[i].keypoints);
		extractor->compute(dataSet[i].image,dataSet[i].keypoints,dataSet[i].keypointDesc);
	}
	cout << "DONE" << endl;
}

void setKeyPointLabels(std::vector<CarImage> &trainSet) {
	cout << "Labeling KeyPoints..." << endl;
	for (size_t i=0;i<trainSet.size();++i) {
		for (size_t j=0;j<trainSet[i].keypoints.size();++j) {
			Point point = trainSet[i].keypoints[j].pt;
			bool isCar = false;
			for (size_t k=0;k<trainSet[i].masks.size();++k) {
				if (trainSet[i].masks[k].empty()) {cout << "VAZIO" << endl;}
				if (trainSet[i].masks[k].at<Vec3b>(point.y,point.x)[2]==255) { // Red in BGR
					isCar = true;
					break;
				}
			}
			trainSet[i].isKeypointCar.push_back(isCar);
		}
	}
	cout << "DONE" << endl;
}

Mat trainVocabulary(std::vector<CarImage> &trainSet, BOWKMeansTrainer &bowTrainer) {
	cout << "Training Vocabulary..." << endl;
	for (size_t i=0;i<trainSet.size();++i) {
		bowTrainer.add(trainSet[i].keypointDesc);
	}
	Mat vocabulary = bowTrainer.cluster();
	cout << "DONE" << endl;
	return vocabulary;
}


int main (){
	string detectorStr = "SIFT";
	string extractorStr = "SIFT";
	string matcherStr = "FlannBased";
	int clusterCount = 100;

	initModule_nonfree();
	
	Ptr<FeatureDetector> detector = FeatureDetector::create(detectorStr);
	Ptr<DescriptorExtractor> extractor = DescriptorExtractor::create(extractorStr);
	Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create(matcherStr);
	BOWKMeansTrainer bowTrainer(clusterCount, TermCriteria(), 3, KMEANS_PP_CENTERS);
	BOWImgDescriptorExtractor bowExtractor(detector, matcher);

	vector<CarImage> dataSet=loadDataset();
	vector<CarImage> trainSet;
	vector<CarImage> testSet;
	for (size_t i=0;i<dataSet.size();++i) {
		if (i<10) {trainSet.push_back(dataSet[i]);}
		else {testSet.push_back(dataSet[i]);}
	}
	findKeyPoints(trainSet,detector,extractor);
	setKeyPointLabels(trainSet);
	Mat vocabulary = trainVocabulary(trainSet,bowTrainer);
	bowExtractor.setVocabulary(vocabulary);

	Mat teste = trainSet[0].image;
	for (size_t i=0;i<trainSet[0].keypoints.size();++i) {
		bool isCar = trainSet[0].isKeypointCar[i];
		circle(teste,trainSet[0].keypoints[i].pt,5,isCar?Scalar(0,255,0):Scalar(0,0,255),2);
	}
	namedWindow("Car", CV_WINDOW_KEEPRATIO);
	imshow("Car", teste);
	waitKey(0);

	return 0;
}