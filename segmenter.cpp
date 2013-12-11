#include "segmenter.hpp"
#include <iostream>
#include <cmath>
#include <queue>
using namespace std;
using namespace cv;

#define KNN_K 3
bool operator<(const KeypointKNN a,const KeypointKNN b) {
	return a.dist>b.dist;
}

void Segmenter::findKeyPoints() {
	cout << "Finding KeyPoints..." << endl;
	for (size_t i=0;i<trainSet.size();++i) {
		detector->detect(trainSet[i].image,trainSet[i].keypoints);
		extractor->compute(trainSet[i].image,trainSet[i].keypoints,trainSet[i].keypointDesc);
	}
	cout << "DONE" << endl;
}

void Segmenter::setKeyPointLabels() {
	cout << "Labeling KeyPoints..." << endl;
	for (size_t i=0;i<trainSet.size();++i) {
		for (size_t j=0;j<trainSet[i].keypoints.size();++j) {
			Point point = trainSet[i].keypoints[j].pt;
			bool isCar = false;
			for (size_t k=0;k<trainSet[i].masks.size();++k) {
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

SIFTKNNSegmenter::SIFTKNNSegmenter() {
	detector = FeatureDetector::create("SIFT");
	extractor = DescriptorExtractor::create("SIFT");
}

void SIFTKNNSegmenter::train(vector<CarImage> trainSet) {
	this->trainSet=trainSet;
	findKeyPoints();
	setKeyPointLabels();

	Mat trainFeatures(0,128,CV_32F);
	Mat trainLabels(0,1,CV_16U);
	for (size_t i=0;i<this->trainSet.size();++i) {
		CarImage img = this->trainSet[i];
		Mat_<int> responses(img.isKeypointCar.size(),1);
		for (size_t j=0;j<img.isKeypointCar.size();++j) {
			responses(j) = img.isKeypointCar[j]?1:0;
		}
		trainFeatures.push_back(img.keypointDesc);
		trainLabels.push_back(responses);
	}
	cout << "Training SVM..." << endl;
	classifier.train(trainFeatures,trainLabels);
	cout << "DONE" << endl;
}

Mat SIFTKNNSegmenter::apply(Mat image) {
	cout << "Classifying KeyPoints..." << endl;
	CarImage img;
	img.image = image;
	detector->detect(img.image,img.keypoints);
	extractor->compute(img.image,img.keypoints,img.keypointDesc);
	Mat_<float> results;
	classifier.predict(img.keypointDesc,results);
	for (size_t i=0;i<img.keypoints.size();++i) {
		img.isKeypointCar.push_back(results(i,1)==1.0);
	}
	cout << "DONE" << endl;

	cout << "Applying KNN..." << endl;
	// KNN
	for (size_t y=0;y<image.rows;++y) {
		for (size_t x=0;x<image.cols;++x) {
			priority_queue<KeypointKNN> points;

			for (size_t i=0;i<img.keypoints.size();++i) {
				Point pt = img.keypoints[i].pt;
				double dist = (x-pt.x)*(x-pt.x)+(y-pt.y)*(y-pt.y);
				KeypointKNN kp;
				kp.dist = dist;
				kp.value = img.isKeypointCar[i];
				points.push(kp);
			}
			size_t trueVotes=0;
			size_t falseVotes=0;
			for (size_t i=0;i<KNN_K && points.size()>0;++i) {
				points.top().value?trueVotes++:falseVotes++;
				points.pop();
				//cout << points.top().dist << endl;
			}
			image.at<Vec3b>(y,x)[(trueVotes>falseVotes)?1:2]=255;
		}
	}
	cout << "DONE" << endl;
	return image;
}