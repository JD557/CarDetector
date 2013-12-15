#include "segmenter.hpp"
#include <fstream>
#include <iostream>
#include <cmath>
#include <queue>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
using namespace std;
using namespace cv;

#define SVM_ITER_LIMIT 50000
#define IMG_PATCH_SIZE 256
#define WORD_COUNT 500
#define WIN_SLIDE 128

bool fileExists(string filename) {
	ifstream file(filename.c_str());
	if (file.is_open()) {
		file.close();
		return true;
	}
	file.close();
	return false;
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

// BOW

void BOWSegmenter::train(vector<CarImage> trainSet) {
	cout << "Training Vocabulary..." << endl;
	this->trainSet=trainSet;
	findKeyPoints();
	setKeyPointLabels();
	
	if (fileExists(bowFilename)) {
		FileStorage fs; 
		fs.open(bowFilename, FileStorage::READ);
		fs["Vocabulary"] >> vocabulary;
		fs.release();

	}
	else {
		for (size_t i=0;i<this->trainSet.size();++i) {
			bowTrainer->add(this->trainSet[i].keypointDesc);
		}
		cout << "Clustering... " << endl;
		vocabulary = bowTrainer->cluster();
		cout << "DONE" << endl;
		FileStorage fs; 
		fs.open(bowFilename, FileStorage::WRITE);
		fs << "Vocabulary" << vocabulary ;
		fs.release();
	}

	bowExtractor->setVocabulary(vocabulary);
	cout << "DONE" << endl;
	// Checks if SVM is already trained
	if (fileExists(svmFilename)) {
		classifier.load(svmFilename.c_str());
	}
	else {

		cout << "Extracting Patch Features" << endl;

		Mat trainFeatures;
		Mat trainLabels;

		for (size_t i=0;i<this->trainSet.size();++i) {
			cout << i+1 << "/" << this->trainSet.size() << endl;
			vector<KeyPoint> carPoints;
			vector<KeyPoint> noncarPoints;
			for (size_t j=0;j<this->trainSet[i].keypoints.size();++j) {
				bool isCar = this->trainSet[i].isKeypointCar[j];
				if (isCar) {
					carPoints.push_back(this->trainSet[i].keypoints[j]);
				}
				else {
					noncarPoints.push_back(this->trainSet[i].keypoints[j]);
				}
			}
			Mat histCar;
			bowExtractor->compute(this->trainSet[i].image,carPoints,histCar);
			if (histCar.rows>0) {
				trainFeatures.push_back(histCar);
				trainLabels.push_back(1.0);
			}
			Mat histNoncar;
			bowExtractor->compute(this->trainSet[i].image,noncarPoints,histNoncar);
			if (histNoncar.rows>0) {
				trainFeatures.push_back(histNoncar);
				trainLabels.push_back(0.0);
			}
		}
		cout << "DONE" << endl;

		cout << "Training SVM..." << endl;
		Mat varIdx;
		Mat samIdx;
		CvSVMParams params;
		params.term_crit = cvTermCriteria( CV_TERMCRIT_ITER+CV_TERMCRIT_EPS, SVM_ITER_LIMIT, FLT_EPSILON );

		Mat trainFeatures_32f;
		Mat trainLabels_32f;
		trainFeatures.convertTo(trainFeatures_32f, CV_32F);
		trainLabels.convertTo(trainLabels_32f, CV_32F);

		classifier.train_auto(trainFeatures_32f,trainLabels_32f,varIdx,samIdx,params);
		classifier.save(svmFilename.c_str());
		cout << "DONE" << endl;
	}

}

Mat BOWSegmenter::apply(Mat image) {
	Mat finalImage(image.rows,image.cols,CV_8U);
	for (size_t y=0;y<image.rows;++y) {
		for (size_t x=0;x<image.cols;++x) {
			finalImage.at<unsigned char>(y,x)=0;
		}
	}
	bowExtractor->setVocabulary(vocabulary);

	cout << "Extracting Patch Features" << endl;

	// Get number of patches
	size_t patchesW = (image.cols-1)/IMG_PATCH_SIZE + 1;
	size_t patchesH = (image.rows-1)/IMG_PATCH_SIZE + 1;

	// Update patch size
	size_t patchW = image.cols/patchesW;
	size_t patchH = image.rows/patchesH;

	// Check patches
	for (size_t y=0;y+patchH<=image.rows;y+=WIN_SLIDE) {
		for (size_t x=0;x+patchW<=image.cols;x+=WIN_SLIDE) {
			Mat patch = image(Rect(x,y,patchW,patchH));
			vector<KeyPoint> kp;
			Mat hist;
			detector->detect(patch,kp);
			bowExtractor->compute(patch,kp,hist);
			Mat hist_32f;
			hist.convertTo(hist_32f, CV_32F);
			bool isCar=hist.rows>0 && classifier.predict(hist_32f)==1.0;
			if (isCar) {
				for (size_t i=0;i<patchH;++i) {
					for (size_t j=0;j<patchW;++j) {
						finalImage.at<unsigned char>(y+i,x+j)=255;
					}
				}
			}
		}
	}

	cout << "DONE" << endl;
	return finalImage;
}

SIFTBOWSegmenter::SIFTBOWSegmenter() {
	detector = FeatureDetector::create("SIFT");
	extractor = DescriptorExtractor::create("SIFT");
	matcher = DescriptorMatcher::create("BruteForce");
	bowTrainer = new BOWKMeansTrainer(WORD_COUNT, TermCriteria(), 1, KMEANS_PP_CENTERS);
	bowExtractor = new BOWImgDescriptorExtractor(detector, matcher);
	svmFilename = "models/siftsvm.xml";
	bowFilename = "models/siftbow.xml";
}

SURFBOWSegmenter::SURFBOWSegmenter() {
	detector = FeatureDetector::create("SURF");
	extractor = DescriptorExtractor::create("SURF");
	matcher = DescriptorMatcher::create("BruteForce");
	bowTrainer = new BOWKMeansTrainer(WORD_COUNT, TermCriteria(), 1, KMEANS_PP_CENTERS);
	bowExtractor = new BOWImgDescriptorExtractor(extractor, matcher);
	svmFilename = "models/surfsvm.xml";
	bowFilename = "models/surfbow.xml";
}

FASTBOWSegmenter::FASTBOWSegmenter() {
	detector = FeatureDetector::create("FAST");
	extractor = DescriptorExtractor::create("SURF");
	matcher = DescriptorMatcher::create("BruteForce");
	bowTrainer = new BOWKMeansTrainer(WORD_COUNT, TermCriteria(), 1, KMEANS_PP_CENTERS);
	bowExtractor = new BOWImgDescriptorExtractor(extractor, matcher);
	svmFilename = "models/fastsvm.xml";
	bowFilename = "models/fastbow.xml";
}