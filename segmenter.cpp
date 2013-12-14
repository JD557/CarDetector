#include "segmenter.hpp"
#include <iostream>
#include <cmath>
#include <queue>
using namespace std;
using namespace cv;

#define SVM_ITER_LIMIT 10000
#define IMG_PATCH_SIZE 128
#define WORD_COUNT 200
#define WIN_SLIDE 32
#define KNN_K 3
bool operator<(const KeypointKNN a,const KeypointKNN b) {
	return a.dist>b.dist;
}

/*CarImage::CarImage() {}

CarImage::CarImage(const CarImage& ci) {
	this->image         = ci.image.clone();
	for (size_t i=0;i<ci.masks.size();++i) {
		this->masks.push_back(ci.masks[i].clone());
	}
	this->keypoints     = ci.keypoints;
	this->keypointDesc  = ci.keypointDesc.clone();
	this->isKeypointCar = ci.isKeypointCar;
}

CarImage& CarImage::operator=(const CarImage& ci) {
	this->image         = ci.image.clone();
	for (size_t i=0;i<ci.masks.size();++i) {
		this->masks.push_back(ci.masks[i].clone());
	}
	this->keypoints     = ci.keypoints;
	this->keypointDesc  = ci.keypointDesc.clone();
	this->isKeypointCar = ci.isKeypointCar;
	return *this;
}*/

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

// KNN

void KNNSegmenter::train(vector<CarImage> trainSet) {
	this->trainSet=trainSet;
	findKeyPoints();
	setKeyPointLabels();

	Mat trainFeatures(0,0,CV_32F);
	Mat trainLabels(0,1,CV_32F);
	for (size_t i=0;i<this->trainSet.size();++i) {
		CarImage img = this->trainSet[i];
		Mat responses(img.isKeypointCar.size(),1,CV_32F);
		for (size_t j=0;j<img.isKeypointCar.size();++j) {
			responses.at<float>(j) = img.isKeypointCar[j]?1.0:0.0;
		}
		trainFeatures.push_back(img.keypointDesc);
		trainLabels.push_back(responses);
	}
	cout << "Training SVM..." << endl;
	//classifier.train(trainFeatures,trainLabels);
	Mat varIdx;
	Mat samIdx;
	CvSVMParams params;
	params.term_crit = cvTermCriteria( CV_TERMCRIT_ITER+CV_TERMCRIT_EPS, SVM_ITER_LIMIT, FLT_EPSILON );

	classifier.train(trainFeatures,trainLabels,varIdx,samIdx,params);
	cout << "DONE" << endl;
}

Mat KNNSegmenter::apply(Mat image) {
	cout << "Classifying KeyPoints..." << endl;
	CarImage img;
	img.image = image.clone();
	detector->detect(img.image,img.keypoints);
	extractor->compute(img.image,img.keypoints,img.keypointDesc);
	Mat results;
	//classifier.predict(img.keypointDesc,results);
	//cout << "KEYPOINTS:" << img.keypoints.size() << endl;
	//cout << "DESCRIPTIONS:" << img.keypointDesc.rows << endl;
	//double sum=0;
	for (size_t i=0;i<img.keypoints.size();++i) {
		//img.isKeypointCar.push_back(results.at<float>(i)==1.0);
		//sum+=results.at<float>(i);
		img.isKeypointCar.push_back(classifier.predict(img.keypointDesc.row(i))==1.0);
	}
	//cout << "RESULTS:" << results.rows << " " << img.isKeypointCar.size() << endl;
	//cout << "SUM:" << sum << endl;
	cout << "DONE" << endl;

	Mat finalImage = image.clone();
	/*cout << "Applying KNN..." << endl;
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
			}
			image.at<Vec3b>(y,x)[(trueVotes>falseVotes)?1:2]=255;
		}
	}*/

	for (size_t i=0;i<img.keypoints.size();++i) {
		bool isCar = img.isKeypointCar[i];
		circle(finalImage,img.keypoints[i].pt,5,isCar?Scalar(0,255,0):Scalar(0,0,255),2);
	}
	cout << "DONE" << endl;
	return finalImage;
}

SIFTKNNSegmenter::SIFTKNNSegmenter() {
	detector = FeatureDetector::create("SIFT");
	extractor = DescriptorExtractor::create("SIFT");
}

SURFKNNSegmenter::SURFKNNSegmenter() {
	detector = FeatureDetector::create("SURF");
	extractor = DescriptorExtractor::create("SURF");
}

// BOW

void BOWSegmenter::train(vector<CarImage> trainSet) {
	cout << "Training Vocabulary..." << endl;
	this->trainSet=trainSet;
	findKeyPoints();
	setKeyPointLabels();
    
    for (size_t i=0;i<this->trainSet.size();++i) {
        bowTrainer->add(this->trainSet[i].keypointDesc);
    }
    vocabulary = bowTrainer->cluster();
    bowExtractor->setVocabulary(vocabulary);
    cout << "DONE" << endl;

    cout << "Extracting Patch Features" << endl;

    Mat trainFeatures;
    Mat trainLabels;


    for (size_t i=0;i<this->trainSet.size();++i) {
    	// Get number of patches
		size_t patchesW = (this->trainSet[i].image.cols-1)/IMG_PATCH_SIZE + 1;
		size_t patchesH = (this->trainSet[i].image.rows-1)/IMG_PATCH_SIZE + 1;

		// Update patch size
		size_t patchW = this->trainSet[i].image.cols/patchesW;
		size_t patchH = this->trainSet[i].image.rows/patchesH;

		// Train patches
		for (size_t y=0;y+patchH<=this->trainSet[i].image.rows;y+=WIN_SLIDE) {
			for (size_t x=0;x+patchW<=this->trainSet[i].image.cols;x+=WIN_SLIDE) {
				Mat patch = this->trainSet[i].image(Rect(x,y,patchW,patchH));
				vector<KeyPoint> kp;
				bool hasCar = false;
				for (size_t j=0;j<this->trainSet[i].keypoints.size();++j) {
					Point pt = this->trainSet[i].keypoints[j].pt;
					if (pt.x>=x && pt.y>=y && pt.x<x+patchW && pt.y<y+patchH) {
						kp.push_back(this->trainSet[i].keypoints[j]);
						if (this->trainSet[i].isKeypointCar[j]) {
							hasCar=true;
						}
					}
				}
				Mat hist;
				detector->detect(patch,kp);
				bowExtractor->compute(patch,kp,hist);
				if (hist.rows>0) {
					trainFeatures.push_back(hist);
					trainLabels.push_back(hasCar?1.0:0.0);
				}
			}
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

	classifier.train(trainFeatures_32f,trainLabels_32f,varIdx,samIdx,params);
	cout << "DONE" << endl;

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
	matcher = DescriptorMatcher::create("FlannBased");
	bowTrainer = new BOWKMeansTrainer(WORD_COUNT, TermCriteria(), 1, KMEANS_PP_CENTERS);
	bowExtractor = new BOWImgDescriptorExtractor(detector, matcher);
}

SURFBOWSegmenter::SURFBOWSegmenter() {
	detector = FeatureDetector::create("SURF");
	extractor = DescriptorExtractor::create("SURF");
	matcher = DescriptorMatcher::create("FlannBased");
	bowTrainer = new BOWKMeansTrainer(WORD_COUNT, TermCriteria(), 1, KMEANS_PP_CENTERS);
	bowExtractor = new BOWImgDescriptorExtractor(extractor, matcher);
}