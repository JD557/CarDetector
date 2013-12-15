#ifndef _SEGMENTER_H_
#define _SEGMENTER_H_

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/contrib/contrib.hpp>
#include <opencv2/nonfree/nonfree.hpp>
#include <opencv2/nonfree/features2d.hpp>
#include <opencv2/ml/ml.hpp>
#include <string>
using namespace cv;

struct CarImage {
	Mat image;
	vector<Mat> masks;
	vector<KeyPoint> keypoints;
	Mat keypointDesc;
	vector<bool> isKeypointCar;
};

class Segmenter {
	protected:
		Ptr<FeatureDetector> detector;
		Ptr<DescriptorExtractor> extractor;
		vector<CarImage> trainSet;
		CvSVM classifier;
		string svmFilename;

		void findKeyPoints();
		void setKeyPointLabels();
	public:
		void virtual train(vector<CarImage> trainSet) = 0;
		Mat virtual apply(Mat image) = 0;
};


class BOWSegmenter:public Segmenter {
	protected:
		Ptr<DescriptorMatcher> matcher;
		Ptr<BOWKMeansTrainer> bowTrainer;
		Ptr<BOWImgDescriptorExtractor> bowExtractor;
		Mat vocabulary;
		string bowFilename;
	public:
		Mat apply(Mat image);
		void train(vector<CarImage> trainSet);
};

class SIFTBOWSegmenter:public BOWSegmenter {
	public:
		SIFTBOWSegmenter();
};

class SURFBOWSegmenter:public BOWSegmenter {
	public:
		SURFBOWSegmenter();
};

class FASTBOWSegmenter:public BOWSegmenter {
	public:
		FASTBOWSegmenter();
};

#endif