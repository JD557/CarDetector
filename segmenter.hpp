#ifndef _SEGMENTER_H_
#define _SEGMENTER_H_

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/contrib/contrib.hpp>
#include <opencv2/nonfree/nonfree.hpp>
#include <opencv2/nonfree/features2d.hpp>
#include <opencv2/ml/ml.hpp>
using namespace cv;

struct CarImage {
	Mat image;
	vector<Mat> masks;
	vector<KeyPoint> keypoints;
	Mat keypointDesc;
	vector<bool> isKeypointCar;
	/*CarImage& operator=(const CarImage& ci);
	CarImage();
	CarImage(const CarImage& ci);*/
};

struct KeypointKNN {
	double dist;
	bool value;
};

bool operator<(const KeypointKNN a,const KeypointKNN b);

class Segmenter {
	protected:
		Ptr<FeatureDetector> detector;
		Ptr<DescriptorExtractor> extractor;
		vector<CarImage> trainSet;
		CvSVM classifier;

		void findKeyPoints();
		void setKeyPointLabels();
	public:
		void virtual train(vector<CarImage> trainSet) = 0;
		Mat virtual apply(Mat image) = 0;
};

class KNNSegmenter:public Segmenter {
	public:
		Mat apply(Mat image);
		void train(vector<CarImage> trainSet);
};

class SIFTKNNSegmenter:public KNNSegmenter {
	public:
		SIFTKNNSegmenter();
};

class SURFKNNSegmenter:public KNNSegmenter {
	public:
		SURFKNNSegmenter();
};

class BOWSegmenter:public Segmenter {
	protected:
		Ptr<DescriptorMatcher> matcher;
		Ptr<BOWKMeansTrainer> bowTrainer;
		Ptr<BOWImgDescriptorExtractor> bowExtractor;
		Mat vocabulary;
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

#endif