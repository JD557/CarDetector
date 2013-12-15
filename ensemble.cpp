#include "ensemble.hpp"
using namespace std;

Ensemble::Ensemble(vector<Segmenter*> models) {
	this->models = models;
}

void Ensemble::train(vector<CarImage> trainSet) {
	for (size_t i=0;i<models.size();++i) {
		models[i]->train(trainSet);
	}
}

MajorityVoter::MajorityVoter(vector<Segmenter*> models) : Ensemble(models) {}

Mat MajorityVoter::apply(Mat image) {
	vector<Mat> results;
	for (size_t i=0;i<models.size();++i) {
		results.push_back(models[i]->apply(image));
	}

	Mat finalImage(image.rows,image.cols,CV_8U);

	for (size_t y=0;y<image.rows;++y) {
		for (size_t x=0;x<image.cols;++x) {
			size_t votes = 0;
			for (size_t i=0;i<results.size();++i) {
				if (results[i].at<unsigned char>(y,x)==255) {
					votes++;
				}
			}
			finalImage.at<unsigned char>(y,x)=(votes>models.size()/2?255:0);
		}
	}

	return finalImage;
}