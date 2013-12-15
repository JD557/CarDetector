#ifndef _ENSEMBLE_H_
#define _ENSEMBLE_H_

#include "segmenter.hpp"
#include <vector>
using namespace std;

class Ensemble {
	protected:
		vector<Segmenter*> models;
	public:
		Ensemble(vector<Segmenter*> models);
		void train(vector<CarImage> trainSet);
		Mat virtual apply(Mat image) = 0;
};


class MajorityVoter:public Ensemble {
	public:
		MajorityVoter(vector<Segmenter*> models);
		Mat apply(Mat image);
};

#endif