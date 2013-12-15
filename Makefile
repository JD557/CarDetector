OPENCV_LIBS = -lopencv_core -lopencv_imgproc -lopencv_highgui -lopencv_ml -lopencv_video -lopencv_nonfree -lopencv_features2d -lopencv_calib3d -lopencv_objdetect -lopencv_contrib -lopencv_legacy -lopencv_flann

cardetection:cardetection.cpp segmenter.cpp ensemble.cpp
	g++ cardetection.cpp segmenter.cpp ensemble.cpp -o cardetection -Wall $(OPENCV_LIBS)

