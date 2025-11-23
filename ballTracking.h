#ifndef BALLTRACKING_H_
#define BALLTRACKING_H_

#include <opencv2/opencv.hpp>
#include <opencv2/core/ocl.hpp>

int findImpactFrameIndex(cv::VideoCapture& cap, double fps=240, double motionThreshold=0.2);

cv::Mat computeOpticalFlow(cv::VideoCapture& cap);

#endif