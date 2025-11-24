#ifndef BALLTRACKING_H_
#define BALLTRACKING_H_

#include <opencv2/opencv.hpp>
#include <opencv2/core/ocl.hpp>

int findImpactFrameIndex(cv::VideoCapture& cap, double fps = 240, double motionThreshold = 0.2);

cv::Mat computeOpticalFlow(cv::VideoCapture& cap, int startFrame);

cv::Mat computeMotionMask(const cv::Mat& flow, double motionThreshold = 0.5);

cv::Mat computeColorMask(const cv::Mat& frame);

cv::Rect getBoundingBox(cv::VideoCapture& cap,
                        int impactFrameIndex,
                        double motionThreshold = 0.5);

#endif