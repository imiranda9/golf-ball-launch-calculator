#ifndef BALLTRACKING_H_
#define BALLTRACKING_H_

#include "video.h"
#include <opencv2/opencv.hpp>
#include <opencv2/core/ocl.hpp>
#include <opencv2/tracking.hpp>

int findImpactFrameIndex(cv::VideoCapture& cap, double fps = 240, double motionThreshold = 0.2);

cv::Mat computeOpticalFlow(cv::VideoCapture& cap, int startFrame);

cv::Mat computeOpticalFlow(const cv::Mat& frame1, const cv::Mat& frame2);

cv::Mat computeMotionMask(const cv::Mat& flow, double motionThreshold = 0.5);

cv::Mat computeColorMask(const cv::Mat& frame);

cv::Rect getBoundingBox(cv::VideoCapture& cap, int impactFrameIndex, double motionThreshold = 0.5);

cv::Point2f getBoundingBoxPoint(cv::VideoCapture& cap, int impactFrameIndex, double motionThreshold = 0.5);

// Frame by frame tracking using farneback optical flow, motion masking, and color masking
std::vector<cv::Point2f> trackBallTrajectory(cv::VideoCapture& cap, int startFrame, cv::Rect initialBox);

#endif