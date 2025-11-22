#ifndef VIDEO_H_
#define VIDEO_H_

#include <opencv2/opencv.hpp>
#include <iostream>

// Calculates true FPS if metadata is incorrect
// Prints meta FPS and calculated FPS to console
double computeFPS(cv::VideoCapture& cap);

// in progress
int findImpactFrameIndex(cv::VideoCapture& cap, double motionThreshold = 5.0);

cv::Mat computeOpticalFlow(cv::VideoCapture& cap);

// Plays video parameter in window, rotated 90 degrees
// Press [esc] to exit window
void playPortraitVideo(cv::VideoCapture& cap);

#endif