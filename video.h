#ifndef VIDEO_H_
#define VIDEO_H_

#include <opencv2/opencv.hpp>
#include <iostream>

// Calculates true FPS if metadata is incorrect
double computeFPS(cv::VideoCapture& cap);

void playVideo(cv::VideoCapture& cap);

void displayFrame(cv:: VideoCapture& cap, int frameIndex);

#endif