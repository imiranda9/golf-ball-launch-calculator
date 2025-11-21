#ifndef VIDEO_H_
#define VIDEO_H_

#include <opencv2/opencv.hpp>
#include <iostream>

// Calculates true FPS if metadata is incorrect
// Prints meta FPS and calculated FPS to console
double computeFPS(cv::VideoCapture& cap);

#endif