#ifndef VIDEO_H_
#define VIDEO_H_

#include <opencv2/opencv.hpp>
#include <iostream>

enum Orientation {
    LANDSCAPE = 0,
    PORTRAIT = 1
};

bool isLandscape(cv::InputArray img);

bool isLandscape(cv::VideoCapture cap);

// Calculates true FPS if metadata is incorrect
// Prints meta FPS and calculated FPS to console
double computeFPS(cv::VideoCapture& cap);

// Rotates video 90 degrees and plays in window
// Press [esc] to exit
void playVideo(cv::VideoCapture& cap);

// Rotates image 90 degrees and displays in window
void displayFrame(cv:: VideoCapture& cap, int frameIndex);

#endif