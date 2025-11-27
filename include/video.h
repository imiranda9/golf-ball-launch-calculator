#ifndef VIDEO_H_
#define VIDEO_H_

#include <opencv2/opencv.hpp>

/**
 * @brief Computes the actual frames-per-second (FPS) of a video.
 *
 * Retrieves  metadata FPS, then measures the timestamp difference
 * the real FPS. If the difference between calculated FPS and metadata
 * FPS is significant, the calculated value is returned instead.
 *
 * @param cap  Reference to an opened cv::VideoCapture object.
 *
 * @return The FPS of the video.
 * 
 * @throws std::runtime_error If the video timestamps cannot be read.
 */
double computeFPS(cv::VideoCapture &cap);

/**
 * @brief Plays the provided video in a display window.
 *
 * This function iterates through all frames in the video and displays them
 * at a constant rate. Useful for debugging or previewing raw input footage.
 *
 * @param cap  Reference to an opened cv::VideoCapture object.
 * 
 * @note cv::VideoCapture read position will be reset to 0 on completion.
 */
void playVideo(cv::VideoCapture &cap);

/**
 * @brief Displays a single frame at the specified index.
 *
 * The function seeks to given  frame, reads it, and shows it in a window.
 * Useful for inspecting specific moments in the video.
 *
 * @param cap        Reference to an opened cv::VideoCapture object.
 * @param frameIndex Index of the frame to show.
 * 
 * @throws std::runtime_error If the frame cannot be read.
 *
 * @note cv::VideoCapture read position will be reset to 0 on completion.
 */
void displayFrame(cv::VideoCapture &cap, int frameIndex);

#endif