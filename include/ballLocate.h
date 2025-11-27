#ifndef BALLLOCATE_H_
#define BALLLOCATE_H_

#include <opencv2/opencv.hpp>
#include "opticalFlow.h"

/**
 * @brief Estimates the frame index where the golf ball begins moving after impact.
 *
 * Analyzes optical flow across the video to detect the moment of ball launch. Begins
 * by skipping ~0.75 seconds of backswing motion, then reads consecutive frame
 * pairs, computes dense optical flow via Farneback’s algorithm, and measures the
 * average motion magnitude.
 *
 * A frame is considered part of the “impact window” when its average motion
 * magnitude exceeds motionThreshold. The function requires a minimum number
 * of consecutive high motion frames to avoid false positives caused by camera
 * shake or body movement.
 *
 * @param cap              Reference to an opened cv::VideoCapture.
 * @param fps              Video framerate, used to compute skip duration.
 * @param motionThreshold  Minimum required average flow magnitude to detect ball launch (default = 0.2).
 *
 * @return The frame index where the ball is determined to begin moving.
 *
 * @throws std::runtime_error  If initial frames cannot be read.
 * @throws std::logic_error    If no suitable impact frame is detected.
 *
 * @note cv::VideoCapture read position will be reset to 0 on completion.
 */
int findImpactFrameIndex(cv::VideoCapture& cap, double fps = 240, double motionThreshold = 0.2);

/**
 * @brief Creates a bounding box around the ball at the moment after impact.
 *
 * Isolates the region of the frame where the ball is expected
 * using a right-side crop (bottom 40% of the frame width). Then:
 *
 *  - Computes optical flow for the impactFrame.
 * 
 *  - Computes a motion mask using motion magnitude.
 * 
 *  - Computes a color mask (white/yellow ball pixels).
 * 
 *  - ANDs both masks to isolate pixels that are moving *and* match ball color.
 * 
 *  - Finds contours and selects the largest contour as the ball.
 *
 * The bounding box is returned in full-frame coordinates, adjusting for the ROI.
 *
 * @param cap               Reference to an opened cv::VideoCapture.
 * @param impactFrameIndex  Frame index at which to search for the ball.
 * @param motionThreshold   Normalized threshold for marking motion (default = 0.5).
 *
 * @return A cv::Rect bounding the detected ball.
 *
 * @throws std::runtime_error If the frame cannot be read or no ball-like region is found.
 *
 * @note cv::VideoCapture read position will be reset to 0 on completion.
 */
cv::Rect getBoundingBox(cv::VideoCapture& cap, int impactFrameIndex, double motionThreshold = 0.5);

/**
 * @brief Creates a bounding box around the ball at the moment after impact.
 *
 * Isolates the region of the frame where the ball is expected
 * using a right-side crop (bottom 40% of the frame width). Then:
 *
 *  - Computes optical flow for the impactFrame.
 * 
 *  - Computes a motion mask using motion magnitude.
 * 
 *  - Computes a color mask (white/yellow ball pixels).
 * 
 *  - ANDs both masks to isolate pixels that are moving *and* match ball color.
 * 
 *  - Finds contours and selects the largest contour as the ball.
 *
 * The bounding box is returned in center coordinates, adjusting for the ROI.
 *
 * @param cap               Reference to an opened cv::VideoCapture.
 * @param impactFrameIndex  Frame index at which to search for the ball.
 * @param motionThreshold   Normalized threshold for marking motion (default = 0.5).
 *
 * @return A cv::Point2f containing the coordinates of the center of the detected ball.
 *
 * @throws std::runtime_error If the frame cannot be read or no ball-like region is found.
 *
 * @note cv::VideoCapture read position will be reset to 0 on completion.
 */
cv::Point2f getBoundingBoxCenter(cv::VideoCapture& cap, int impactFrameIndex, double motionThreshold = 0.5);

#endif