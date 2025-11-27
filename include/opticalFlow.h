#ifndef OPTICALFLOW_H_
#define OPTICALFLOW_H_

#include <opencv2/opencv.hpp>

/**
 * @brief Default optical flow parameters used by Farneback's algorithm.
 *
 * These are exposed as constants for configuration clarity and to allow
 * consistent tuning across all optical flow computations.
 */
static const double OF_PYR_SCALE  = 0.5;
static const int    OF_LEVELS     = 3;
static const int    OF_WINSIZE    = 15;
static const int    OF_ITERATIONS = 3;
static const int    OF_POLY_N     = 5;
static const double OF_POLY_SIGMA = 1.2;
static const int    OF_FLAGS      = 0;

/**
 * @brief Computes dense optical flow between two consecutive frames in a video.
 *
 * The function reads two frames starting from startFrame using the provided
 * cv::VideoCapture and applies Farneback's dense optical flow algorithm. The
 * resulting flow field contains, for every pixel, a 2-component vector
 * describing its motion.
 *
 * @param cap         Reference to an opened cv::VideoCapture.
 * @param startFrame  Frame index at which to begin reading frames for flow.
 *
 * @return A CV_32FC2 matrix representing dense optical flow between the frames.
 *
 * @throws std::runtime_error If either frame cannot be read successfully.
 *
 * @note cv::VideoCapture read position will be reset to 0 on completion.
 */
cv::Mat computeOpticalFlow(cv::VideoCapture& cap, int startFrame);

/**
 * @brief Computes dense optical flow between two consecutive frames in a video.
 *
 * The function reads two frames starting from startFrame using the provided
 * cv::VideoCapture and applies Farneback's dense optical flow algorithm. The
 * resulting flow field contains, for every pixel, a 2-component vector
 * describing its motion.
 *
 * @param frame1  First input frame (BGR or grayscale).
 * @param frame2  Second input frame (same size and type as frame1).
 *
 * @return A CV_32FC2 matrix representing dense optical flow between the frames.
 */
cv::Mat computeOpticalFlow(const cv::Mat& frame1, const cv::Mat& frame2);

/**
 * @brief Generates a binary mask from an optical flow field by thresholding
 *        the motion vector magnitude.
 *
 * The function splits the flow into its horizontal and vertical components,
 * computes magnitude at each pixel, normalizes by the maximum magnitude in
 * the frame, and thresholds values above motionThreshold to identify
 * moving pixels.
 *
 * @param flow             A CV_32FC2 optical flow matrix.
 * @param motionThreshold  Normalized threshold for marking motion (default = 0.5).
 *
 * @return A CV_8UC1 binary mask where moving pixels are set to 1.
 */
cv::Mat computeMotionMask(const cv::Mat& flow, double motionThreshold = 0.5);

/**
 * @brief Produces a binary color mask highlighting regions similar to a white
 *        or yellow golf ball.
 *
 * The input frame is converted to HSV. Two color ranges are applied:
 * 
 *   - Low saturation, high brightness -> white golf balls
 * 
 *   - Med-High saturation, high brightness -> yellow golf balls
 *
 * The masks are combined using cv::bitwise_or().
 *
 * @param frame  Input BGR frame.
 *
 * @return A CV_8UC1 binary mask where colored pixels are set to 1.
 */
cv::Mat computeColorMask(const cv::Mat& frame);

#endif