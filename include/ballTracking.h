#ifndef BALLTRACKING_H_
#define BALLTRACKING_H_

#include "opticalFlow.h"
#include "ballLocate.h"
#include <opencv2/opencv.hpp>
#include <opencv2/tracking.hpp>

/**
 * @struct PosFrame
 * @brief Stores a detected ball position and its corresponding frame index.
 * 
 * Members:
 * 
 *  - pos        Center of the ball in pixel coordinates.
 * 
 *  - frameIndex Index of the frame in which the position was found.
 */
struct PosFrame {
    cv::Point2f pos;
    int frameIndex;
};

/**
 * @brief Tracks the ball forward from a known starting location.
 *
 * Starting at startFrame, the function reads frames sequentially and attempts
 * to locate the ball by calling getBoundingBoxCenter() on each frame. Each
 * successful detection produces a (position, frameIndex) pair appended to the
 * trajectory vector. Stops tracking when a frame cannot be read, or
 * getBoundingBoxCenter() throws (ball is no longer detectable).
 *
 * @param cap         Reference to an opened cv::VideoCapture.
 * @param startFrame  Frame at which ball tracking begins.
 *
 * @return A vector of PosFrame values representing the tracked ball trajectory.
 * 
 * @note cv::VideoCapture read position will be reset to 0 on completion.
 */
std::vector<PosFrame> getBallTrajectory(cv::VideoCapture& cap, int startFrame);

/**
 * @brief Rotates a position point 90 degrees clockwise.
 * 
 * Used to address how OpenCV forces portrait videos into landscape.
 *
 * @param point  The PosFrame to rotate.
 * @param height Frame height.
 *
 * @return A new PosFrame with rotated pixel coordinates.
 */
PosFrame getRotatedPoint(const PosFrame& point, int height);

/**
 * @brief Applies getRotatedPoint() to an entire std::vector<PosFrame> in-place.
 *
 * @param points Vector of PosFrame values to rotate.
 * @param height Frame height.
 */
void rotatePointVector(std::vector<PosFrame>& points, int height);

/**
 * @brief Filters raw trajectory points using motion constraints.
 *
 * This function removes points that violate expected golf-ball flight
 * characteristics:
 *
 *  - X must strictly increase (ball moves horizontally to the right).
 * 
 *  - Y must strictly decrease (ball rises upward in pixel coordinates).
 *
 * @param points  std::vector of trajectory points returned from getBallTrajectory().
 *
 * @return A cleaned vector of PosFrame values containing only valid motion.
 */
std::vector<PosFrame> filterPoints(const std::vector<PosFrame>& points);

/**
 * @brief Computes estimated carry distance and launch angle from trajectory data.
 *
 * The function estimates velocity components between consecutive trajectory
 * points, averages them, and uses projectile equations to compute:
 *
 *  - Horizontal velocity    vx
 * 
 *  - Vertical velocity      -vy  (sign-corrected so upward motion is positive)
 * 
 *  - Flight time using      t = 2 * vy / g
 * 
 *  - Carry distance:        carry = vx * flightTime
 * 
 *  - Launch angle:          atan2(vy, vx)
 *
 * @param points          Cleaned vector of trajectory points.
 * @param carry           Output: horizontal carry distance in meters.
 * @param angle           Output: launch angle in degrees.
 * @param ballDiameterPx  Ball diameter in pixels for scale conversion.
 * @param fps             Video framerate (default = 240).
 *
 * @throws std::runtime_error If fewer than 2 usable points exist.
 */
void computeCarryAndAngle(const std::vector<PosFrame>& points, float& carry, float& angle, float ballDiameterPx, float fps = 240);

#endif