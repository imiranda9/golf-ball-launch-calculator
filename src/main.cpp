/**
 * @file main.cpp
 * @brief Full golf ball tracking pipeline.
 *
 * This program performs the following steps:
 *  1. Enables OpenCL acceleration in OpenCV (if available).
 *  2. Opens the input video file (sample_video.mov).
 *  3. Computes the true video FPS using timestamp analysis.
 *  4. Detects the impact frame where the golf ball begins moving.
 *  5. Computes an initial bounding box around the ball at impact.
 *  6. Tracks the ball forward through the video to build a trajectory.
 *  7. Rotates tracked pointsand filters noisy points.
 *  8. Estimates carry distance and launch angle from the trajectory.
 *
 * The results (carry distance in meters and launch angle in degrees)
 * are printed to stdout.
 */

#include <iostream>
#include <opencv2/core/ocl.hpp>
#include "video.h"
#include "ballLocate.h"
#include "ballTracking.h"

int main() {
    try {
        // Use GPU acceleration if possible
        cv::ocl::setUseOpenCL(true);

        cv::VideoCapture vid("../assets/sample_video.mov");
        if (!vid.isOpened())
            throw std::runtime_error("[main]: Failed to open video.");

        int frameHeight = vid.get(cv::CAP_PROP_FRAME_HEIGHT);

        std::cout << "Computing FPS:            ";
        int fps = computeFPS(vid);
        std::cout << "DONE\n";

        std::cout << "Finding Impact Frame:     ";
        int impactIndex = findImpactFrameIndex(vid);
        std::cout << "DONE\n";

        std::cout << "Creating Bounding Box:    ";
        cv::Rect boundingBox = getBoundingBox(vid, impactIndex);
        std::cout << "DONE\n";

        std::cout << "Tracking Ball Trajectory: ";
        std::vector<PosFrame> trackedPoints = getBallTrajectory(vid, impactIndex);
        rotatePointVector(trackedPoints, frameHeight); // Account for OpenCV forced video rotation
        std::vector<PosFrame> trajectory = filterPoints(trackedPoints);
        std::cout << "DONE\n\n";

        float ballDiameter = (boundingBox.width + boundingBox.height) / 2.0;
        float carry = 0.0;
        float angle = 0.0;

        computeCarryAndAngle(trajectory, carry, angle, ballDiameter, fps);

        std::cout << "==============================\n";
        std::cout << "Carry Distance (m): " << carry << std::endl;
        std::cout << "Launch Angle (deg): " << angle << std::endl;
        std::cout << "==============================";
    }
    catch (const std::runtime_error& ex) {
        std::cerr << ex.what();
    }
    catch(const std::logic_error& ex2) {
        std::cerr << ex2.what();
    }

    return 0;
}