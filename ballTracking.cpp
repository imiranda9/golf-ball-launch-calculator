#include "ballTracking.h"

// Optical flow parameters stored as constants for clarity
static const double OF_PYR_SCALE  = 0.5;
static const int    OF_LEVELS     = 3;
static const int    OF_WINSIZE    = 15;
static const int    OF_ITERATIONS = 3;
static const int    OF_POLY_N     = 5;
static const double OF_POLY_SIGMA = 1.2;
static const int    OF_FLAGS      = 0;

int findImpactFrameIndex(cv::VideoCapture& cap, double fps, double motionThreshold) {
    const double AVG_BACKSWING_SEC = 0.75;
    const int SKIP = fps * AVG_BACKSWING_SEC;
    const int REQUIRED_CONSECUTIVE_FRAMES = 8;
    const double RESIZE_SCALE = 0.5;

    cv::Mat prev, next, gray1, gray2, flow;

    // Skip backswing motion
    cap.set(cv::CAP_PROP_POS_FRAMES, SKIP);

    if (!cap.read(prev))
        throw std::runtime_error("\n[ERROR]: Video data or read position invalid.\n");

    cv::cvtColor(prev, gray1, cv::COLOR_BGR2GRAY);
    cv::resize(gray1, gray1, cv::Size(), RESIZE_SCALE, RESIZE_SCALE, cv::INTER_AREA);

    int frameIndex = SKIP + 1;
    int consecutiveFrames = 0;

    while (true) {
        if (!cap.read(next)) break;

        cv::cvtColor(next, gray2, cv::COLOR_BGR2GRAY);
        cv::resize(gray2, gray2, cv::Size(), RESIZE_SCALE, RESIZE_SCALE, cv::INTER_AREA);

        cv::calcOpticalFlowFarneback(
            gray1, gray2, flow,
            OF_PYR_SCALE, OF_LEVELS, OF_WINSIZE, OF_ITERATIONS, OF_POLY_N, OF_POLY_SIGMA, OF_FLAGS
        );

        cv::Mat xy[2];
        cv::split(flow, xy);

        cv::Mat mag;
        cv::magnitude(xy[0], xy[1], mag);

        double avgMag = cv::mean(mag)[0];

        std::cout << "Frame: " << frameIndex << " avgMag = " << avgMag << std::endl;

        if (avgMag > motionThreshold) {
            consecutiveFrames++;

            // Consecutive frames must be above motion threshold
            if (consecutiveFrames >= REQUIRED_CONSECUTIVE_FRAMES) {
                return frameIndex;
            }
        }
        else consecutiveFrames = 0;

        gray1 = gray2.clone();
        frameIndex++;
    }

    throw std::logic_error("[ERROR]: Failed to find impact frame.");
}

cv::Mat computeOpticalFlow(cv::VideoCapture& cap) {
    cv::Mat frame1, frame2, gray1, gray2, flow;

    cap.set(cv::CAP_PROP_POS_FRAMES, 0);

    if (!cap.read(frame1) || !cap.read(frame2))
        throw std::runtime_error("\n[ERROR]: Failed to read two frames in optical flow calculation.\n");

    cv::cvtColor(frame1, gray1, cv::COLOR_BGR2GRAY);
    cv::cvtColor(frame2, gray2, cv::COLOR_BGR2GRAY);

    cv::calcOpticalFlowFarneback(
        gray1, gray2, flow,
        OF_PYR_SCALE, OF_LEVELS, OF_WINSIZE, OF_ITERATIONS, OF_POLY_N, OF_POLY_SIGMA, OF_FLAGS
    );

    cap.set(cv::CAP_PROP_POS_FRAMES, 0);
    return flow;
}