#include "video.h"

// Optical flow parameters stored as constants for clarity and easy tuning
static const double OF_PYR_SCALE  = 0.5;
static const int    OF_LEVELS     = 3;
static const int    OF_WINSIZE    = 15;
static const int    OF_ITERATIONS = 3;
static const int    OF_POLY_N     = 5;
static const double OF_POLY_SIGMA = 1.2;
static const int    OF_FLAGS      = 0;

double computeFPS(cv::VideoCapture& cap) {
    double metaFPS = cap.get(cv::CAP_PROP_FPS);
    int frameCount = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_COUNT));

    // Move read position to end of video
    cap.set(cv::CAP_PROP_POS_FRAMES, 1.0);

    // Calculate vid duration
    double timestampFirst = cap.get(cv::CAP_PROP_POS_MSEC);
    double timestampLast = timestampFirst;

    cv::Mat frame;
    while (true) {
        if (!cap.read(frame) || frame.empty()) break;
        timestampLast = cap.get(cv::CAP_PROP_POS_MSEC);
    }
    double durationSec = (timestampLast - timestampFirst) / 1000;
    if (durationSec <= 0)
        throw std::runtime_error("\n[ERROR]: Video data or read position invalid.\n");

    double calculatedFPS = frameCount / durationSec;

    // Reset read position
    cap.set(cv::CAP_PROP_POS_FRAMES, 0);
    std::cout << "FPS (metadata):   " << metaFPS << std::endl;
    std::cout << "FPS (calculated): " << calculatedFPS << std::endl;

    return (std::abs(calculatedFPS - metaFPS) > 5 ? calculatedFPS : metaFPS);
}

// in progress
int findImpactFrameIndex(cv::VideoCapture& cap, double motionThreshold) {
    cv::Mat prev, next, gray1, gray2, flow;
    cap.set(cv::CAP_PROP_POS_FRAMES, 0);

    if (!cap.read(prev)) return -1;
    cv::cvtColor(prev, gray1, cv::COLOR_BGR2GRAY);

    int frameIndex = 1;

    while (true) {
        if (!cap.read(next)) break;

        cv::cvtColor(next, gray2, cv::COLOR_BGR2GRAY);

        cv::calcOpticalFlowFarneback(
            gray1, gray2, flow,
            OF_PYR_SCALE, OF_LEVELS, OF_WINSIZE, OF_ITERATIONS, OF_POLY_N, OF_POLY_SIGMA, OF_FLAGS
        );

        cv::Mat xy[2];
        cv::split(flow, xy);

        cv::Mat mag;
        cv::magnitude(xy[0], xy[1], mag);

        double maxMag;
        cv::minMaxLoc(mag, nullptr, &maxMag);

        if (maxMag > motionThreshold) {
            return frameIndex;
        }

        gray1 = gray2.clone();
        frameIndex++;
    }

    return -1;
}

cv::Mat computeOpticalFlow(cv::VideoCapture& cap) {
    cv::Mat frame1, frame2, gray1, gray2, flow;

    cap.set(cv::CAP_PROP_POS_FRAMES, 0);

    if (!cap.read(frame1) || !cap.read(frame2))
        throw std::runtime_error("\n[ERROR]: Failed to read two frames in optical flow calculation.\n");

    cv::cvtColor(frame1, gray1, cv::COLOR_BGR2GRAY);
    cv::cvtColor(frame2, gray2, cv::COLOR_BGR2GRAY);

    cv::calcOpticalFlowFarneback(gray1, gray2, flow, 0.5, 3, 15, 3, 5, 1.2, 0);

    cap.set(cv::CAP_PROP_POS_FRAMES, 0);
    return flow;
}

void playPortraitVideo(cv::VideoCapture& cap) {
    cv::Mat frame, small;

    while (true) {
        if (!cap.read(frame) || frame.empty()) break;

        cv::rotate(frame, frame, cv::ROTATE_90_CLOCKWISE);
        cv::resize(frame, small, cv::Size(), 0.4, 0.4);

        cv::imshow("", small);

        // [esc] button to close window
        if (cv::waitKey(1) == 27) break;
    }
}