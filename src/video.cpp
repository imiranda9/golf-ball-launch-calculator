#include "video.h"

double computeFPS(cv::VideoCapture& cap) {
    double metaFPS = cap.get(cv::CAP_PROP_FPS);
    int frameCount = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_COUNT));

    // Move read position to end of video
    cap.set(cv::CAP_PROP_POS_FRAMES, 1.0);

    double timestampFirst = cap.get(cv::CAP_PROP_POS_MSEC);
    double timestampLast = timestampFirst;

    cv::Mat frame;
    while (true) {
        if (!cap.read(frame) || frame.empty()) break;
        timestampLast = cap.get(cv::CAP_PROP_POS_MSEC);
    }
    double durationSec = (timestampLast - timestampFirst) / 1000;
    if (durationSec <= 0)
        throw std::runtime_error("\n[computeFPS]: Video data or read position invalid.");

    double calculatedFPS = frameCount / durationSec;

    cap.set(cv::CAP_PROP_POS_FRAMES, 0);

    return (std::abs(calculatedFPS - metaFPS) > 5 ? calculatedFPS : metaFPS);
}

void playVideo(cv::VideoCapture& cap) {
    cv::Mat frame, small;

    while (true) {
        if (!cap.read(frame) || frame.empty()) break;

        cv::resize(frame, small, cv::Size(), 0.4, 0.4);
        cv::imshow("Video", small);
        if (cv::waitKey(1) == 27) break;
    }

    cap.set(cv::CAP_PROP_POS_FRAMES, 0);
}

void displayFrame(cv:: VideoCapture& cap, int frameIndex) {
    cap.set(cv::CAP_PROP_POS_FRAMES, frameIndex);
    cv::Mat frame;
    if (!cap.read(frame) || frame.empty())
        throw std::runtime_error("\n[displayFrame]: Could not read frame.");

    cv::resize(frame, frame, cv::Size(), 0.4, 0.4);
    cv::imshow("Frame", frame);
    cv::waitKey(0);
    
    cap.set(cv::CAP_PROP_POS_FRAMES, 0);
}