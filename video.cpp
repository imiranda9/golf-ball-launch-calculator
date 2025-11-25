#include "video.h"

bool isLandscape(cv::InputArray img) {
    cv::Mat m = img.getMat();
    return m.cols > m.rows;
}

bool isLandscape(cv::VideoCapture cap) {
    cv::Mat frame;

    if (!cap.read(frame))
        throw std::runtime_error("\n[isLandscape]: Failed to read frame.");

    return frame.cols > frame.rows;
}

double computeFPS(cv::VideoCapture& cap) {
    double metaFPS = cap.get(cv::CAP_PROP_FPS);
    int frameCount = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_COUNT));
    std::cout << "Total Frames: " << frameCount << std::endl;

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
        throw std::runtime_error("\n[computeFPS]: Video data or read position invalid.");

    double calculatedFPS = frameCount / durationSec;

    // Reset read position
    cap.set(cv::CAP_PROP_POS_FRAMES, 0);
    std::cout << "FPS (metadata):   " << metaFPS << std::endl;
    std::cout << "FPS (calculated): " << calculatedFPS << std::endl;

    return (std::abs(calculatedFPS - metaFPS) > 5 ? calculatedFPS : metaFPS);
}

void playVideo(cv::VideoCapture& cap) {
    cv::Mat frame, small;

    while (true) {
        if (!cap.read(frame) || frame.empty()) break;

        // if (isLandscape(frame))
        //     cv::rotate(frame, frame, cv::ROTATE_90_CLOCKWISE);

        cv::resize(frame, small, cv::Size(), 0.4, 0.4);

        cv::imshow("Video", small);

        // [esc] button to close window
        if (cv::waitKey(1) == 27) break;
    }
}

void displayFrame(cv:: VideoCapture& cap, int frameIndex) {
    cv::Mat frame;
    cap.set(cv::CAP_PROP_POS_FRAMES, frameIndex);
    cap.read(frame);

    if (isLandscape(cap))
        cv::rotate(frame, frame, cv::ROTATE_90_CLOCKWISE);

    cv::resize(frame, frame, cv::Size(), 0.4, 0.4);
    cv::imshow("Frame", frame);
    cv::waitKey(0);
}