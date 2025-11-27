#include "ballLocate.h"

int findImpactFrameIndex(cv::VideoCapture& cap, double fps, double motionThreshold) {
    const double AVG_BACKSWING_SEC = 0.75;
    const int SKIP = fps * AVG_BACKSWING_SEC;
    const int REQUIRED_CONSECUTIVE_FRAMES = 8;
    const double RESIZE_SCALE = 0.5;

    cv::Mat prev, next, gray1, gray2, flow;

    // Skip backswing motion
    cap.set(cv::CAP_PROP_POS_FRAMES, SKIP);

    if (!cap.read(prev))
        throw std::runtime_error("\n[findImpactFrameIndex]: Video data or read position invalid.");

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

        // Split motion into two images
        cv::Mat xy[2];
        cv::split(flow, xy);

        // Calculate magnitude of pixel motion vectors
        cv::Mat mag;
        cv::magnitude(xy[0], xy[1], mag);

        // Find average motion vector
        double avgMag = cv::mean(mag)[0];

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

    throw std::logic_error("\n[findImpactFrameIndex]: Failed to find impact frame.");
}

cv::Rect getBoundingBox(cv::VideoCapture& cap, int impactFrameIndex, double motionThreshold) {
    cap.set(cv::CAP_PROP_POS_FRAMES, impactFrameIndex);

    cv::Mat impactFrame;
    if (!cap.read(impactFrame))
        throw std::runtime_error("\n[getBoundingBox]: Failed to read frame.");

    int H = impactFrame.rows;
    int W = impactFrame.cols;

    // Region of interest - bottom 40% of frame
    const float CROP_RATIO = 0.4;
    int roiW = static_cast<int>(W * CROP_RATIO);
    int roiX = W - roiW;
    cv::Rect ROI(roiX, 0, roiW, H);

    cv::Mat flowFull = computeOpticalFlow(cap, impactFrameIndex);
    cv::Mat flow = flowFull(ROI);
    cv::Mat motionMask = computeMotionMask(flow, 0.2);
    cv::Mat colorMaskFull = computeColorMask(impactFrame);
    cv::Mat colorMask = colorMaskFull(ROI);
    cv::Mat combinedMask;
    cv::bitwise_and(motionMask, colorMask, combinedMask);

    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(combinedMask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    if (contours.empty())
        throw std::runtime_error("\n[getBoundingBox]: Ball region not found.");

    // Find largest contour
    int bestIndex = 0;
    double bestArea = 0.0;

    for (int i = 0; i < contours.size(); i++) {
        double area = cv::contourArea(contours[i]);
        if (area > bestArea) {
            bestArea = area;
            bestIndex = i;
        }
    }

    cv::Rect box = cv::boundingRect(contours[bestIndex]);
    box.x += roiX;

    cap.set(cv::CAP_PROP_POS_FRAMES, 0);

    return box;
}

cv::Point2f getBoundingBoxCenter(cv::VideoCapture& cap, int impactFrameIndex, double motionThreshold) {
    cap.set(cv::CAP_PROP_POS_FRAMES, impactFrameIndex);

    cv::Mat impactFrame;
    if (!cap.read(impactFrame))
        throw std::runtime_error("\n[getBoundingBoxCenter]: Failed to read frame.");

    int H = impactFrame.rows;
    int W = impactFrame.cols;

    // Region of interest - bottom 40% of frame
    const float CROP_RATIO = 0.45;
    int roiW = static_cast<int>(W * CROP_RATIO);
    int roiX = W - roiW;
    cv::Rect ROI(roiX, 0, roiW, H);

    cv::Mat flowFull = computeOpticalFlow(cap, impactFrameIndex);
    cv::Mat flow = flowFull(ROI);
    cv::Mat motionMask = computeMotionMask(flow, 0.2);
    cv::Mat colorMaskFull = computeColorMask(impactFrame);
    cv::Mat colorMask = colorMaskFull(ROI);
    cv::Mat combinedMask;
    cv::bitwise_and(motionMask, colorMask, combinedMask);

    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(combinedMask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    if (contours.empty())
        throw std::runtime_error("\n[getBoundingBoxCenter]: Ball region not found.");

    // Ignore contours much larger than ball
    const int MAX_BALL_WIDTH = 22;
    int bestIndex = 0;
    double bestArea = 0.0;

    for (int i = 0; i < contours.size(); i++) {
        double area = cv::contourArea(contours[i]);

        cv::Rect boxLocal = cv::boundingRect(contours[i]);
        if (boxLocal.width > MAX_BALL_WIDTH) continue;

        if (area > bestArea) {
            bestArea = area;
            bestIndex = i;
        }
    }

    cv::Rect box = cv::boundingRect(contours[bestIndex]);
    box.x += roiX;

    cv::Point2f center(
        box.x + box.width / 2.0,
        box.y + box.height / 2.0
    );

    cap.set(cv::CAP_PROP_POS_FRAMES, 0);

    return center;
}