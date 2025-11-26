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

    throw std::logic_error("\n[findImpactFrameIndex]: Failed to find impact frame.");
}

cv::Mat computeOpticalFlow(cv::VideoCapture& cap, int startFrame) {
    cv::Mat frame1, frame2, gray1, gray2, flow;

    cap.set(cv::CAP_PROP_POS_FRAMES, startFrame);

    if (!cap.read(frame1) || !cap.read(frame2))
        throw std::runtime_error("\n[computeOpticalFlow]: Failed to read two frames.");

    cv::cvtColor(frame1, gray1, cv::COLOR_BGR2GRAY);
    cv::cvtColor(frame2, gray2, cv::COLOR_BGR2GRAY);

    cv::calcOpticalFlowFarneback(
        gray1, gray2, flow,
        OF_PYR_SCALE, OF_LEVELS, OF_WINSIZE, OF_ITERATIONS, OF_POLY_N, OF_POLY_SIGMA, OF_FLAGS
    );

    cap.set(cv::CAP_PROP_POS_FRAMES, 0);
    return flow;
}

cv::Mat computeOpticalFlow(const cv::Mat& frame1, const cv::Mat& frame2) {
    cv::Mat gray1, gray2;
    cv::cvtColor(frame1, gray1, cv::COLOR_BGR2GRAY);
    cv::cvtColor(frame2, gray2, cv::COLOR_BGR2GRAY);

    cv::Mat flow;
    cv::calcOpticalFlowFarneback(
        gray1, gray2, flow,
        OF_PYR_SCALE, OF_LEVELS, OF_WINSIZE, OF_ITERATIONS, OF_POLY_N, OF_POLY_SIGMA, OF_FLAGS
    );

    return flow;
}

cv::Mat computeMotionMask(const cv::Mat& flow, double motionThreshold) {
    // Split motion into two images
    cv::Mat xy[2];
    cv::split(flow, xy);

    // Calculate magnitude of pixel motion vectors
    cv::Mat mag;
    cv::magnitude(xy[0], xy[1], mag);

    // Find maximum motion vector
    double maxVal;
    cv::minMaxLoc(mag, nullptr, &maxVal);
    // Normalize magnitudes
    cv::Mat motionNorm = mag / maxVal;

    cv::Mat mask;
    cv::threshold(motionNorm, mask, motionThreshold, 1.0, cv::THRESH_BINARY);
    mask.convertTo(mask, CV_8U);

    return mask;
}

cv::Mat computeColorMask(const cv::Mat& frame) {
    cv::Mat hsv;
    cv::cvtColor(frame, hsv, cv::COLOR_BGR2HSV);

    // Two most common golf ball colors
    cv::Mat whiteMask, yellowMask;

    // White Ball: Low saturation, high brightness
    cv::inRange(hsv,
                cv::Scalar(0, 20, 140),
                cv::Scalar(180, 120, 255),
                whiteMask);

    // Yellow Ball: med-high saturation, high brightness
    cv::inRange(hsv,
                cv::Scalar(15, 80, 150),
                cv::Scalar(40, 255, 255),
                yellowMask);

    cv::Mat colorMask;
    cv::bitwise_or(whiteMask, yellowMask, colorMask);

    // Normalize from 0-255 to 0-1
    return colorMask / 255;
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

cv::Point2f getBoundingBoxPoint(cv::VideoCapture& cap, int impactFrameIndex, double motionThreshold) {
    cap.set(cv::CAP_PROP_POS_FRAMES, impactFrameIndex);

    cv::Mat impactFrame;
    if (!cap.read(impactFrame))
        throw std::runtime_error("\n[getBoundingBox]: Failed to read frame.");

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
        throw std::runtime_error("\n[getBoundingBox]: Ball region not found.");

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
        box.x + box.width * 0.5f,
        box.y + box.height * 0.5f
    );

    cap.set(cv::CAP_PROP_POS_FRAMES, 0);

    return center;
}

std::vector<cv::Point2f> trackBallTrajectory(cv::VideoCapture& cap, int startFrame, cv::Rect initialBox) {
    const double MOTION_THRESHOLD = 0.15;

    cap.set(cv::CAP_PROP_POS_FRAMES, startFrame);

    int frameIndex = startFrame;
    std::vector<cv::Point2f> trajectory;

    while (true) {
        cv::Mat frame;
        if (!cap.read(frame) || frame.empty()) break;

        int savedIndex = (int)cap.get(cv::CAP_PROP_POS_FRAMES);

        cv::Point2f center;

        try {
            center = getBoundingBoxPoint(cap, savedIndex - 1, MOTION_THRESHOLD);
        }
        catch (...) {
            break;
        }

        trajectory.push_back(center);

        cv::circle(frame, center, 5, cv::Scalar(240, 19, 255), -1);
        cv::imshow("BallTracking", frame);
        if (cv::waitKey(1) == 27) break;

        cap.set(cv::CAP_PROP_POS_FRAMES, savedIndex);

        frameIndex;
    }

    return trajectory;
}