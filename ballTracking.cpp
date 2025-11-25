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
    bool rotate = isLandscape(cap);
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

std::vector<cv::Point2f> trackBallTrajectory(cv::VideoCapture& cap, int startFrame, cv::Rect initialBox) {
    cap.set(cv::CAP_PROP_POS_FRAMES, startFrame);
    cv::Mat frame;

    // Check capture and reset
    if (!cap.read(frame) || frame.empty())
        throw std::runtime_error("\n[trackBallTrajectory]: Failed to read frame.");
    cap.set(cv::CAP_PROP_POS_FRAMES, startFrame);
    
    cv::Ptr<cv::TrackerCSRT> tracker = cv::TrackerCSRT::create();

    // Expand box around the ball
    int pad = 30;   // try 12–20 pixels
    initialBox.x      = std::max(0, initialBox.x - pad);
    initialBox.y      = std::max(0, initialBox.y - pad);
    initialBox.width  = std::min(frame.cols - initialBox.x, initialBox.width  + pad * 2);
    initialBox.height = std::min(frame.rows - initialBox.y, initialBox.height + pad * 2);

    tracker->init(frame, initialBox);

    std::vector<cv::Point2f> trajectory;

    cv::Rect prevBox = initialBox;

    // Find center of bounding box
    trajectory.emplace_back(
        initialBox.x + initialBox.width / 2,
        initialBox.y + initialBox.height / 2
    );

    while (true) {
        if (!cap.read(frame) || frame.empty()) break;

        cv::Rect box;
        bool tracking = tracker->update(frame, box);
        if (!tracking) break;

        cv::Mat colorMaskFull = computeColorMask(frame);
        cv::Rect safeBox = box & cv::Rect(0, 0, frame.cols, frame.rows);
        if (safeBox.area() <= 0) break;
        cv::Mat maskROI = colorMaskFull(safeBox);

        double ballInROI = cv::mean(maskROI)[0];

        if (ballInROI < 0.3)
            box = prevBox;
        else
            prevBox = box;

        bool outOfFrame = (
            box.x < 0 || box.y < 0 ||
            box.x + box.width >= frame.cols ||
            box.y + box.height >= frame.rows
        );
        if (outOfFrame) break;

        // Find center
        cv::Point center(
            box.x+ box.width / 2,
            box.y + box.height / 2
        );

        cv::rectangle(frame, box, cv::Scalar(0, 0, 255), 2);
        cv::imshow("Tracking", frame);
        if (cv::waitKey(1) == 27) break;

        trajectory.emplace_back(center.x, center.y);
    }

    return trajectory;
}