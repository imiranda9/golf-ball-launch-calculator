#include "ballTracking.h"

std::vector<PosFrame> getBallTrajectory(cv::VideoCapture& cap, int startFrame) {
    const double MOTION_THRESHOLD = 0.15;

    cap.set(cv::CAP_PROP_POS_FRAMES, startFrame);

    int frameIndex = startFrame;
    std::vector<PosFrame> trajectory;

    while (true) {
        cv::Mat frame;
        if (!cap.read(frame) || frame.empty()) break;

        int savedIndex = (int)cap.get(cv::CAP_PROP_POS_FRAMES);

        cv::Point2f center;

        try {
            center = getBoundingBoxCenter(cap, savedIndex - 1, MOTION_THRESHOLD);
        }
        catch (...) {
            break;
        }

        trajectory.push_back({center, savedIndex - 1});

        /*************************************
         *  Optional tracking visualization
        *************************************/
        // cv::circle(frame, center, 5, cv::Scalar(240, 19, 255), -1);
        // cv::imshow("BallTracking", frame);
        // if (cv::waitKey(1) == 27) break;

        cap.set(cv::CAP_PROP_POS_FRAMES, savedIndex);
    }

    cap.set(cv::CAP_PROP_POS_FRAMES, 0);
    return trajectory;
}

PosFrame getRotatedPoint(const PosFrame& point, int height) {
    float newX = height - point.pos.y;
    float newY = point.pos.x;

    PosFrame out;
    out.pos = cv::Point2f(newX, newY);
    out.frameIndex = point.frameIndex;

    return out;
}

void rotatePointVector(std::vector<PosFrame>& points, int height) {
    for (auto& p : points)
        p = getRotatedPoint(p, height);
}

std::vector<PosFrame> filterPoints(const std::vector<PosFrame>& points) {
    std::vector<PosFrame> out;
    
    out.push_back(points[0]);
    float lastX = points[0].pos.x;
    float lastY = points[0].pos.y;

    for (size_t i = 1; i < points.size(); i++) {
        bool goodX = (points[i].pos.x >= lastX); // Increasing X = movement right
        bool goodY = (points[i].pos.y <= lastY); // Decreasing Y = movement up

        if (goodX && goodY) {
            out.push_back(points[i]);
            lastX = points[i].pos.x;
            lastY = points[i].pos.y;
        }
    }

    return out;
}

void computeCarryAndAngle(const std::vector<PosFrame>& points, float& carry, float& angle, float ballDiameterPx, float fps) {
    if (points.size() < 2)
        throw std::runtime_error("[computeCarryAndAngle]: Minumum of 2 valid points required.");

    const float BALL_DIAMETER_M = 0.04267;

    float metersPerPx = BALL_DIAMETER_M / ballDiameterPx;
    float sumVx = 0.0;
    float sumVy = 0.0;
    int count = 0;

    for (size_t i = 1; i < points.size(); i++) {
        const PosFrame& p1 = points[i - 1];
        const PosFrame& p2 = points[i];

        int df = p2.frameIndex - p1.frameIndex;
        // Check for redundant or out of order frames
        if (df <= 0) continue;

        float dt = df / fps;

        float dx = (p2.pos.x - p1.pos.x) * metersPerPx;
        float dy = (p2.pos.y - p1.pos.y) * metersPerPx;

        float vx = dx / dt;
        float vy = dy / dt;

        sumVx += vx;
        sumVy += vy;
        count++;
    }

    float vx = sumVx / count;
    float vy = -(sumVy / count);

    float flightTime = (2.0 * vy) / 9.81;

    carry = vx * flightTime;
    angle = std::atan2(vy, vx) * 180.0 / CV_PI;
}