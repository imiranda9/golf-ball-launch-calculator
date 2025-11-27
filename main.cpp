#include "video.h"
#include "ballTracking.h"

int main() {
    try {
        cv::ocl::setUseOpenCL(true);
        cv::VideoCapture vid("vid240.mov");

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
        std::vector<PosFrame> trackedPoints = getBallTrajectory(vid, impactIndex, boundingBox);
        rotatePointVector(trackedPoints, frameHeight);
        std::vector<PosFrame> trajectory = filterPoints(trackedPoints);
        std::cout << "DONE\n\n";

        float ballDiameter = (boundingBox.width + boundingBox.height) / 2.0;
        float carry = 0.0;
        float angle = 0.0;

        computeCarryAndAngle(trajectory, carry, angle, ballDiameter, fps);

        std::cout << "===================================\n";
        std::cout << "Carry Distance (m): " << carry << std::endl;
        std::cout << "Launch Angle (deg): " << angle << std::endl;
        std::cout << "===================================";
    }
    catch (const std::runtime_error& ex) {
        std::cerr << ex.what();
    }
    catch(const std::logic_error& ex2) {
        std::cerr << ex2.what();
    }

    return 0;
}