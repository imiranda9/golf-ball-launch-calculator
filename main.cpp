#include "video.h"
#include "ballTracking.h"

int main() {
    try {
        cv::ocl::setUseOpenCL(true);
        cv::VideoCapture vid("white_240.mov");
        // cv::VideoCapture vid162("yellow_162.mov");

        if (!vid.isOpened())
            throw std::runtime_error("[main]: Failed to open video.");

        // Rotate vid
        cv::Mat f;

        while (true) {
            if (!vid.read(f) || f.empty()) break;
            cv::rotate(f, f, cv::ROTATE_90_CLOCKWISE);
        }
        vid.set(cv::CAP_PROP_POS_FRAMES, 0);

        /********************************
         *  vid240 impact frame = 498
         *  vid162 impact frame = 149
        ********************************/
        // int index = findImpactFrameIndex(vid);
        // cv::Rect boundingBox = getBoundingBox(vid, index);

        std::cout << "\nBox Test Start\n";
        cv::Rect boundingBox = getBoundingBox(vid, 498);
        std::cout << "\nBox Test End\n";

        vid.set(cv::CAP_PROP_POS_FRAMES, 498);
        cv::Mat impactFrame;
        vid.read(impactFrame);
        cv::rectangle(impactFrame, boundingBox, cv::Scalar(0, 0, 255), 2);
        cv::rotate(impactFrame, impactFrame, cv::ROTATE_90_CLOCKWISE);
        cv::resize(impactFrame, impactFrame, cv::Size(), 0.4, 0.4);
        cv::imshow("box?", impactFrame);
        cv::waitKey(0);

        std::vector<cv::Point2f> test = trackBallTrajectory(vid, 498, boundingBox);
    }
    catch (const std::runtime_error& ex) {
        std::cerr << ex.what();
    }
    catch(const std::logic_error& ex2) {
        std::cerr << ex2.what();
    }

    return 0;
}