#include "video.h"
#include "ballTracking.h"

int main() {
    try {
        cv::ocl::setUseOpenCL(true);
        cv::VideoCapture vid("white_240.mov");
        // cv::VideoCapture vid162("yellow_162.mov");

        if (!vid.isOpened()) {
            std::cerr << "\nError: Cannot open video file.\n";
            return -1;
        }

        /********************************
         *  vid240 impact frame = 498
         *  vid162 impact frame = 149
        ********************************/
        cv::Rect boundingBox = getBoundingBox(vid, 498);
        // int index = findImpactFrameIndex(vid);
        // cv::Rect boundingBox = getBoundingBox(vid, index);

        vid.set(cv::CAP_PROP_POS_FRAMES, 498);
        cv::Mat impactFrame;
        vid.read(impactFrame);
        cv::rectangle(impactFrame, boundingBox, cv::Scalar(0, 0, 255), 2);

        cv::rotate(impactFrame, impactFrame, cv::ROTATE_90_CLOCKWISE);
        cv::resize(impactFrame, impactFrame, cv::Size(), 0.4, 0.4);
        cv::imshow("box?", impactFrame);
        cv::waitKey(0);

    }
    catch (const std::runtime_error& ex) {
        std::cerr << ex.what();
    }
    catch(const std::logic_error& ex2) {
        std::cerr << ex2.what();
    }

    return 0;
}