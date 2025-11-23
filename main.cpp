#include "video.h"
#include "ballTracking.h"

int main() {
    try {
        cv::ocl::setUseOpenCL(true);
        cv::VideoCapture vid240("white_240.mov");
        cv::VideoCapture vid162("yellow_162.mov");

        if (!vid240.isOpened() || !vid162.isOpened()) {
            std::cerr << "\nError: Cannot open video file.\n";
            return -1;
        }

        /********************************
         *  vid240 impact frame = 497
         *  vid162 impact frame = 149
        ********************************/
        int index = findImpactFrameIndex(vid240, computeFPS(vid240), 0.2);
        displayFrame(vid240, index);
    }
    catch (const std::runtime_error& ex) {
        std::cerr << ex.what();
    }
    catch(const std::logic_error& ex2) {
        std::cerr << ex2.what();
    }

    return 0;
}