#include "video.h"

int main() {
    try {
        cv::VideoCapture vid240("white_240.mov");
        cv::VideoCapture vid162("yellow_162.mov");

        if (!vid240.isOpened() || !vid162.isOpened()) {
            std::cerr << "\nError: Cannot open video file.\n";
            return -1;
        }

        // std::cout << "White Ball Vid:\n";
        // double rangeBallFPS = computeFPS(vid240);
        // std::cout << std::endl;

        // std::cout << "Yellow Ball Vid:\n";
        // double foamBallFPS = computeFPS(vid162);
        // std::cout << std::endl;

        // playPortraitVideo(vid240);

        // int index = findImpactFrameIndex(vid240, 30);

        // if (index > 0) {
        //     vid240.set(cv::CAP_PROP_POS_FRAMES, index);
        //     cv::Mat frame;
        //     vid240.read(frame);
        //     cv::imshow("Impact Frame", frame);
        //     cv::waitKey(0);
        // }

    }
    catch (const std::runtime_error& ex) {
        std::cerr << ex.what();
    }

    return 0;
}