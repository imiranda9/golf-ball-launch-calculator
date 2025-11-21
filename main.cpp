#include "video.h"

int main() {
    cv::VideoCapture cap("testvid.mp4");

    if (!cap.isOpened()) {
        std::cerr << "\nError: Cannot open video file.\n";
        return -1;
    }

    double test = computeFPS(cap);

    return 0;
}