#include "video.h"

double computeFPS(cv::VideoCapture& cap) {
    double metaFPS = cap.get(cv::CAP_PROP_FPS);

    std::cout << "FPS (metadata): " << metaFPS << std::endl;
    return metaFPS;
}