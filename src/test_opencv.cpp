#include <opencv2/opencv.hpp>
#include <iostream>

int main() {
    std::cout << "OpenCV Version: " << CV_VERSION << std::endl;
    cv::dnn::Net net;
    try {
        net = cv::dnn::readNetFromONNX("/home/peach/ros2_ws/install/people_detection/share/people_detection/models/yolov5s.onnx");
        std::cout << "Model loaded successfully." << std::endl;
    } catch (const cv::Exception& e) {
        std::cerr << "Error loading model: " << e.what() << std::endl;
    }
    return 0;
}
