#ifndef PEOPLE_DETECTION__PEOPLE_DETECTOR_HPP_
#define PEOPLE_DETECTION__PEOPLE_DETECTOR_HPP_

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>
#include <vector>
#include <string>

namespace people_detection
{

class PeopleDetector : public rclcpp::Node
{
public:
  PeopleDetector();

private:
  void color_callback(const sensor_msgs::msg::Image::SharedPtr msg);
  void depth_callback(const sensor_msgs::msg::Image::SharedPtr msg);
  void ir_callback(const sensor_msgs::msg::Image::SharedPtr msg);
  void load_yolo_model();
  bool is_person_detected_within_distance(const cv::Rect& box, double distance_threshold);

  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr color_sub_;
  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr depth_sub_;
  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr ir_sub_;

  cv_bridge::CvImagePtr cv_ptr_color_;
  cv_bridge::CvImagePtr cv_ptr_depth_;

  cv::dnn::Net net_;
  std::vector<std::string> class_names_;
  int input_width_;
  int input_height_;
  double scale_;
  cv::Scalar mean_val_;
  bool swap_rb_;

  const double DISTANCE_THRESHOLD;
};

}  // namespace people_detection

#endif  // PEOPLE_DETECTION__PEOPLE_DETECTOR_HPP_
