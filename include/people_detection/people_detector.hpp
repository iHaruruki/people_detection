#ifndef PEOPLE_DETECTION__PEOPLE_DETECTOR_HPP_
#define PEOPLE_DETECTION__PEOPLE_DETECTOR_HPP_

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>

namespace people_detection
{

class PeopleDetector : public rclcpp::Node
{
public:
  PeopleDetector();

private:
  void color_callback(const sensor_msgs::msg::Image::SharedPtr msg);
  void depth_callback(const sensor_msgs::msg::Image::SharedPtr msg);

  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr color_sub_;
  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr depth_sub_;

  cv_bridge::CvImagePtr cv_ptr_color_;
  cv_bridge::CvImagePtr cv_ptr_depth_;

  cv::HOGDescriptor hog_;
};

}  // namespace people_detection

#endif  // PEOPLE_DETECTION__PEOPLE_DETECTOR_HPP_
