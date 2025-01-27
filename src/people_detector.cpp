#include "people_detection/people_detector.hpp"

namespace people_detection
{

PeopleDetector::PeopleDetector()
: Node("people_detector")
{
  // カラー画像の購読
  color_sub_ = this->create_subscription<sensor_msgs::msg::Image>(
    "/camera/color/image_raw",
    10,
    std::bind(&PeopleDetector::color_callback, this, std::placeholders::_1));

  // 深度画像の購読（必要に応じて使用）
  depth_sub_ = this->create_subscription<sensor_msgs::msg::Image>(
    "/camera/depth/image_raw",
    10,
    std::bind(&PeopleDetector::depth_callback, this, std::placeholders::_1));

  // 人検知用のHOGディスクリプタの設定
  hog_.setSVMDetector(cv::HOGDescriptor::getDefaultPeopleDetector());

  RCLCPP_INFO(this->get_logger(), "People Detector node has been started.");
}

void PeopleDetector::color_callback(const sensor_msgs::msg::Image::SharedPtr msg)
{
  // カラー画像をOpenCV形式に変換
  try {
    cv_ptr_color_ = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
  } catch (cv_bridge::Exception& e) {
    RCLCPP_ERROR(this->get_logger(), "cv_bridge exception: %s", e.what());
    return;
  }

  cv::Mat image = cv_ptr_color_->image;

  // 人検知
  std::vector<cv::Rect> detections;
  std::vector<double> weights;
  hog_.detectMultiScale(image, detections, weights, 0, cv::Size(8,8), cv::Size(32,32), 1.05, 2, false);

  // 検知結果の描画
  for (size_t i = 0; i < detections.size(); ++i) {
    cv::rectangle(image, detections[i], cv::Scalar(0, 255, 0), 2);
  }

  // 結果の表示
  cv::imshow("People Detection", image);
  cv::waitKey(1);
}

void PeopleDetector::depth_callback(const sensor_msgs::msg::Image::SharedPtr msg)
{
  // 深度画像の処理が必要な場合はここに実装
  // 例えば、検出した人の距離を計測するなど
}

}  // namespace people_detection

int main(int argc, char* argv[])
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<people_detection::PeopleDetector>();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}
