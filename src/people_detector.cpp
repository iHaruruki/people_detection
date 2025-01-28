#include "people_detection/people_detector.hpp"

#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <fstream>
#include <sstream>
#include <ament_index_cpp/get_package_share_directory.hpp>

namespace people_detection
{

PeopleDetector::PeopleDetector()
: Node("people_detector"),
  input_width_(640),
  input_height_(640),
  scale_(1.0 / 255.0),
  mean_val_{0, 0, 0},
  swap_rb_(true),
  DISTANCE_THRESHOLD(2.0)
{
  RCLCPP_INFO(this->get_logger(), "Initializing PeopleDetector node.");

  // カラー画像の購読
  color_sub_ = this->create_subscription<sensor_msgs::msg::Image>(
    "/camera/color/image_raw",
    10,
    std::bind(&PeopleDetector::color_callback, this, std::placeholders::_1));

  // 深度画像の購読
  depth_sub_ = this->create_subscription<sensor_msgs::msg::Image>(
    "/camera/depth/image_raw",
    10,
    std::bind(&PeopleDetector::depth_callback, this, std::placeholders::_1));

  RCLCPP_INFO(this->get_logger(), "Subscriptions initialized.");

  // YOLOモデルの読み込み
  load_yolo_model();

  // クラス名の読み込み
  std::string classes_file = ament_index_cpp::get_package_share_directory("people_detection") + "/models/coco.names";
  RCLCPP_INFO(this->get_logger(), "Loading class names from: %s", classes_file.c_str());
  std::ifstream ifs(classes_file.c_str());
  if (!ifs.is_open()) {
    RCLCPP_ERROR(this->get_logger(), "Failed to open classes file: %s", classes_file.c_str());
    throw std::runtime_error("Failed to open classes file");
  }
  std::string line;
  while (getline(ifs, line)) {
    class_names_.push_back(line);
  }
  RCLCPP_INFO(this->get_logger(), "Loaded %zu class names.", class_names_.size());

  RCLCPP_INFO(this->get_logger(), "People Detector node has been started.");
}

void PeopleDetector::load_yolo_model()
{
  std::string model_path = ament_index_cpp::get_package_share_directory("people_detection") + "/models/yolov5s.onnx";
  //std::string model_path = "/home/peach/ros2_ws/install/people_detection/share/people_detection/models/yolov5s.onnx";
  RCLCPP_INFO(this->get_logger(), "Loading YOLOv5 model from: %s", model_path.c_str());

  try {
    net_ = cv::dnn::readNetFromONNX(model_path);
    RCLCPP_INFO(this->get_logger(), "YOLOv5 model loaded successfully.");
  } catch (const cv::Exception& e) {
    RCLCPP_ERROR(this->get_logger(), "Error loading YOLOv5 model: %s", e.what());
    throw;
  }

  // CPUを使用
  net_.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
  net_.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
}

bool PeopleDetector::is_person_detected_within_distance(const cv::Rect& box, double distance_threshold)
{
  if (!cv_ptr_depth_) {
    RCLCPP_WARN(this->get_logger(), "Depth image not received yet.");
    return false;
  }

  // バウンディングボックスの中心点の座標
  int center_x = box.x + box.width / 2;
  int center_y = box.y + box.height / 2;

  // 深度画像のピクセル位置から距離を取得（単位はメートル）
  // 深度画像が16UC1の場合、値はミリメートル単位
  if (cv_ptr_depth_->image.type() != CV_16UC1) {
    RCLCPP_WARN(this->get_logger(), "Depth image type is not CV_16UC1.");
    return false;
  }

  // 画像の範囲を超えないようにクランプ
  center_x = std::max(0, std::min(center_x, static_cast<int>(cv_ptr_depth_->image.cols - 1)));
  center_y = std::max(0, std::min(center_y, static_cast<int>(cv_ptr_depth_->image.rows - 1)));

  uint16_t depth_mm = cv_ptr_depth_->image.at<uint16_t>(center_y, center_x);
  if (depth_mm == 0) {
    // 無効な深度値
    return false;
  }

  double depth_m = depth_mm / 1000.0; // ミリメートルからメートルへ変換

  return depth_m <= distance_threshold;
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

  // 入力画像の前処理
  cv::Mat blob;
  cv::dnn::blobFromImage(image, blob, scale_, cv::Size(input_width_, input_height_), mean_val_, swap_rb_, false);
  net_.setInput(blob);

  // 推論実行
  std::vector<cv::Mat> outs;
  net_.forward(outs, net_.getUnconnectedOutLayersNames());

  // 結果の解析
  // YOLOv5のONNX出力形式に基づく解析
  // 出力は [N, 85] のテンソル（バウンディングボックス情報 + クラス確率）
  float conf_threshold = 0.5;
  float nms_threshold = 0.4;
  std::vector<int> class_ids;
  std::vector<float> confidences;
  std::vector<cv::Rect> boxes;

  cv::Mat detection = outs[0];
  for (int i = 0; i < detection.rows; i++) {
    float confidence = detection.at<float>(i, 4);
    if (confidence >= conf_threshold) {
      // クラス確率の最大値を取得
      cv::Mat scores = detection.row(i).colRange(5, detection.cols);
      cv::Point class_id_point;
      double max_class_score;
      cv::minMaxLoc(scores, 0, &max_class_score, 0, &class_id_point);
      if (max_class_score < conf_threshold)
        continue;

      int class_id = class_id_point.x;
      std::string class_name = class_names_[class_id];
      if (class_name != "person")
        continue;

      // バウンディングボックスの座標
      float x_center = detection.at<float>(i, 0) * image.cols;
      float y_center = detection.at<float>(i, 1) * image.rows;
      float width = detection.at<float>(i, 2) * image.cols;
      float height = detection.at<float>(i, 3) * image.rows;
      int left = static_cast<int>(x_center - width / 2);
      int top = static_cast<int>(y_center - height / 2);

      class_ids.push_back(class_id);
      confidences.push_back(static_cast<float>(confidence));
      boxes.emplace_back(left, top, static_cast<int>(width), static_cast<int>(height));
    }
  }

  // 非最大抑制
  std::vector<int> indices;
  cv::dnn::NMSBoxes(boxes, confidences, conf_threshold, nms_threshold, indices);

  bool person_detected_within_2m = false;

  for (size_t i = 0; i < indices.size(); ++i) {
    int idx = indices[i];
    cv::Rect box = boxes[idx];

    // 深度情報を用いて2m以内かを確認
    if (is_person_detected_within_distance(box, DISTANCE_THRESHOLD)) {
      person_detected_within_2m = true;
      // バウンディングボックスの描画
      cv::rectangle(image, box, cv::Scalar(0, 255, 0), 2);
      // ラベルの描画
      std::string label = cv::format("%.2f", confidences[idx]);
      label = class_names_[class_ids[idx]] + ":" + label;
      int baseLine;
      cv::Size label_size = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
      int top_label = std::max(box.y, label_size.height);
      cv::rectangle(image, cv::Point(box.x, top_label - label_size.height),
                    cv::Point(box.x + label_size.width, top_label + baseLine),
                    cv::Scalar(255, 255, 255), cv::FILLED);
      cv::putText(image, label, cv::Point(box.x, top_label),
                  cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0,0,0), 1);
    }
  }

  // 2m以内に人が検出された場合に「人検知」を表示
  if (person_detected_within_2m) {
    cv::putText(image, "人検知", cv::Point(30, 30), cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0,0,255), 2);
  }

  // 結果の表示
  cv::imshow("People Detection", image);
  cv::waitKey(1);
}

void PeopleDetector::depth_callback(const sensor_msgs::msg::Image::SharedPtr msg)
{
  // 深度画像をOpenCV形式に変換
  try {
    cv_ptr_depth_ = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::TYPE_16UC1);
  } catch (cv_bridge::Exception& e) {
    RCLCPP_ERROR(this->get_logger(), "cv_bridge exception: %s", e.what());
    return;
  }

  // 深度画像が正常に取得できたことをログに出力
  RCLCPP_DEBUG(this->get_logger(), "Depth image received.");
}

}  // namespace people_detection

int main(int argc, char* argv[])
{
  rclcpp::init(argc, argv);
  try {
    auto node = std::make_shared<people_detection::PeopleDetector>();
    rclcpp::spin(node);
  } catch (const std::exception& e) {
    RCLCPP_FATAL(rclcpp::get_logger("rclcpp"), "Exception: %s", e.what());
  }
  rclcpp::shutdown();
  return 0;
}
