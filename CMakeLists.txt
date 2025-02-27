cmake_minimum_required(VERSION 3.5)
project(people_detection)

# C++標準
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 必要なパッケージの検出
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(image_transport REQUIRED)
find_package(OpenCV REQUIRED)
find_package(ament_index_cpp REQUIRED)  # 追加

# インクルードディレクトリ
include_directories(
  include
  ${OpenCV_INCLUDE_DIRS}
)

# 実行ファイルの追加
add_executable(people_detector src/people_detector.cpp)
ament_target_dependencies(people_detector rclcpp sensor_msgs cv_bridge OpenCV)

# リンクライブラリ
ament_target_dependencies(people_detector
  rclcpp
  sensor_msgs
  cv_bridge
  image_transport
  OpenCV
  ament_index_cpp  # 追加
)

install(TARGETS
  people_detector
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY models/
        DESTINATION share/${PROJECT_NAME}/models
        FILES_MATCHING PATTERN "*.onnx" PATTERN "*.names"
)

# パッケージのビルド
ament_package()
