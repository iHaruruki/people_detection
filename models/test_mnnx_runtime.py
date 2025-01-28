import onnxruntime as ort

model_path = "/home/peach/ros2_ws/install/people_detection/share/people_detection/models/yolov5s.onnx"

try:
    session = ort.InferenceSession(model_path)
    print("モデルの読み込みに成功しました。")
except Exception as e:
    print(f"モデルの読み込みに失敗しました: {e}")

