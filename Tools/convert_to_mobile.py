# tools/convert_to_mobile.py
# Script này chuyển đổi các model YOLOv8 và Keras sang định dạng TFLite để sử dụng trên mobile app.
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import tensorflow as tf
from ultralytics import YOLO

#========================toàn bộ đường dẫn gốc cho dễ kiểm tra nếu lỗi =========================
# Đường dẫn model gốc (sau khi train)
YOLO_PT_PATH      = "models/hand_det_yolo.pt"
FEATURE_H5_PATH   = "models/feature_extractor.h5"
GRU_H5_PATH       = "models/action_recognizer.h5"

# Đường dẫn output cho mobile
OUTPUT_DIR        = "mobile_app/assets"
YOLO_TFLITE       = os.path.join(OUTPUT_DIR, "hand_det_yolo.tflite")
FEATURE_TFLITE    = os.path.join(OUTPUT_DIR, "feature_extractor.tflite")
GRU_TFLITE        = os.path.join(OUTPUT_DIR, "action_recognizer.tflite")



def convert_yolo_to_tflite(pt_path, output_path): #Chuyển model nặng → nhẹ cho mobile
    """Chuyển YOLOv8 PyTorch -> TFLite (float32 hoặc int8), YOLO dùng thư viện Ultralytics để tự export"""
    print(f"Đang convert YOLO: {pt_path}")

    if not os.path.exists(pt_path):
        print(f"❌ Không tìm thấy file: {pt_path}")
        return False

    try:
        model = YOLO(pt_path)
        model.export(format="tflite", imgsz=320, int8=False)  # Xuất file .tflite
        # File sẽ được lưu cùng thư mục với tên tương tự .tflite
        print(f"YOLO converted to TFLite: {output_path}")
    except Exception as e:
        print(f"❌ Lỗi khi convert YOLO: {e}")
        return False

def convert_keras_to_tflite(h5_path, output_path, quantize=False):
    """Chuyển model Keras (.h5) sang TFLite"""
    model = tf.keras.models.load_model(h5_path)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    if quantize:
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()
    with open(output_path, 'wb') as f:
        f.write(tflite_model)
    print(f"Keras model converted to TFLite: {output_path}")

if __name__ == "__main__":
    import config
    
    # Đường dẫn model gốc
    yolo_pt = os.path.join(config.SHARED_ASSETS_DIR, "hand_det_yolo.pt")
    feature_h5 = os.path.join(config.SHARED_ASSETS_DIR, "feature_extractor.h5") # EfficientNet nhận diện đặc trưng tay
    gru_h5 = os.path.join(config.SHARED_ASSETS_DIR, "action_recognizer.h5")     # GRU nhận diện chuỗi ký hiệu

    # Tạo thư mục assets nếu chưa có
    os.makedirs("mobile_app/assets", exist_ok=True)

    # Chuyển đổi
    convert_yolo_to_tflite(yolo_pt, "mobile_app/assets/hand_det_yolo.tflite")
    convert_keras_to_tflite(feature_h5, "mobile_app/assets/feature_extractor.tflite")
    convert_keras_to_tflite(gru_h5, "mobile_app/assets/action_recognizer.tflite") 