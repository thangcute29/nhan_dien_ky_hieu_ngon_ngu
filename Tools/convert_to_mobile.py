# tools/convert_to_mobile.py
# Script này chuyển đổi các model YOLOv8 và Keras sang định dạng TFLite để sử dụng trên mobile app.
import os
import sys

# Thiết lập TF_USE_LEGACY_KERAS=1 trước khi import tensorflow để tương thích tốt với model Keras 2 (.h5) chứa TFOpLambda
os.environ['TF_USE_LEGACY_KERAS'] = '1'
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
        exported_path = None
        try:
            exported_path = model.export(format="tflite", imgsz=320, int8=False)  # Xuất file .tflite
        except Exception as export_err:
            print(f"⚠️ Export TFLite trực tiếp gặp lỗi ({export_err}). Đang thử convert qua SavedModel...")
            saved_model_path = model.export(format="saved_model", imgsz=320)
            converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_path)
            tflite_model = converter.convert()
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, 'wb') as f:
                f.write(tflite_model)
            print(f"✅ YOLO converted & saved to: {output_path}")
            return True

        if exported_path and os.path.exists(exported_path):
            import shutil
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            shutil.copy(exported_path, output_path)
            print(f"✅ YOLO converted & saved to: {output_path}")
            return True
        else:
            print(f"⚠️ YOLO export xong nhưng không tìm thấy file tại: {exported_path}")
            return False
    except Exception as e:
        print(f"❌ Lỗi khi convert YOLO: {e}")
        return False

def convert_keras_to_tflite(h5_path, output_path, quantize=False):
    """Chuyển model Keras (.h5) sang TFLite"""
    if not os.path.exists(h5_path):
        print(f"❌ Không tìm thấy file: {h5_path}")
        return False

    try:
        model = tf.keras.models.load_model(h5_path)
    except Exception as e:
        print(f"⚠️ tf.keras load_model gặp lỗi ({e}), đang thử dùng tf_keras...")
        import tf_keras
        model = tf_keras.models.load_model(h5_path)

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    if quantize:
        converter.optimizations = [tf.lite.Optimize.DEFAULT]

    try:
        tflite_model = converter.convert()
    except Exception as e:
        print(f"⚠️ Convert mặc định gặp lỗi: {e}\nĐang thử convert với SELECT_TF_OPS (hỗ trợ RNN/GRU)...")
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS,
            tf.lite.OpsSet.SELECT_TF_OPS
        ]
        converter._experimental_lower_tensor_list_ops = False
        tflite_model = converter.convert()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'wb') as f:
        f.write(tflite_model)
    print(f"✅ Keras model converted to TFLite: {output_path}")
    return True

def verify_hand_model(tflite_path):
    """Kiểm tra an toàn: đảm bảo model tay vừa convert đúng là model 1-class 'hand'
    (output shape [1, 5, N]), KHÔNG PHẢI model COCO 80-class bị fallback nhầm
    (output shape [1, 84/85, N])."""
    if not os.path.exists(tflite_path):
        print(f"⚠️ Không tìm thấy file để kiểm tra: {tflite_path}")
        return False
    try:
        interpreter = tf.lite.Interpreter(model_path=tflite_path)
        interpreter.allocate_tensors()
        out_shape = interpreter.get_output_details()[0]['shape']
        if len(out_shape) < 2 or out_shape[1] != 5:
            print(f"❌ CẢNH BÁO NGHIÊM TRỌNG: {tflite_path} có output shape {out_shape}, "
                  f"KHÔNG PHẢI model 1-class 'hand' (kỳ vọng [1, 5, N]). "
                  f"Rất có thể đang dùng nhầm model YOLOv8n COCO gốc (fallback)! "
                  f"Kiểm tra lại đường dẫn best.pt và quá trình train YOLO.")
            return False
        print(f"✅ Xác nhận đúng model tay 1-class: output shape {out_shape}")
        return True
    except Exception as e:
        print(f"⚠️ Lỗi khi kiểm tra model: {e}")
        return False


if __name__ == "__main__":
    import config
    import shutil
    
    # Đường dẫn model gốc
    yolo_pt = os.path.join(config.SHARED_ASSETS_DIR, "hand_det_yolo.pt")
    feature_h5 = os.path.join(config.SHARED_ASSETS_DIR, "feature_extractor.h5") # EfficientNet nhận diện đặc trưng tay
    gru_h5 = os.path.join(config.SHARED_ASSETS_DIR, "action_recognizer.h5")     # GRU nhận diện chuỗi ký hiệu

    # Nếu chưa có hand_det_yolo.pt ở Shared_lib/assets, tự tìm file best.pt đã train hoặc yolov8n.pt
    if not os.path.exists(yolo_pt):
        # Đường dẫn TUYỆT ĐỐI, khớp với project= trong train_yolo.py (đã sửa để
        # không còn phụ thuộc cwd). Không còn tiền tố "runs/detect" thừa.
        possible_pt = os.path.join(config.PROJECT_ROOT, "Cloud_server", "Trainer", "runs", "yolo_hands", "weights", "best.pt")
        if os.path.exists(possible_pt):
            os.makedirs(os.path.dirname(yolo_pt), exist_ok=True)
            shutil.copy(possible_pt, yolo_pt)
            print(f"📋 Đã tự động copy model YOLO trained từ {possible_pt} sang {yolo_pt}")
        else:
            fallback_pt = os.path.join(config.PROJECT_ROOT, "yolov8n.pt")
            if os.path.exists(fallback_pt):
                os.makedirs(os.path.dirname(yolo_pt), exist_ok=True)
                shutil.copy(fallback_pt, yolo_pt)
                print(f"📋 Đã tự động copy model YOLOv8 mặc định từ {fallback_pt} sang {yolo_pt}")

    # Tạo thư mục assets nếu chưa có
    os.makedirs("mobile_app/assets", exist_ok=True)

    # Chuyển đổi
    convert_yolo_to_tflite(yolo_pt, "mobile_app/assets/hand_det_yolo.tflite")
    convert_keras_to_tflite(feature_h5, "mobile_app/assets/feature_extractor.tflite")
    convert_keras_to_tflite(gru_h5, "mobile_app/assets/action_recognizer.tflite")

    # Kiểm tra an toàn ngay sau khi convert xong, tránh lặp lại bug fallback COCO
    verify_hand_model("mobile_app/assets/hand_det_yolo.tflite")
 