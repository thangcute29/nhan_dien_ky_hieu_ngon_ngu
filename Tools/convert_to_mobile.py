# Tools/convert_to_mobile.py
import os
import sys

# Bật Legacy Keras để an toàn đọc file h5
os.environ['TF_USE_LEGACY_KERAS'] = '1'
import tensorflow as tf

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Nguồn file h5
CNN_H5 = os.path.join(BASE_DIR, "Cloud_server", "Trainer", "runs", "cnn_checkpoints", "best_cnn.h5")
GRU_H5 = os.path.join(BASE_DIR, "Shared_lib", "Assets", "action_recognizer.h5")

# Đích đến cho Mobile App
OUTPUT_DIR = os.path.join(BASE_DIR, "Mobile_app", "assets")
os.makedirs(OUTPUT_DIR, exist_ok=True)
CNN_TFLITE = os.path.join(OUTPUT_DIR, "feature_extractor.tflite")
GRU_TFLITE = os.path.join(OUTPUT_DIR, "action_recognizer.tflite")

def convert_h5_to_tflite(h5_path, output_path, use_select_tf_ops=False):
    if not os.path.exists(h5_path):
        print(f"[ERROR] Khong tim thay file goc: {h5_path}")
        return False
        
    print(f"[INFO] Dang nen model: {h5_path}...")
    try:
        try:
            model = tf.keras.models.load_model(h5_path, compile=False)
        except Exception:
            import tf_keras
            model = tf_keras.models.load_model(h5_path, compile=False)

        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        if use_select_tf_ops:
            converter.target_spec.supported_ops = [
                tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS
            ]
            converter._experimental_lower_tensor_list_ops = False
            print("[WARNING] GRU uses Select TF Ops; the app must bundle the Flex delegate.")
        tflite_model = converter.convert()
        temporary_path = output_path + ".tmp"
        with open(temporary_path, 'wb') as f:
            f.write(tflite_model)
        os.replace(temporary_path, output_path)
        print(f"[SUCCESS] Da nen thanh cong ra file TFLite: {output_path}")
        return True
    except Exception as exc:
        print(f"[ERROR] Chuyen doi that bai: {exc}")
        return False

if __name__ == "__main__":
    import io
    if hasattr(sys.stdout, 'buffer'):
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    print("[RUNNING] BAT DAU EP KIEU TFLITE CHO MOBILE (CNN & GRU)")
    
    cnn_ok = convert_h5_to_tflite(CNN_H5, CNN_TFLITE, use_select_tf_ops=False)
    gru_ok = convert_h5_to_tflite(GRU_H5, GRU_TFLITE, use_select_tf_ops=True)
    if not (cnn_ok and gru_ok):
        print("\n[FAILED] Khong cap nhat day du model Mobile.")
        raise SystemExit(1)
    print("\n[SUCCESS] XONG! App Mobile da nhan du tflite!")
