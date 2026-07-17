# cloud_server/trainer/retrain.py
import os
import shutil
import subprocess
from cloud_server.model_registry.registry import ModelRegistry

LABELED_DIR = "cloud_server/database/edge_cases/labeled"
SEQUENCES_TRAIN_DIR = "data/sequences/train"

def merge_labeled_data():
    """Di chuyển video đã gán nhãn vào dataset sequences."""
    for label in os.listdir(LABELED_DIR):
        src_label_dir = os.path.join(LABELED_DIR, label)
        dst_label_dir = os.path.join(SEQUENCES_TRAIN_DIR, label)
        os.makedirs(dst_label_dir, exist_ok=True)
        for video_file in os.listdir(src_label_dir):
            if video_file.endswith(('.mp4', '.avi', '.mov')):
                src = os.path.join(src_label_dir, video_file)
                dst = os.path.join(dst_label_dir, video_file)
                shutil.move(src, dst)
        # Sau khi move, xóa thư mục rỗng
        if not os.listdir(src_label_dir):
            os.rmdir(src_label_dir)

def run_retraining():
    print("Starting retraining process...")
    # Bước 1: Merge dữ liệu đã gán nhãn vào dataset chính
    merge_labeled_data()

    # Bước 2: Chạy script huấn luyện lại (có thể dùng subprocess gọi train_feature_extractor.py và train_gru.py)
    # Lưu ý: Bạn cần đảm bảo các script huấn luyện nằm trong cloud_server/trainer/train_scripts/
    subprocess.run(["python", "cloud_server/trainer/train_scripts/train_feature_extractor.py"], check=True)
    subprocess.run(["python", "cloud_server/trainer/train_scripts/train_gru.py"], check=True)

    # Bước 3: Đăng ký phiên bản mới
    new_version = increment_version()
    registry = ModelRegistry()
    registry.register_new_version(
        new_version,
        "models/feature_extractor.h5",
        "models/action_recognizer.h5",
        "models/hand_det_yolo.pt"  # YOLO thường không cần retrain thường xuyên
    )

    # Bước 4: Chuyển đổi sang TFLite và cập nhật mobile assets (tùy chọn)
    subprocess.run(["python", "tools/convert_to_mobile.py"], check=True)

    print(f"Retraining completed. New version: {new_version}")

def increment_version():
    registry = ModelRegistry()
    latest = registry.get_latest_version()
    major, minor, patch = map(int, latest.split('.'))
    patch += 1
    return f"{major}.{minor}.{patch}"

if __name__ == "__main__":
    run_retraining()