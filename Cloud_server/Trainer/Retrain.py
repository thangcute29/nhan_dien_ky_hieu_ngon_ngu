# Cloud_server/Trainer/Retrain.py
import os
import sys
import shutil
import json
import subprocess
from datetime import datetime

# Thiết lập đường dẫn gốc BASE_DIR
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import config
from Cloud_server.Model_registry.registry import ModelRegistry

# Các đường dẫn dữ liệu
LABELED_DIR = config.LABELED_DIR if hasattr(config, 'LABELED_DIR') else os.path.join(BASE_DIR, "Cloud_server", "Database", "edge_cases", "Labeled")
SEQUENCES_TRAIN_DIR = os.path.join(config.RESEARCH_DATA_DIR, "Sequences", "processed", "train") if hasattr(config, 'RESEARCH_DATA_DIR') else os.path.join(BASE_DIR, "Research_and_Data", "Dataset", "Sequences", "processed", "train")
MIN_NEW_SAMPLES_THRESHOLD = 5  # Ngưỡng số mẫu mới tối thiểu để kích hoạt Retrain

# =====================================================================
# NÂNG CẤP 1: BỘ LỌC KÍCH HOẠT THÔNG MINH (Threshold Trigger)
# =====================================================================
def count_new_labeled_samples():
    """Đếm tổng số lượng tệp mẫu mới đang chờ trong thư mục Labeled."""
    if not os.path.exists(LABELED_DIR):
        return 0
    total = 0
    for label in os.listdir(LABELED_DIR):
        label_dir = os.path.join(LABELED_DIR, label)
        if os.path.isdir(label_dir):
            files = [f for f in os.listdir(label_dir) if f.endswith(('.npy', '.json', '.mp4', '.avi'))]
            total += len(files)
    return total

# =====================================================================
# NÂNG CẤP 2: TỰ ĐỘNG GỘP DỮ LIỆU & TIỀN XỬ LÝ KEYPOINTS
# =====================================================================
def merge_labeled_data():
    """Di chuyển các mẫu dữ liệu mới đã gán nhãn vào tập Dataset chính."""
    if not os.path.exists(LABELED_DIR):
        print(f"Thư mục {LABELED_DIR} chưa tồn tại. Bỏ qua bước gộp dữ liệu.")
        return 0
        
    merged_count = 0
    for label in os.listdir(LABELED_DIR):
        src_label_dir = os.path.join(LABELED_DIR, label)
        if not os.path.isdir(src_label_dir):
            continue
        dst_label_dir = os.path.join(SEQUENCES_TRAIN_DIR, label)
        os.makedirs(dst_label_dir, exist_ok=True)
        
        for file_name in os.listdir(src_label_dir):
            if file_name.endswith(('.npy', '.mp4', '.avi', '.mov', '.json')):
                src = os.path.join(src_label_dir, file_name)
                dst = os.path.join(dst_label_dir, file_name)
                shutil.move(src, dst)
                merged_count += 1
                
        # Sau khi move, xóa thư mục rỗng
        if not os.listdir(src_label_dir):
            os.rmdir(src_label_dir)
            
    print(f"Đã gộp thành công {merged_count} mẫu mới vào tập dữ liệu huấn luyện.")
    return merged_count

# =====================================================================
# NÂNG CẤP 3: CHỐT CHẶN AN TOÀN & ROLLBACK (Safety Gate & Evaluation)
# =====================================================================
def evaluate_new_model(previous_best_acc=0.30):
    """
    Đọc chỉ số Val Accuracy từ mô hình mới train xong và so sánh với mô hình cũ.
    Nếu khá hơn (hoặc đạt ngưỡng tối thiểu) -> Cho phép cập nhật.
    Nếu tệ hơn -> Từ chối và Rollback.
    """
    metrics_log_file = os.path.join(BASE_DIR, "Cloud_server", "Trainer", "latest_train_metrics.json")
    new_acc = 0.0
    if os.path.exists(metrics_log_file):
        try:
            with open(metrics_log_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                new_acc = data.get("val_accuracy", 0.0)
        except Exception:
            pass

    print(f"📊 Đánh giá độ chính xác: Mô hình mới = {new_acc:.2%} | Mô hình cũ = {previous_best_acc:.2%}")
    if new_acc >= previous_best_acc:
        print("✅ Mô hình mới đạt độ chính xác vượt trội/ổn định. Cho phép nâng cấp phiên bản!")
        return True, new_acc
    else:
        print("⚠️ CẢNH BÁO: Mô hình mới có độ chính xác thấp hơn mô hình cũ. Kích hoạt Rollback giữ nguyên phiên bản cũ!")
        return False, new_acc

# =====================================================================
# QUY TRÌNH RETRAINING HOÀN CHỈNH
# =====================================================================
def run_retraining(force=False):
    print("=== BẮT ĐẦU QUY TRÌNH TỰ ĐỘNG RETRAINING ===")
    
    # 1. Kiểm tra kích hoạt thông minh
    new_samples = count_new_labeled_samples()
    print(f"Số lượng mẫu mới nhận diện được: {new_samples} mẫu.")
    if not force and new_samples < MIN_NEW_SAMPLES_THRESHOLD:
        print(f"Chưa đủ ngưỡng {MIN_NEW_SAMPLES_THRESHOLD} mẫu mới để Retrain. Tiến trình dừng an toàn.")
        return

    # 2. Gộp dữ liệu mới
    merge_labeled_data()

    # 3. Thực thi script train BiGRU
    train_gru_script = os.path.join(BASE_DIR, "Cloud_server", "Trainer", "train_scripts", "train_gru.py")
    if os.path.exists(train_gru_script):
        print(f"🚀 Đang chạy huấn luyện lại mô hình BiGRU...")
        subprocess.run([sys.executable, train_gru_script], check=True)

    # 4. Kiểm tra Chốt chặn An toàn (Safety Gate)
    is_safe, new_acc = evaluate_new_model(previous_best_acc=0.30)
    if not is_safe:
        print("❌ Hủy bỏ đăng ký phiên bản mới để đảm bảo độ ổn định cho hệ thống.")
        return

    # 5. Nếu đạt yêu cầu an toàn: Đăng ký phiên bản mới
    new_version = increment_version()
    registry = ModelRegistry()
    
    yolo_path = os.path.join(config.SHARED_ASSETS_DIR, "hand_det_yolo.pt") if hasattr(config, "SHARED_ASSETS_DIR") else os.path.join(BASE_DIR, "Shared_lib", "Assets", "hand_det_yolo.pt")
    feat_path = os.path.join(config.SHARED_ASSETS_DIR, "feature_extractor.h5") if hasattr(config, "SHARED_ASSETS_DIR") else os.path.join(BASE_DIR, "Shared_lib", "Assets", "feature_extractor.h5")
    gru_path = os.path.join(config.SHARED_ASSETS_DIR, "action_recognizer.h5") if hasattr(config, "SHARED_ASSETS_DIR") else os.path.join(BASE_DIR, "Shared_lib", "Assets", "action_recognizer.h5")

    registry.register_new_version(
        new_version,
        yolo_path,
        feat_path,
        gru_path
    )

    # 6. Chuyển đổi mô hình sang TFLite cho App
    convert_script = os.path.join(BASE_DIR, "Tools", "convert_to_mobile.py")
    if os.path.exists(convert_script):
        print(f"📲 Đang chuyển đổi sang TFLite cho ứng dụng di động...")
        subprocess.run([sys.executable, convert_script], check=True)

    print(f"🎉 Hoàn tất Retraining xuất sắc! Đã phát hành phiên bản mới: {new_version} (Val Acc: {new_acc:.2%})")

def increment_version():
    registry = ModelRegistry()
    latest = registry.get_latest_version()
    major, minor, patch = map(int, latest.split('.'))
    patch += 1
    return f"{major}.{minor}.{patch}"

if __name__ == "__main__":
    run_retraining(force=False)