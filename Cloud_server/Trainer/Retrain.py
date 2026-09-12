# Cloud_server/Trainer/Retrain.py
"""
==============================================================================
HỆ THỐNG ĐIỀU PHỐI TỰ ĐỘNG RETRAINING TOÀN DIỆN (MLOPS AUTOMATION PIPELINE)
==============================================================================
Nhiệm vụ:
1. Quét đa nguồn dữ liệu: Feedback API Server (Labeled/) + Video Ngắn + Video Dài Cắt Đoạn (custom_enrollment/).
2. Tự động trích xuất 126 Keypoints (MediaPipe Hands) từ các tệp video .mp4 thô sang .npy.
3. Đồng bộ vào tập huấn luyện chính Sequences/processed/train/.
4. Kích hoạt huấn luyện mô hình GRU (train_gru.py).
5. Đánh giá Chốt chặn An toàn (Safety Gate) & Tự động xuất model sang TFLite cho Mobile/App.
==============================================================================
"""

import os
import sys
import shutil
import json
import subprocess
import cv2
import numpy as np
from datetime import datetime

# Đảm bảo mã hóa UTF-8 khi chạy trên Terminal Windows
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

# Thiết lập đường dẫn gốc BASE_DIR
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import config
from Cloud_server.Model_registry.registry import ModelRegistry
from Shared_lib.sequence_utils import resample_sequence

# Các đường dẫn dữ liệu đa nguồn
LABELED_DIR = config.LABELED_DIR if hasattr(config, 'LABELED_DIR') else os.path.join(BASE_DIR, "Cloud_server", "Database", "edge_cases", "Labeled")
CUSTOM_ENROLL_DIR = os.path.join(BASE_DIR, "Research_and_Data", "Dataset", "Sequences", "custom_enrollment")
LONG_SEGMENTS_DIR = os.path.join(CUSTOM_ENROLL_DIR, "long_video_segments")
SEQUENCES_TRAIN_DIR = os.path.join(config.SEQUENCES_PROCESSED_DIR, "train")

MIN_NEW_SAMPLES_THRESHOLD = 5  # Ngưỡng số mẫu mới tối thiểu để kích hoạt Retrain

# =====================================================================
# 1. BỘ ĐẾM MẪU MỚI THÔNG MINH (MULTI-SOURCE SAMPLE COUNTER)
# =====================================================================
def count_all_pending_samples():
    """Đếm tổng số lượng mẫu mới từ cả 3 nguồn (API Server + Video Ngắn + Video Dài Cắt Đoạn)."""
    counts = {
        "labeled": 0,
        "custom_enrollment": 0,
        "long_segments": 0
    }

    # 1. Nguồn Labeled (API Server feedback)
    if os.path.exists(LABELED_DIR):
        for label in os.listdir(LABELED_DIR):
            p = os.path.join(LABELED_DIR, label)
            if os.path.isdir(p):
                counts["labeled"] += len([f for f in os.listdir(p) if f.endswith(('.npy', '.json', '.mp4', '.avi'))])

    # 2. Nguồn Custom Enrollment (Video mẫu ngắn)
    if os.path.exists(CUSTOM_ENROLL_DIR):
        for label in os.listdir(CUSTOM_ENROLL_DIR):
            if label == "long_video_segments":
                continue
            p = os.path.join(CUSTOM_ENROLL_DIR, label)
            if os.path.isdir(p):
                counts["custom_enrollment"] += len([f for f in os.listdir(p) if f.endswith(('.mp4', '.avi', '.npy'))])

    # 3. Nguồn Video Dài Cắt Đoạn (Long Video Segments)
    if os.path.exists(LONG_SEGMENTS_DIR):
        for vid_folder in os.listdir(LONG_SEGMENTS_DIR):
            p = os.path.join(LONG_SEGMENTS_DIR, vid_folder)
            if os.path.isdir(p):
                counts["long_segments"] += len([f for f in os.listdir(p) if f.endswith(('.mp4', '.avi'))])

    total = sum(counts.values())
    return total, counts

# =====================================================================
# 2. TRÍCH XUẤT 126 KEYPOINTS TỪ TỆP VIDEO .MP4
# =====================================================================
def extract_keypoints_from_video(video_path, hands_detector, seq_len=30, yolo_model=None):
    """Trích xuất mảng (30, 126) keypoints từ video bằng MediaPipe Hands (Hỗ trợ Hybrid YOLO)."""
    NUM_POINTS = 21 * 3             # 63 toạ độ / 1 tay
    TOTAL_FEATURES = NUM_POINTS * 2 # 126 toạ độ = 2 tay (Trái + Phải)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        cap.release()
        return None

    if total_frames > seq_len:
        target_frame_indices = sorted(list(set(np.linspace(0, total_frames - 1, seq_len, dtype=int))))
    else:
        target_frame_indices = list(range(total_frames))

    kp_seq = []
    for f_idx in target_frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(f_idx))
        ret, frame = cap.read()
        if not ret:
            break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_h, img_w, _ = frame_rgb.shape
        
        # --- HYBRID PIPELINE LOGIC (Training) ---
        kp = [0.0] * TOTAL_FEATURES
        if yolo_model:
            yolo_results = yolo_model(frame_rgb, verbose=False)
            boxes = yolo_results[0].boxes.xyxy.cpu().numpy()
            
            for box in boxes:
                x1, y1, x2, y2 = map(int, box[:4])
                pad_x = int((x2 - x1) * 0.25)
                pad_y = int((y2 - y1) * 0.25)
                x1 = max(0, x1 - pad_x)
                y1 = max(0, y1 - pad_y)
                x2 = min(img_w, x2 + pad_x)
                y2 = min(img_h, y2 + pad_y)
                
                crop_w, crop_h = x2 - x1, y2 - y1
                if crop_w < 10 or crop_h < 10: continue
                
                crop = frame_rgb[y1:y2, x1:x2]
                if crop_w < 256 or crop_h < 256:
                    crop = cv2.resize(crop, (256, 256), interpolation=cv2.INTER_LINEAR)
                    
                mp_res = hands_detector.process(crop)
                if mp_res.multi_hand_landmarks:
                    hand_lm = mp_res.multi_hand_landmarks[0]
                    label = mp_res.multi_handedness[0].classification[0].label if mp_res.multi_handedness else "Left"
                    offset = 0 if label == "Left" else NUM_POINTS
                    
                    for j, lm in enumerate(hand_lm.landmark):
                        orig_x = (lm.x * crop_w + x1) / img_w
                        orig_y = (lm.y * crop_h + y1) / img_h
                        kp[offset + j * 3]     = orig_x
                        kp[offset + j * 3 + 1] = orig_y
                        kp[offset + j * 3 + 2] = lm.z
        else:
            results = hands_detector.process(frame_rgb)
            if results.multi_hand_landmarks and results.multi_handedness:
                for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                    label = results.multi_handedness[idx].classification[0].label
                    offset = 0 if label == "Left" else NUM_POINTS
                    for j, lm in enumerate(hand_landmarks.landmark):
                        kp[offset + j * 3]     = lm.x
                        kp[offset + j * 3 + 1] = lm.y
                        kp[offset + j * 3 + 2] = lm.z

        # Store raw coordinates. The shared fixed-anchor normalization is
        # applied once, identically, by training and inference.
        kp_seq.append(np.array(kp, dtype=np.float32))

    cap.release()
    
    if not kp_seq:
        return None
        
    return resample_sequence(kp_seq, seq_len)

# =====================================================================
# 3. QUẢN LÝ DỮ LIỆU & ĐỒNG BỘ NPY (DATASET MANAGER)
# =====================================================================
def sync_and_extract_new_data():
    """Tự động đồng bộ toàn bộ dữ liệu mới (Video/Npy) vào Sequences/processed/train/."""
    from mediapipe.python.solutions import hands as mp_hands
    
    # Kích hoạt ống nhòm YOLO cho Train nếu có
    yolo_model = None
    yolo_path = os.path.join(config.SHARED_ASSETS_DIR, 'hand_det_yolo.pt') if hasattr(config, 'SHARED_ASSETS_DIR') else os.path.join(BASE_DIR, "Shared_lib", "Assets", "hand_det_yolo.pt")
    if getattr(config, 'USE_YOLO_HAND_CROPS', False) and os.path.exists(yolo_path):
        from ultralytics import YOLO
        yolo_model = YOLO(yolo_path)
        print("👁️ [Hybrid Retrain] Ống nhòm YOLOv8 đã được kích hoạt cho việc trích xuất dữ liệu đào tạo!")

    # Nếu dùng YOLO thì tĩnh, không thì động
    static_mode = True if yolo_model else False
    detector = mp_hands.Hands(static_image_mode=static_mode, max_num_hands=2, min_detection_confidence=0.3)
    total_synced = 0

    # A. Xử lý Video từ custom_enrollment (Video ngắn)
    if os.path.exists(CUSTOM_ENROLL_DIR):
        for word_name in os.listdir(CUSTOM_ENROLL_DIR):
            if word_name == "long_video_segments":
                continue
            src_dir = os.path.join(CUSTOM_ENROLL_DIR, word_name)
            if not os.path.isdir(src_dir):
                continue

            dst_dir = os.path.join(SEQUENCES_TRAIN_DIR, word_name)
            os.makedirs(dst_dir, exist_ok=True)

            for f_name in os.listdir(src_dir):
                f_path = os.path.join(src_dir, f_name)
                npy_name = os.path.splitext(f_name)[0] + ".npy"
                dst_npy = os.path.join(dst_dir, npy_name)

                if f_name.endswith(('.mp4', '.avi', '.mov')):
                    if not os.path.exists(dst_npy):
                        kp = extract_keypoints_from_video(f_path, detector, yolo_model=yolo_model)
                        if kp is not None:
                            np.save(dst_npy, kp)
                            total_synced += 1
                elif f_name.endswith('.npy'):
                    if not os.path.exists(dst_npy):
                        shutil.copy2(f_path, dst_npy)
                        total_synced += 1

    # B. Xử lý Video dài cắt đoạn (long_video_segments)
    if os.path.exists(LONG_SEGMENTS_DIR):
        for vid_folder in os.listdir(LONG_SEGMENTS_DIR):
            src_dir = os.path.join(LONG_SEGMENTS_DIR, vid_folder)
            if not os.path.isdir(src_dir):
                continue

            dst_dir = os.path.join(SEQUENCES_TRAIN_DIR, vid_folder)
            os.makedirs(dst_dir, exist_ok=True)

            for f_name in os.listdir(src_dir):
                if f_name.endswith(('.mp4', '.avi', '.mov')):
                    f_path = os.path.join(src_dir, f_name)
                    npy_name = os.path.splitext(f_name)[0] + ".npy"
                    dst_npy = os.path.join(dst_dir, npy_name)

                    if not os.path.exists(dst_npy):
                        kp = extract_keypoints_from_video(f_path, detector, yolo_model=yolo_model)
                        if kp is not None:
                            np.save(dst_npy, kp)
                            total_synced += 1

    # C. Xử lý dữ liệu Labeled (API Server)
    if os.path.exists(LABELED_DIR):
        for label in os.listdir(LABELED_DIR):
            src_label_dir = os.path.join(LABELED_DIR, label)
            if not os.path.isdir(src_label_dir):
                continue
            dst_label_dir = os.path.join(SEQUENCES_TRAIN_DIR, label)
            os.makedirs(dst_label_dir, exist_ok=True)

            for file_name in os.listdir(src_label_dir):
                src = os.path.join(src_label_dir, file_name)
                if file_name.endswith('.npy'):
                    dst = os.path.join(dst_label_dir, file_name)
                    shutil.move(src, dst)
                    total_synced += 1
                elif file_name.endswith('.json'):
                    try:
                        with open(src, 'r', encoding='utf-8') as stream:
                            feedback = json.load(stream)
                        raw = np.asarray(feedback.get('raw_data'), dtype=np.float32)
                        if raw.ndim == 1 and raw.size == 126:
                            raw = raw.reshape(1, 126)
                        if raw.ndim != 2 or raw.shape[1] != 126 or len(raw) == 0:
                            raise ValueError(f"raw_data has invalid shape {raw.shape}")
                        dst_npy = os.path.join(dst_label_dir, os.path.splitext(file_name)[0] + '.npy')
                        np.save(dst_npy, resample_sequence(raw, 30))
                        os.remove(src)
                        total_synced += 1
                    except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
                        print(f"⚠️ Bỏ qua feedback không hợp lệ {src}: {exc}")
                elif file_name.endswith(('.mp4', '.avi')):
                    npy_name = os.path.splitext(file_name)[0] + ".npy"
                    dst_npy = os.path.join(dst_label_dir, npy_name)
                    kp = extract_keypoints_from_video(src, detector, yolo_model=yolo_model)
                    if kp is not None:
                        np.save(dst_npy, kp)
                        total_synced += 1
                    os.remove(src)

            if not os.listdir(src_label_dir):
                os.rmdir(src_label_dir)

    detector.close()
    print(f"✅ Đã đồng bộ & trích xuất thành công {total_synced} tệp Keypoints (.npy) mới vào kho Train!")
    return total_synced

# =====================================================================
# 4. CHỐT CHẶN AN TOÀN & ĐÁNH GIÁ MÔ HÌNH (SAFETY GATE)
# =====================================================================
def evaluate_new_model(previous_best_acc=0.30):
    """Đọc Val Accuracy từ mô hình mới và so sánh với mô hình cũ."""
    metrics_log_file = os.path.join(BASE_DIR, "Cloud_server", "Trainer", "latest_train_metrics.json")
    new_acc = 0.0
    if os.path.exists(metrics_log_file):
        try:
            with open(metrics_log_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                new_acc = data.get("val_accuracy", 0.0)
        except Exception:
            pass

    print(f"📊 Đánh giá độ chính xác: Mô hình mới = {new_acc:.2%} | Ngưỡng an toàn = {previous_best_acc:.2%}")
    if new_acc >= previous_best_acc:
        print("✅ Mô hình mới đạt độ chính xác ổn định. Cho phép nâng cấp phiên bản!")
        return True, new_acc
    else:
        print("⚠️ CẢNH BÁO: Độ chính xác chưa đạt. Kích hoạt Rollback giữ nguyên phiên bản cũ!")
        return False, new_acc

def increment_version():
    """Tự động tăng số phiên bản Semantic Versioning."""
    registry = ModelRegistry()
    latest = registry.get_latest_version()
    major, minor, patch = map(int, latest.split('.'))
    patch += 1
    return f"{major}.{minor}.{patch}"

# =====================================================================
# 5. ĐIỀU PHỐI QUY TRÌNH RETRAINING TOÀN DIỆN
# =====================================================================
def run_retraining(force=False):
    print("=" * 70, flush=True)
    print("🚀 BẮT ĐẦU QUY TRÌNH TỰ ĐỘNG RETRAINING (FULL PIPELINE)", flush=True)
    print("=" * 70, flush=True)

    registry = ModelRegistry()
    previous_best_acc = registry.get_latest_accuracy(default=0.30)

    # 1. Kiểm tra kích hoạt thông minh đa nguồn
    total_new, counts = count_all_pending_samples()
    print(f"📊 Thống kê mẫu mới phát hiện:")
    print(f"   • API Server Labeled : {counts['labeled']} mẫu")
    print(f"   • Custom Enrollment   : {counts['custom_enrollment']} mẫu")
    print(f"   • Long Video Segments : {counts['long_segments']} mẫu")
    print(f"   👉 TỔNG CỘNG: {total_new} mẫu mới.")

    if not force and total_new < MIN_NEW_SAMPLES_THRESHOLD:
        print(f"⚠️ Chưa đủ ngưỡng {MIN_NEW_SAMPLES_THRESHOLD} mẫu mới để Retrain. Tiến trình dừng an toàn.")
        return False

    # 2. Tự động đồng bộ và trích xuất Keypoints 126 toạ độ sang .npy
    print("\n[Bước 1/4] 📥 Đang trích xuất Keypoints & Đồng bộ dữ liệu vào Dataset...", flush=True)
    sync_and_extract_new_data()

    # 3. Kích hoạt huấn luyện mạng GRU
    print("\n[Bước 2/4] 🧠 Đang kích hoạt huấn luyện mô hình GRU (train_gru.py)...", flush=True)
    train_gru_script = os.path.join(BASE_DIR, "Cloud_server", "Trainer", "train_scripts", "train_gru.py")
    if os.path.exists(train_gru_script):
        subprocess.run([sys.executable, train_gru_script], check=True)
    else:
        print(f"❌ Không tìm thấy script: {train_gru_script}")
        return False

    # 4. Kiểm tra Chốt chặn An toàn (Safety Gate)
    print("\n[Bước 3/4] 🛡️ Đang kiểm tra chất lượng mô hình qua Chốt chặn An toàn...", flush=True)
    is_safe, new_acc = evaluate_new_model(previous_best_acc=previous_best_acc)
    if not is_safe:
        print("❌ Hủy bỏ đăng ký phiên bản mới để bảo vệ độ ổn định.")
        return False

    # 5. Đăng ký phiên bản mới & Convert sang TFLite cho App
    print("\n[Bước 4/4] 📲 Đang đăng ký phiên bản mới và xuất mô hình TFLite cho App...", flush=True)
    new_version = increment_version()
    yolo_path = os.path.join(config.SHARED_ASSETS_DIR, "hand_det_yolo.pt") if hasattr(config, "SHARED_ASSETS_DIR") else os.path.join(BASE_DIR, "Shared_lib", "Assets", "hand_det_yolo.pt")
    feat_path = os.path.join(config.SHARED_ASSETS_DIR, "feature_extractor.h5") if hasattr(config, "SHARED_ASSETS_DIR") else os.path.join(BASE_DIR, "Shared_lib", "Assets", "feature_extractor.h5")
    gru_path = os.path.join(config.SHARED_ASSETS_DIR, "action_recognizer.h5") if hasattr(config, "SHARED_ASSETS_DIR") else os.path.join(BASE_DIR, "Shared_lib", "Assets", "action_recognizer.h5")

    registry.register_new_version(
        new_version, yolo_path, feat_path, gru_path,
        metrics={"val_accuracy": new_acc}
    )

    convert_script = os.path.join(BASE_DIR, "Tools", "convert_to_mobile.py")
    if os.path.exists(convert_script):
        subprocess.run([sys.executable, convert_script], check=True)

    print("\n" + "=" * 70, flush=True)
    print(f"🎉 HOÀN TẤT RETRAINING XUẤT SẮC! PHIÊN BẢN MỚI: {new_version} (Val Acc: {new_acc:.2%})", flush=True)
    print("=" * 70, flush=True)
    return True

if __name__ == "__main__":
    force_run = "--force" in sys.argv
    run_retraining(force=force_run)
