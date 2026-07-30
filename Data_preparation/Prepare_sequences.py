# SIGN_LANGUAGE_MARKET_READY/data_preparation/Prepare_sequences.py
import os
import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import json
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import config

def extract_keypoints(video_path, hands, seq_len=30):
    """Trích xuất chuỗi keypoints từ video — 2 tay, phân biệt trái/phải."""
    NUM_POINTS = 21 * 3             # 63 số/tay
    TOTAL_FEATURES = NUM_POINTS * 2  # 126 số = 2 tay
    
    cap = cv2.VideoCapture(video_path)
    kp_seq = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)
        
        # Tạo vector 126 số: [tay_trái (63) | tay_phải (63)], mặc định = 0
        kp = [0.0] * TOTAL_FEATURES
        
        if results.multi_hand_landmarks and results.multi_handedness:
            for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                # Đọc nhãn trái/phải từ MediaPipe
                label = results.multi_handedness[idx].classification[0].label
                
                # "Left" → vị trí 0-62 (63 số đầu)
                # "Right" → vị trí 63-125 (63 số sau)
                offset = 0 if label == "Left" else NUM_POINTS
                
                for j, lm in enumerate(hand_landmarks.landmark):
                    kp[offset + j * 3]     = lm.x
                    kp[offset + j * 3 + 1] = lm.y
                    kp[offset + j * 3 + 2] = lm.z
        
        kp_seq.append(kp)
    cap.release()
    
    if len(kp_seq) == 0:
        return None
    
    seq = np.array(kp_seq, dtype=np.float32)
    if len(seq) >= seq_len:
        return seq[:seq_len]
    else:
        pad = np.zeros((seq_len - len(seq), TOTAL_FEATURES), dtype=np.float32)
        return np.vstack([seq, pad])

#=========XỬ LÝ DỮ LIỆU NGOÀI DANH SÁCH WLASL (CSV)=========
#Đôi khi trong quá trình làm việc, bạn tải thêm một vài video mới hoặc tự quay video bổ sung nhưng chưa kịp cập nhật ID vào file JSON hay CSV.
def process_split_from_folders(base_dir, split_name, hands, seq_len, output_base, processed_ids=None):
    """Xử lý khi dữ liệu đã được tổ chức sẵn trong train/val theo lớp."""
    if processed_ids is None:
        processed_ids = set()
    split_dir = os.path.join(base_dir, split_name)
    if not os.path.exists(split_dir):
        print(f"⚠️ Không tìm thấy thư mục {split_dir}")
        return 0, 0

    success = 0
    failed = 0
    for class_name in os.listdir(split_dir):
        class_path = os.path.join(split_dir, class_name)
        if not os.path.isdir(class_path):
            continue
        output_class_dir = os.path.join(output_base, split_name, class_name)
        os.makedirs(output_class_dir, exist_ok=True)

        video_files = [f for f in os.listdir(class_path) if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))]
        for vid in tqdm(video_files, desc=f"{split_name}/{class_name}"):
            vid_id = os.path.splitext(vid)[0]
            if vid_id in processed_ids:
                continue
            video_path = os.path.join(class_path, vid)
            seq = extract_keypoints(video_path, hands, seq_len)
            if seq is None:
                failed += 1
                continue
            np.save(os.path.join(output_class_dir, f"{vid_id}.npy"), seq)
            processed_ids.add(vid_id)
            success += 1
    return success, failed

def process_from_wlasl_json(json_path, videos_dir, hands, seq_len, output_base):
    """Xử lý dữ liệu WLASL, tự động chia train/val 80/20 và gộp test vào val."""
    processed = set()
    if not os.path.exists(json_path):
        print(f"❌ Không tìm thấy file nhãn: {json_path}")
        return processed

    with open(json_path, 'r') as f:
        data = json.load(f)

    print(f"📂 Đang xử lý dữ liệu từ: {os.path.basename(json_path)}")

    train_data = []
    val_data = []
    missing = 0

    for video_id, info in tqdm(data.items(), desc="Thu thập video WLASL"):
        subset = info.get('subset')
        if subset not in ['train', 'val', 'test']:
            continue

        label = info['action'][0]

        # Tìm file video (hỗ trợ nhiều định dạng)
        video_path = None
        for ext in ['.mp4', '.avi', '.mov', '.mkv']:
            candidate = os.path.join(videos_dir, f"{video_id}{ext}")
            if os.path.exists(candidate):
                video_path = candidate
                break

        if video_path is None:
            missing += 1
            continue

        if subset == 'train':
            train_data.append((video_id, video_path, label))
        else: # Gộp 'val' và 'test' thành tập val chung
            val_data.append((video_id, video_path, label))

    print(f"🔍 Tìm thấy {len(train_data) + len(val_data)} video hợp lệ (bỏ qua {missing} video bị thiếu).")

    if len(train_data) + len(val_data) == 0:
        print("❌ Không có video nào để xử lý.")
        return processed

    print(f"✂️ Phân loại theo JSON gốc: Train = {len(train_data)} video, Val = {len(val_data)} video.")

    # Hàm con xử lý từng tập
    def process_split(video_list, split_name):
        success = 0
        for (vid_id, vid_path, lbl) in tqdm(video_list, desc=f"Xử lý {split_name}"):
            seq = extract_keypoints(vid_path, hands, seq_len)
            if seq is not None:
                dst_dir = os.path.join(output_base, split_name, str(lbl))
                os.makedirs(dst_dir, exist_ok=True)
                np.save(os.path.join(dst_dir, f"{vid_id}.npy"), seq)
                processed.add(vid_id)
                success += 1
        return success

    train_success = process_split(train_data, 'train')
    val_success = process_split(val_data, 'val')

    print(f"✅ Đã xử lý Train: {train_success}, Val: {val_success}. (Tổng: {len(processed)} video)")
    return processed

def main():
    print("=== Chuẩn bị dữ liệu Sequences (kết hợp giữa JSON VÀ CSV) ===")
    
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    SEQ_LEN = 30
    output_dir = os.path.join(config.SEQUENCES_DIR, 'processed')
    os.makedirs(output_dir, exist_ok=True)
    processed_ids = set()
   
    # --- 1. XỬ LÝ NGUỒN JSON (WLASL) ---
    json_path = config.SEQUENCES_JSON
    if json_path and os.path.exists(json_path):
        new_ids = process_from_wlasl_json(json_path, config.SEQUENCES_VIDEOS_DIR, hands, SEQ_LEN, output_dir)
        processed_ids.update(new_ids)
        print(f"✅ Đã xử lý {len(new_ids)} video từ JSON. (Tổng: {len(processed_ids)})")
    else:
        print("ℹ️ Không có file JSON được cấu hình, bỏ qua WLASL.")

    # --- 2. XỬ LÝ NGUỒN CSV (Dữ liệu bổ sung) ---
    csv_path = config.SEQUENCES_CSV
    if csv_path and os.path.exists(csv_path) and csv_path.endswith('.csv'):
        print(f"📂 Xử lý video mới từ CSV: {os.path.basename(csv_path)}...")
        df = pd.read_csv(csv_path)
        
        labels_in_csv = [col for col in df.columns if col != 'set_id']
        csv_data = []
        
        for index, row in df.iterrows():
            group_id = row['set_id']
            for label in labels_in_csv:
                video_rel_path = row[label]
                if pd.isna(video_rel_path): continue
                
                vid_id = f"{group_id}_{os.path.splitext(os.path.basename(str(video_rel_path)))[0]}"
                if vid_id in processed_ids:
                    continue
                
                v_path = os.path.join(config.SEQUENCES_DIR, str(video_rel_path))
                if os.path.exists(v_path):
                    csv_data.append((vid_id, v_path, label, group_id))
        
        if csv_data:
            from sklearn.model_selection import GroupShuffleSplit
            
            paths = [item[1] for item in csv_data]
            lbls = [item[2] for item in csv_data]
            groups = [item[3] for item in csv_data]
            
            gss = GroupShuffleSplit(n_splits=1, train_size=0.8, random_state=42)
            train_idx, val_idx = next(gss.split(paths, lbls, groups))
            
            train_csv = [csv_data[i] for i in train_idx]
            val_csv = [csv_data[i] for i in val_idx]
            
            print(f"✂️ CSV chia theo Group (set_id): Train = {len(train_csv)}, Val = {len(val_csv)}")
            
            def process_csv_list(v_list, split_name):
                for (vid_id, v_path, lbl, _) in tqdm(v_list, desc=f"CSV {split_name}"):
                    seq = extract_keypoints(v_path, hands, SEQ_LEN)
                    if seq is not None:
                        d_dir = os.path.join(output_dir, split_name, str(lbl))
                        os.makedirs(d_dir, exist_ok=True)
                        np.save(os.path.join(d_dir, f"{vid_id}.npy"), seq)
                        processed_ids.add(vid_id)
            
            process_csv_list(train_csv, 'train')
            process_csv_list(val_csv, 'val')
            print(f"✅ Đã xử lý thêm video từ CSV. (Tổng: {len(processed_ids)})")

    # --- 3. QUÉT THƯ MỤC CÓ SẴN (Vét dữ liệu còn sót) ---
    base_dirs = [config.SEQUENCES_RAW_DIR, os.path.join(config.SEQUENCES_DIR, 'files')]
    for b in base_dirs:
        if os.path.exists(b):
            s1, f1 = process_split_from_folders(b, 'train', hands, SEQ_LEN, output_dir, processed_ids)
            s2, f2 = process_split_from_folders(b, 'val', hands, SEQ_LEN, output_dir, processed_ids)
            print(f"   Thư mục {b}: train thành công {s1} (lỗi {f1}), val thành công {s2} (lỗi {f2})")

    print(f"✅ HOÀN TẤT! Tổng cộng đã xử lý {len(processed_ids)} video.")
    print(f"📍 Dữ liệu đích: {output_dir}")

if __name__ == "__main__":
    main()                                                                                                                                            