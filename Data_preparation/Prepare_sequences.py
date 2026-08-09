# SIGN_LANGUAGE_MARKET_READY/data_preparation/Prepare_sequences.py
import os
import cv2
from mediapipe.python.solutions import hands as mp_hands
import numpy as np
import pandas as pd
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

def process_from_dataset_folder(dataset_dir, hands, seq_len, output_base, processed_ids=None):
    """Quét dữ liệu trực tiếp từ các thư mục lớp từ vựng (ví dụ: archive/dataset/SL/apple, book...)."""
    if processed_ids is None:
        processed_ids = set()

    if not os.path.exists(dataset_dir):
        print(f"⚠️ Không tìm thấy thư mục dataset: {dataset_dir}")
        return processed_ids

    print(f"📂 Đang quét dữ liệu từ thư mục: {dataset_dir}")
    class_names = [d for d in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, d))]
    print(f"🔍 Phát hiện {len(class_names)} lớp từ vựng.")

    total_train = 0
    total_val = 0

    for class_name in tqdm(class_names, desc="Xử lý từng lớp từ vựng"):
        class_path = os.path.join(dataset_dir, class_name)
        video_files = [f for f in os.listdir(class_path) if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))]

        if not video_files:
            continue

        # Chia 80% Train, 20% Val cho mỗi lớp từ vựng
        if len(video_files) > 1:
            train_vids, val_vids = train_test_split(video_files, train_size=0.8, random_state=42)
        else:
            train_vids = video_files
            val_vids = []

        # Hàm con xử lý danh sách video
        def process_list(v_list, split_name):
            count = 0
            for vid in v_list:
                vid_id = os.path.splitext(vid)[0]
                full_id = f"{class_name}_{vid_id}"
                if full_id in processed_ids:
                    continue

                # Kiểm tra nếu file .npy đã tồn tại trên ổ đĩa -> Bỏ qua ngay lập tức để chạy tiếp nối!
                dst_dir = os.path.join(output_base, split_name, class_name)
                npy_file = os.path.join(dst_dir, f"{vid_id}.npy")
                if os.path.exists(npy_file):
                    processed_ids.add(full_id)
                    continue

                v_path = os.path.join(class_path, vid)
                try:
                    seq = extract_keypoints(v_path, hands, seq_len)
                    if seq is not None:
                        os.makedirs(dst_dir, exist_ok=True)
                        np.save(npy_file, seq)
                        processed_ids.add(full_id)
                        count += 1
                except Exception as e:
                    print(f"⚠️ Bỏ qua video lỗi {v_path}: {e}")
            return count

        total_train += process_list(train_vids, 'train')
        total_val += process_list(val_vids, 'val')

    print(f"✅ Hoàn tất trích xuất từ Dataset: Train = {total_train} video, Val = {total_val} video.")
    return processed_ids

def main():
    print("=== Chuẩn bị dữ liệu Sequences (Trích xuất Keypoints từ Video Folder) ===")
    
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

    # --- 1. XỬ LÝ NGUỒN DATASET THƯ MỤC LỚP (archive/dataset/SL) ---
    dataset_dir = getattr(config, 'SEQUENCES_DATASET_DIR', os.path.join(config.SEQUENCES_DIR, 'archive', 'dataset', 'SL'))
    if os.path.exists(dataset_dir):
        process_from_dataset_folder(dataset_dir, hands, SEQ_LEN, output_dir, processed_ids)
    else:
        print(f"ℹ️ Không tìm thấy thư mục dataset tại {dataset_dir}, bỏ qua.")

    # --- 2. XỬ LÝ NGUỒN CSV BỔ SUNG (nếu có) ---
    csv_path = getattr(config, 'SEQUENCES_CSV', None)
    if csv_path and os.path.exists(csv_path) and csv_path.endswith('.csv'):
        print(f"📂 Xử lý video bổ sung từ CSV: {os.path.basename(csv_path)}...")
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

    print(f"✅ HOÀN TẤT! Tổng cộng đã xử lý {len(processed_ids)} chuỗi cử chỉ.")
    print(f"📍 Dữ liệu đích: {output_dir}")

    hands.close()

if __name__ == "__main__":
    main()