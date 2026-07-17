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
    """Trích xuất chuỗi keypoints từ video."""
    cap = cv2.VideoCapture(video_path)
    kp_seq = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)
        if results.multi_hand_landmarks:
            hand = results.multi_hand_landmarks[0]
            kp = []
            for lm in hand.landmark:
                kp.extend([lm.x, lm.y, lm.z])
            kp_seq.append(kp)
        else:
            kp_seq.append([0.0] * 63)
    cap.release()
    
    if len(kp_seq) == 0:
        return None
    
    seq = np.array(kp_seq, dtype=np.float32)
    if len(seq) >= seq_len:
        return seq[:seq_len]
    else:
        pad = np.zeros((seq_len - len(seq), 63), dtype=np.float32)
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

    # Gom tất cả video hợp lệ (có file tồn tại) và nhãn
    all_data = []   # Mỗi phần tử là (video_id, video_path, label)
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

        all_data.append((video_id, video_path, label))

    print(f"🔍 Tìm thấy {len(all_data)} video hợp lệ (bỏ qua {missing} video bị thiếu).")

    if len(all_data) == 0:
        print("❌ Không có video nào để xử lý.")
        return processed

    # Chia train/val 80/20, giữ phân bố lớp (stratify)
    labels = [lbl for (_, _, lbl) in all_data]
    train_data, val_data = train_test_split(
        all_data, train_size=0.8, random_state=42, stratify=labels
    )

    print(f"✂️ Tự chia: Train = {len(train_data)} video, Val = {len(val_data)} video.")

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
        max_num_hands=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    SEQ_LEN = 30
    output_dir = os.path.join(config.SEQUENCES_DIR, 'processed')
    os.makedirs(output_dir, exist_ok=True)
    processed_ids = set()
   
    # --- 1. XỬ LÝ NGUỒN JSON (WLASL) ---
    json_path = config.SEQUENCES_CSV if config.SEQUENCES_CSV.endswith('.json') else None
    if json_path:
        new_ids = process_from_wlasl_json(json_path, config.SEQUENCES_VIDEOS_DIR, hands, SEQ_LEN, output_dir)
        processed_ids.update(new_ids)
        print(f"✅ Đã xử lý {len(new_ids)} video từ JSON. (Tổng: {len(processed_ids)})")
    else:
        print("ℹ️ Không có file JSON được cấu hình, bỏ qua WLASL.")

    # --- 2. XỬ LÝ NGUỒN CSV (Dữ liệu bổ sung) ---
    csv_path = config.SEQUENCES_CSV.replace('.json', '.csv') if json_path else config.SEQUENCES_CSV
    if os.path.exists(csv_path) and csv_path.endswith('.csv'):
        df = pd.read_csv(csv_path)
        df['vid_id'] = df['video_name'].apply(lambda x: os.path.splitext(str(x))[0])
        df_new = df[~df['vid_id'].isin(processed_ids)]
        
        if not df_new.empty:
            print(f"📂 Xử lý {len(df_new)} video mới từ CSV...")
            train_v, val_v, train_l, val_l = train_test_split(
                df_new['video_name'].tolist(), df_new['label'].tolist(), 
                train_size=0.8, random_state=42
            )
            for v_list, l_list, split in [(train_v, train_l, 'train'), (val_v, val_l, 'val')]:
                for vid, lbl in tqdm(zip(v_list, l_list), total=len(v_list), desc=f"CSV {split}"):
                    v_path = os.path.join(config.SEQUENCES_VIDEOS_DIR, vid)
                    if not os.path.exists(v_path):
                        continue
                    seq = extract_keypoints(v_path, hands, SEQ_LEN)
                    if seq is not None:
                        d_dir = os.path.join(output_dir, split, str(lbl))
                        os.makedirs(d_dir, exist_ok=True)
                        np.save(os.path.join(d_dir, f"{os.path.splitext(vid)[0]}.npy"), seq)
                        processed_ids.add(os.path.splitext(vid)[0])
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