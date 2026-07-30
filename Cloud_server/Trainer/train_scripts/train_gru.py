# cloud_server/trainer/train_scripts/train_gru.py
"""
Huấn luyện GRU để nhận diện hành động từ chuỗi keypoints.
Sử dụng dữ liệu .npy từ Sequences/processed/train/ và Sequences/processed/val/
"""
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
import config
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import pickle
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau


def load_data(data_dir):
    """Đọc tất cả file .npy và nhãn từ cấu trúc thư mục."""
    X, y = [], []
    classes = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])
    class_to_idx = {cls: i for i, cls in enumerate(classes)}
    
    for cls in classes:
        cls_path = os.path.join(data_dir, cls)
        for file in os.listdir(cls_path):
            if file.endswith('.npy'):
                seq = np.load(os.path.join(cls_path, file))
                X.append(seq)
                y.append(class_to_idx[cls])
    
    return np.array(X), np.array(y), classes

def normalize_keypoints(X):
    """
    Chuẩn hoá keypoints: normalize TỪNG TAY RIÊNG theo cổ tay của chính nó.
    
    Tại sao cần?
      - MediaPipe trả toạ độ TUYỆT ĐỐI trong camera (0.0 → 1.0)
      - Cùng ký hiệu ở vị trí khác → toạ độ KHÁC NHAU hoàn toàn
      - Sau normalize: cùng ký hiệu → toạ độ GIỐNG NHAU (bất kể vị trí tay)
    
    Hỗ trợ cả 1 tay (63 features) và 2 tay (126 features).
    Với 2 tay: [tay_trái (63) | tay_phải (63)] — normalize riêng từng tay.
    """
    X_norm = X.copy().astype(np.float32)
    num_features = X.shape[2]                           # 126 (2 tay) hoặc 63 (1 tay)
    points_per_hand = 21
    features_per_hand = points_per_hand * 3             # 63 số/tay
    num_hands = num_features // features_per_hand       # 2 nếu 126, 1 nếu 63
    
    for i in range(len(X_norm)):
        for t in range(X_norm.shape[1]):                # duyệt 30 frame
            frame = X_norm[i, t]
            
            if np.all(frame == 0):                      # frame padding → bỏ qua
                continue
            
            # Normalize TỪNG TAY RIÊNG BIỆT
            for h in range(num_hands):                  # h=0: tay trái, h=1: tay phải
                start = h * features_per_hand           # tay trái: 0, tay phải: 63
                end = start + features_per_hand         # tay trái: 63, tay phải: 126
                hand_data = frame[start:end]
                
                if np.all(hand_data == 0):              # tay này không có → bỏ qua
                    continue
                
                pts = hand_data.reshape(21, 3)          # [63] → [21 điểm, xyz]
                
                # Lấy cổ tay CỦA TAY NÀY làm gốc
                wrist = pts[0].copy()
                pts = pts - wrist
                
                # Scale theo khoảng cách xa nhất
                max_dist = np.max(np.linalg.norm(pts[:, :2], axis=1))
                if max_dist > 1e-6:
                    pts /= max_dist
                
                X_norm[i, t, start:end] = pts.flatten()
    
    return X_norm

def augment_keypoints(X, y, num_copies=5):
    """
    Tạo biến thể từ dữ liệu gốc (Jitter, Scale, Time Warp, Mirror).
    Hỗ trợ hoán đổi tay trái/phải khi lật gương (Mirror) cho mảng 126 chiều.
    """
    X_aug_list = [X]         # giữ nguyên bản gốc
    y_aug_list = [y]
    seq_len = X.shape[1]     # 30
    num_features = X.shape[2] # 126 (hoặc 63)

    for _ in range(num_copies):
        X_copy = X.copy()
        for i in range(len(X_copy)):
            seq = X_copy[i]
            mask = np.any(seq != 0, axis=1, keepdims=True)  # đánh dấu frame thật (không phải padding)

            # A. Jitter (thêm nhiễu nhẹ, mô phỏng rung tay)
            noise = np.random.normal(0, 0.01, seq.shape)
            seq = seq + noise * mask

            # B. Scale (co giãn kích thước)
            scale = np.random.uniform(0.85, 1.15)
            seq = seq * scale * mask

            # C. Time Warp (co giãn thời gian, xác suất 50%)
            if np.random.random() > 0.5:
                real_len = int(np.sum(np.any(seq != 0, axis=1)))
                if real_len > 5:
                    speed = np.random.uniform(0.8, 1.2)
                    new_len = max(5, int(real_len * speed))
                    indices = np.linspace(0, real_len - 1, new_len).astype(int)
                    warped = seq[indices]
                    if len(warped) >= seq_len:
                        seq = warped[:seq_len]
                    else:
                        pad = np.zeros((seq_len - len(warped), seq.shape[1]))
                        seq = np.vstack([warped, pad])

            X_copy[i] = seq
        X_aug_list.append(X_copy)
        y_aug_list.append(y.copy())

    # D. Mirror: lật gương + ĐỔI CHỖ TAY TRÁI/PHẢI
    X_mirror = X.copy()
    for i in range(len(X_mirror)):
        for t in range(X_mirror.shape[1]):
            frame = X_mirror[i, t]
            if np.all(frame == 0):
                continue
            
            # Nếu dữ liệu có 2 tay (126 features)
            if num_features == 126:
                left_hand = frame[0:63].copy()
                right_hand = frame[63:126].copy()
                
                # Lật trục X (do đã normalize về 0 nên lật là đổi dấu)
                if np.any(left_hand != 0):
                    left_hand[0::3] = -left_hand[0::3]
                if np.any(right_hand != 0):
                    right_hand[0::3] = -right_hand[0::3]
                
                # Tay trái lật xong trở thành tay phải (ghi vào 63:126)
                # Tay phải lật xong trở thành tay trái (ghi vào 0:63)
                frame[0:63] = right_hand
                frame[63:126] = left_hand
            else:
                # Nếu chỉ 1 tay (63 features)
                if np.any(frame != 0):
                    frame[0::3] = -frame[0::3]

    X_aug_list.append(X_mirror)
    y_aug_list.append(y.copy())

    return np.concatenate(X_aug_list), np.concatenate(y_aug_list)

def main():
    print("=== Huấn luyện GRU nhận diện hành động ===")
    
    train_dir = os.path.join(config.SEQUENCES_DIR, 'processed', 'train')
    val_dir = os.path.join(config.SEQUENCES_DIR, 'processed', 'val')
    
    if not os.path.exists(train_dir) or not os.path.exists(val_dir):
        print("❌ Thư mục train/val không tồn tại. Hãy chạy Prepare_sequences.py trước.")
        return False # Trả về False để báo lỗi hệ thống dữ liệu cho Retrain.py

    # Load dữ liệu
    X_train, y_train, classes = load_data(train_dir)
    X_val, y_val, _ = load_data(val_dir)
    
    # [GIẢI PHÁP 1] Chuẩn hoá keypoints: lấy cổ tay làm gốc + scale về [-1, 1]
    # → Cùng ký hiệu ở bất kỳ vị trí nào trong camera đều cho ra vector giống nhau
    print("[GP1] Đang chuẩn hoá keypoints (wrist-relative + scale)...")
    X_train = normalize_keypoints(X_train)
    X_val = normalize_keypoints(X_val)
    
    # [GIẢI PHÁP 2] Tăng cường dữ liệu (Chỉ áp dụng cho tập Train)
    print("[GP2] Đang tăng cường dữ liệu (Augmentation)...")
    X_train, y_train = augment_keypoints(X_train, y_train, num_copies=5)
    
    print(f"Train: {X_train.shape}, Val: {X_val.shape}")
    print(f"Số lớp: {len(classes)}")

    # Chuyển nhãn sang one-hot
    y_train_onehot = tf.keras.utils.to_categorical(y_train, num_classes=len(classes))
    y_val_onehot = tf.keras.utils.to_categorical(y_val, num_classes=len(classes))

    # Tham số
    seq_len = X_train.shape[1]   # 30 frame
    input_dim = X_train.shape[2] # 63 (21 điểm * 3 tọa độ)
    num_classes = len(classes)

    # [GIẢI PHÁP 3 + 4] Thu gọn mô hình chống Overfitting + Tăng cường Regularization (L2)
    # Thu nhỏ BiGRU từ 256/128 xuống 128/64, tăng Dropout lên 0.4, bỏ Dense trung gian
    model = models.Sequential([
        layers.Masking(mask_value=0.0, input_shape=(seq_len, input_dim)),
        
        # Lớp BiGRU 1: Thu nhỏ còn 128
        layers.Bidirectional(layers.GRU(128, return_sequences=True)),
        layers.Dropout(0.4),
        
        # Lớp BiGRU 2: Thu nhỏ còn 64
        layers.Bidirectional(layers.GRU(64, return_sequences=False)),
        layers.Dropout(0.4),
        
        # Phân loại trực tiếp, phạt nặng trọng số với L2 = 0.001
        layers.Dense(num_classes, activation='softmax', kernel_regularizer=tf.keras.regularizers.l2(0.001))
    ])

    # [GIẢI PHÁP 4 + 5] Learning rate = 0.001 và Label Smoothing = 0.1
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), 
        loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
        metrics=['accuracy']
    )

    checkpoint_dir = os.path.join(config.PROJECT_ROOT, 'Cloud_server', 'Trainer', 'runs', 'gru_checkpoints')
    os.makedirs(checkpoint_dir, exist_ok=True)
    last_model_file = os.path.join(checkpoint_dir, 'gru_last.h5')
    last_epoch_file = os.path.join(checkpoint_dir, 'last_epoch.txt')
    initial_epoch = 0

    if os.path.exists(last_model_file) and os.path.exists(last_epoch_file):
        print("[*] Tìm thấy Checkpoint cũ. Đang khôi phục quá trình huấn luyện...")
        model = models.load_model(last_model_file) 
        with open(last_epoch_file, 'r') as f:
            initial_epoch = int(f.read())
        print(f"[*] Đã khôi phục thành công! Tiếp tục từ Epoch {initial_epoch + 1}")

    # CHỐT CHẶN 1: TỰ ĐỘNG DỪNG SỚM CHỐNG OVERFITTING (PATIENCE = 30 EPOCHS)
    early_stopping_gate = EarlyStopping(
        monitor='val_accuracy',
        patience=30, # [GIẢI PHÁP 5] Tăng lên 30 để model có thời gian học dữ liệu Augment
        restore_best_weights=True,
        verbose=1
    )

    # Tự động giảm ga LR nếu val_loss dậm chân tại chỗ tận 8 Epoch (Tránh hoảng loạn hạ ga sớm)
    lr_reducer = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=8, 
        min_lr=0.00001,
        verbose=1
    )

    # BỘ LƯU TRỌNG SỐ ĐỊNH KỲ VÀ LIÊN TỤC
    checkpoint_path = os.path.join(checkpoint_dir, 'checkpoint_epoch_{epoch:02d}.h5')
    
    # BỘ LƯU TRỌNG SỐ ĐỊNH KỲ VÀ GHI TIẾN TRÌNH TỰ VIẾT (VÁ LỖI KERAS)
    class SmartProgressCallback(tf.keras.callbacks.Callback):
        def __init__(self, last_file, epoch_file, checkpoint_fmt, period=5):
            super().__init__()
            self.last_file = last_file
            self.epoch_file = epoch_file
            self.checkpoint_fmt = checkpoint_fmt
            self.period = period

        def on_epoch_end(self, epoch, logs=None):
            current_epoch = epoch + 1
            
            # 1. Luôn ghi nhận số Epoch hiện tại vào file txt để khôi phục (Resume)
            with open(self.epoch_file, 'w') as f:
                f.write(str(current_epoch))
            
            # 2. Thay thế hoàn toàn lệnh period=5 cũ bằng toán tử chia lấy dư chuẩn xác
            if current_epoch % self.period == 0:
                epoch_path = self.checkpoint_fmt.format(epoch=current_epoch)
                self.model.save(epoch_path)
                print(f"\n💾 [CHECKPOINT] Đã lưu mô hình định kỳ tại Epoch {current_epoch} -> {epoch_path}")

    # Khởi tạo bộ gác cổng thông minh mới thay thế cho 2 callback (công cụ hỗ trợ của ML) cũ , Không lưu quá thường xuyên, chỉ theo chu kỳ (ở đây là 5 epoch) 
    progress_manager_gate = SmartProgressCallback(
        last_file=last_model_file,
        epoch_file=last_epoch_file,
        checkpoint_fmt=checkpoint_path,
        period=5 # Kích hoạt lưu mỗi 5 epoch cực kỳ an toàn
    )
    
    # Bộ lưu đè liên tục phục vụ tính năng Resume , Đảm bảo an toàn: Luôn có bản sao mô hình tại mỗi epoch , Khôi phục dễ dàng: Nếu huấn luyện bị gián đoạn, bạn có thể load lại mô hình từ checkpoint gần nhất ,  Linh hoạt: Tuỳ chỉnh để chỉ lưu mô hình tốt nhất (save_best_only=True) hoặc lưu tất cả (False)
    last_checkpoint_gate = ModelCheckpoint(
        filepath=last_model_file,
        save_best_only=False,
        verbose=0
    )

    # Huấn luyện (Có nạp bộ gác cổng callbacks)
    history = model.fit(
        X_train, y_train_onehot,
        validation_data=(X_val, y_val_onehot),
        epochs=200, # [GIẢI PHÁP 5] Tăng lên 200 epochs
        initial_epoch=initial_epoch,
        batch_size=16, # [GIẢI PHÁP 5] Giảm xuống 16 để gradient mượt hơn
        callbacks=[early_stopping_gate, lr_reducer, last_checkpoint_gate, progress_manager_gate] # Đã làm sạch đường ống
    )

    # Đọc chỉ số chính xác cao nhất đạt được trên tập Validation từ bộ lịch sử train
    best_val_acc = max(history.history['val_accuracy'])
    print(f"\n[AI GRU] Độ chính xác cao nhất mô hình đạt được (val_accuracy): {best_val_acc * 100:.2f}%")

    # Dọn dẹp rác tiến trình sau khi train xong toàn bộ file dự phòng xóa đi hết cho nhẹ nhàng, tránh chiếm dung lượng ổ cứng
    import glob
    print("\n--- Đang dọn dẹp các file checkpoint tạm thời ---")
    if os.path.exists(last_model_file): os.remove(last_model_file)
    if os.path.exists(last_epoch_file): os.remove(last_epoch_file)
    for f in glob.glob(os.path.join(checkpoint_dir, 'checkpoint_epoch_*.h5')):
        os.remove(f)

    # 🛠️ CHỐT CHẶN 2: KIỂM TRA NGƯỠNG CHẤT LƯỢNG TIÊU CHUẨN ĐẦU RA >= 90%
    if best_val_acc >= 0.90:
        model_info = {
            'classes': classes,
            'seq_len': seq_len,
            'input_dim': input_dim
        }
        
        # KỊCH BẢN ĐẠT CHUẨN: Lưu model và metadata phục vụ đồng bộ độc lập
        model_path = os.path.join(config.SHARED_ASSETS_DIR, 'action_recognizer.h5')
        model.save(model_path)
        
        # Lưu file từ điển mã hóa (.pkl) vào kho dùng chung
        info_path = os.path.join(config.SHARED_ASSETS_DIR, 'action_recognizer_info.pkl')
        with open(info_path, 'wb') as f:
            pickle.dump(model_info, f)
        
        print(f"✅ [GRU SUCCESS] Kết quả đạt chuẩn! Model đã lưu tại: {model_path}")
        print(f"✅ Metadata lưu tại: {info_path}")
        return True # Trả về True báo hiệu cho file Retrain.py tổng
    else:
        # KỊCH BẢN THẤT BẠI: Dưới 90%, từ chối lưu file, giữ nguyên hệ thống cũ để bảo vệ người dùng
        print(f"❌ [GRU FAILED] Kết quả không đạt ngưỡng an toàn ({best_val_acc * 100:.2f}% < 90%). Hủy bỏ cập nhật.")
        return False # Trả về False báo hiệu lò GRU thất bại

if __name__ == "__main__":
    main()