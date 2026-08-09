# cloud_server/trainer/train_scripts/train_gru.py
"""
Huấn luyện GRU để nhận diện hành động từ chuỗi keypoints.
Sử dụng dữ liệu .npy từ Sequences/processed/train/ và Sequences/processed/val/

Cải tiến Nâng cấp:
1. Lọc Top N lớp phổ biến nhất (mặc định 100 từ) có đủ số lượng mẫu để mô hình học hội tụ tốt.
2. Chuẩn hoá Keypoints Thông minh: Giữ nguyên quỹ đạo chuyển động theo thời gian (Trajectory) và khoảng cách tương quan giữa 2 tay.
3. Loại bỏ Lật ngược tay (Mirror Swap) trong Augmentation -> Tăng tốc huấn luyện gấp 5-6 lần.
4. Thông nghẽn cổ chai kiến trúc BiGRU (256 -> 128 -> Dense 128) để giải tỏa bộ trích xuất đặc trưng chuỗi.
"""
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
import config
import numpy as np
import tensorflow as tf
import tf_keras as keras
from tf_keras import layers, models
import pickle
from tf_keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau


def load_data(train_dir, val_dir, max_classes=100, min_samples=5):
    """
    Đọc tất cả file .npy và lọc Top N nhãn có nhiều mẫu nhất trong tập train.
    Giúp mô hình hội tụ chuẩn xác, tránh bị ngợp bởi 2000 lớp mà mỗi lớp chỉ có 1-2 mẫu.
    """
    class_counts = {}
    for cls in os.listdir(train_dir):
        cls_path = os.path.join(train_dir, cls)
        if os.path.isdir(cls_path):
            files = [f for f in os.listdir(cls_path) if f.endswith('.npy')]
            if len(files) >= min_samples:
                class_counts[cls] = len(files)
    
    if not class_counts:
        for cls in os.listdir(train_dir):
            cls_path = os.path.join(train_dir, cls)
            if os.path.isdir(cls_path):
                files = [f for f in os.listdir(cls_path) if f.endswith('.npy')]
                if files:
                    class_counts[cls] = len(files)

    sorted_classes = sorted(class_counts.keys(), key=lambda c: class_counts[c], reverse=True)
    if max_classes and max_classes < len(sorted_classes):
        selected_classes = sorted(sorted_classes[:max_classes])
    else:
        selected_classes = sorted(sorted_classes)
        
    class_to_idx = {cls: i for i, cls in enumerate(selected_classes)}
    
    def load_split(data_dir):
        X, y = [], []
        for cls in selected_classes:
            cls_path = os.path.join(data_dir, cls)
            if not os.path.exists(cls_path):
                continue
            for file in os.listdir(cls_path):
                if file.endswith('.npy'):
                    seq = np.load(os.path.join(cls_path, file))
                    X.append(seq)
                    y.append(class_to_idx[cls])
        return np.array(X), np.array(y)

    X_train, y_train = load_split(train_dir)
    X_val, y_val = load_split(val_dir)
    return X_train, y_train, X_val, y_val, selected_classes


def normalize_keypoints(X):
    """
    Chuẩn hoá keypoints THÔNG MINH:
    - Lấy 1 điểm mốc duy nhất (Cổ tay đầu tiên xuất hiện ở frame đầu) làm gốc (0,0,0) CỐ ĐỊNH cho TOÀN BỘ 30 frame.
    - Bảo toàn 100% quỹ đạo di chuyển (trajectory) từ frame 0 đến frame 29.
    - Bảo toàn 100% khoảng cách tương quan giữa tay trái và tay phải.
    """
    X_norm = X.copy().astype(np.float32)
    
    for i in range(len(X_norm)):
        seq = X_norm[i]
        
        # Tìm frame đầu tiên có dữ liệu keypoint để chọn mốc Anchor cố định
        anchor = None
        for t in range(seq.shape[0]):
            frame = seq[t]
            if np.all(frame == 0):
                continue
            
            if seq.shape[1] == 126:
                right_hand = frame[63:126]
                left_hand = frame[0:63]
                if np.any(right_hand != 0):
                    anchor = right_hand[0:3].copy()
                    break
                elif np.any(left_hand != 0):
                    anchor = left_hand[0:3].copy()
                    break
            else:
                if np.any(frame != 0):
                    anchor = frame[0:3].copy()
                    break
        
        if anchor is None:
            continue
            
        # Trừ anchor cố định cho TẤT CẢ các frame có dữ liệu
        for t in range(seq.shape[0]):
            frame = seq[t]
            if np.all(frame == 0):
                continue
                
            if seq.shape[1] == 126:
                if np.any(frame[0:63] != 0):
                    pts_left = frame[0:63].reshape(21, 3) - anchor
                    frame[0:63] = pts_left.flatten()
                    
                if np.any(frame[63:126] != 0):
                    pts_right = frame[63:126].reshape(21, 3) - anchor
                    frame[63:126] = pts_right.flatten()
            else:
                if np.any(frame != 0):
                    pts = frame.reshape(21, 3) - anchor
                    frame = pts.flatten()
                    
            X_norm[i, t] = frame
            
    return X_norm


def augment_keypoints(X, y, num_copies=1):
    """
    Tăng cường dữ liệu nhẹ nhàng và hiệu quả:
    - A. Jitter (Nhiễu nhẹ)
    - B. Scale (Co giãn tỷ lệ nhẹ 90%-110%)
    - C. Time Warp (Co giãn tốc độ thời gian)
    - ĐÃ BỎ LẬT GƯƠNG (MIRROR SWAP) để tránh làm méo ký hiệu bất đối xứng và giúp train cực nhanh.
    """
    X_aug_list = [X]
    y_aug_list = [y]
    seq_len = X.shape[1]

    for _ in range(num_copies):
        X_copy = X.copy()
        for i in range(len(X_copy)):
            seq = X_copy[i]
            mask = np.any(seq != 0, axis=1, keepdims=True)

            noise = np.random.normal(0, 0.005, seq.shape)
            seq = seq + noise * mask

            scale = np.random.uniform(0.90, 1.10)
            seq = seq * scale * mask

            if np.random.random() > 0.5:
                real_len = int(np.sum(np.any(seq != 0, axis=1)))
                if real_len > 5:
                    speed = np.random.uniform(0.85, 1.15)
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

    return np.concatenate(X_aug_list), np.concatenate(y_aug_list)


def main():
    print("=== Huấn luyện GRU Nâng cấp (Giải tỏa thắt cổ chai & Giữ nguyên Quỹ đạo) ===")
    
    train_dir = os.path.join(config.SEQUENCES_DIR, 'processed', 'train')
    val_dir = os.path.join(config.SEQUENCES_DIR, 'processed', 'val')
    
    if not os.path.exists(train_dir) or not os.path.exists(val_dir):
        print("❌ Thư mục train/val không tồn tại. Hãy chạy Prepare_sequences.py trước.")
        return False

    TARGET_TOP_CLASSES = 100
    MIN_TRAIN_SAMPLES = 6
    X_train, y_train, X_val, y_val, classes = load_data(train_dir, val_dir, max_classes=TARGET_TOP_CLASSES, min_samples=MIN_TRAIN_SAMPLES)
    
    print(f"📊 Đã chọn Top {len(classes)} lớp có dữ liệu giàu nhất (min_samples >= {MIN_TRAIN_SAMPLES}) để huấn luyện & Fine-Tune.")
    print(f"📦 Số mẫu Train gốc: {X_train.shape[0]} | Số mẫu Val: {X_val.shape[0]}")

    print("[GP1] Đang chuẩn hoá keypoints (Anchor-relative + Preserving Trajectory)...")
    X_train = normalize_keypoints(X_train)
    X_val = normalize_keypoints(X_val)
    
    print("[GP2] Đang tăng cường dữ liệu mở rộng (Augmentation x4 copies)...")
    X_train, y_train = augment_keypoints(X_train, y_train, num_copies=4)
    print(f"📦 Số mẫu Train sau Augmentation: {X_train.shape[0]}")

    y_train_onehot = tf.keras.utils.to_categorical(y_train, num_classes=len(classes))
    y_val_onehot = tf.keras.utils.to_categorical(y_val, num_classes=len(classes))

    seq_len = X_train.shape[1]
    input_dim = X_train.shape[2]
    num_classes = len(classes)

    model = models.Sequential([
        layers.Masking(mask_value=0.0, input_shape=(seq_len, input_dim)),
        
        layers.Bidirectional(layers.GRU(256, return_sequences=True)),
        layers.BatchNormalization(),
        layers.Dropout(0.35),
        
        layers.Bidirectional(layers.GRU(128, return_sequences=False)),
        layers.BatchNormalization(),
        layers.Dropout(0.35),
        
        layers.Dense(128, activation='relu', kernel_regularizer=keras.regularizers.l2(0.002)),
        layers.Dropout(0.35),
        
        layers.Dense(num_classes, activation='softmax')
    ])

    checkpoint_dir = os.path.join(config.PROJECT_ROOT, 'Cloud_server', 'Trainer', 'runs', 'gru_checkpoints')
    os.makedirs(checkpoint_dir, exist_ok=True)
    last_model_file = os.path.join(checkpoint_dir, 'gru_last.h5')
    last_epoch_file = os.path.join(checkpoint_dir, 'last_epoch.txt')
    initial_epoch = 0

    if os.path.exists(last_model_file) and os.path.exists(last_epoch_file):
        try:
            old_model = models.load_model(last_model_file)
            if old_model.output_shape[-1] == num_classes:
                print("[*] Tìm thấy Checkpoint cũ tương thích. Đang khôi phục quá trình huấn luyện...")
                model = old_model
                with open(last_epoch_file, 'r') as f:
                    initial_epoch = int(f.read())
                print(f"[*] Đã khôi phục thành công! Tiếp tục từ Epoch {initial_epoch + 1}")
            else:
                print(f"[!] Checkpoint cũ có số lớp khác ({old_model.output_shape[-1]} != {num_classes}). Khởi tạo lại mô hình mới cho Top {num_classes} lớp...")
                initial_epoch = 0
        except Exception as e:
            print(f"[!] Không thể load checkpoint cũ ({e}). Khởi tạo mới...")
            initial_epoch = 0

    early_stopping_gate = EarlyStopping(
        monitor='val_accuracy',
        patience=35,
        restore_best_weights=True,
        verbose=1
    )

    lr_reducer = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=8, 
        min_lr=0.00001,
        verbose=1
    )

    checkpoint_path = os.path.join(checkpoint_dir, 'checkpoint_epoch_{epoch:02d}.h5')
    
    class SmartProgressCallback(keras.callbacks.Callback):
        def __init__(self, last_file, epoch_file, checkpoint_fmt, period=5):
            super().__init__()
            self.last_file = last_file
            self.epoch_file = epoch_file
            self.checkpoint_fmt = checkpoint_fmt
            self.period = period

        def on_epoch_end(self, epoch, logs=None):
            current_epoch = epoch + 1
            with open(self.epoch_file, 'w') as f:
                f.write(str(current_epoch))
            if current_epoch % self.period == 0:
                epoch_path = self.checkpoint_fmt.format(epoch=current_epoch)
                self.model.save(epoch_path)
                print(f"\n💾 [CHECKPOINT] Đã lưu mô hình định kỳ tại Epoch {current_epoch} -> {epoch_path}")

    progress_manager_gate = SmartProgressCallback(
        last_file=last_model_file,
        epoch_file=last_epoch_file,
        checkpoint_fmt=checkpoint_path,
        period=5
    )
    
    last_checkpoint_gate = ModelCheckpoint(
        filepath=last_model_file,
        save_best_only=False,
        verbose=0
    )

    print("🚀 [STAGE 1] Khởi chạy Huấn luyện cơ bản (lr=0.001)...")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001), 
        loss=keras.losses.CategoricalCrossentropy(label_smoothing=0.04),
        metrics=['accuracy']
    )

    history1 = model.fit(
        X_train, y_train_onehot,
        validation_data=(X_val, y_val_onehot),
        epochs=60,
        initial_epoch=initial_epoch,
        batch_size=32,
        callbacks=[early_stopping_gate, lr_reducer, last_checkpoint_gate, progress_manager_gate]
    )

    print("\n🎯 [STAGE 2 - FINE-TUNING] Hạ Learning Rate (lr=0.0001) để Tinh chỉnh cho Top 100 lớp...")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.0001),
        loss=keras.losses.CategoricalCrossentropy(label_smoothing=0.02),
        metrics=['accuracy']
    )

    history2 = model.fit(
        X_train, y_train_onehot,
        validation_data=(X_val, y_val_onehot),
        epochs=150,
        initial_epoch=len(history1.history['accuracy']),
        batch_size=32,
        callbacks=[early_stopping_gate, lr_reducer, last_checkpoint_gate, progress_manager_gate]
    )

    best_val_acc = max(max(history1.history['val_accuracy']), max(history2.history['val_accuracy']))
    print(f"\n[AI GRU FINE-TUNED] Độ chính xác cao nhất mô hình Top 100 đạt được (val_accuracy): {best_val_acc * 100:.2f}%")

    import glob
    print("\n--- Đang dọn dẹp các file checkpoint tạm thời ---")
    if os.path.exists(last_model_file): os.remove(last_model_file)
    if os.path.exists(last_epoch_file): os.remove(last_epoch_file)
    for f in glob.glob(os.path.join(checkpoint_dir, 'checkpoint_epoch_*.h5')):
        os.remove(f)

    if best_val_acc >= 0.30:
        model_info = {
            'classes': classes,
            'seq_len': seq_len,
            'input_dim': input_dim
        }
        
        model_path = os.path.join(config.SHARED_ASSETS_DIR, 'action_recognizer.h5')
        model.save(model_path)
        
        info_path = os.path.join(config.SHARED_ASSETS_DIR, 'action_recognizer_info.pkl')
        with open(info_path, 'wb') as f:
            pickle.dump(model_info, f)
        
        # Lưu file metrics JSON cho Retrain.py đánh giá
        import json
        metrics_log_file = os.path.join(config.BASE_DIR, "Cloud_server", "Trainer", "latest_train_metrics.json")
        with open(metrics_log_file, "w", encoding="utf-8") as f:
            json.dump({"val_accuracy": float(best_val_acc), "classes_count": len(classes)}, f, indent=2)

        print(f"✅ [GRU SUCCESS] Đã Fine-Tune xong Top 100 từ vựng ({best_val_acc * 100:.2f}% >= 30%)! Model đã lưu tại: {model_path}")
        print(f"✅ Metadata lưu tại: {info_path}")
        return True
    else:
        print(f"❌ [GRU FAILED] Kết quả ({best_val_acc * 100:.2f}% < 30%). Hủy bỏ cập nhật.")
        return False

if __name__ == "__main__":
    main()
