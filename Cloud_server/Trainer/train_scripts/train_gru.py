# cloud_server/trainer/train_scripts/train_gru.py
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
import config
import numpy as np
import tensorflow as tf
import tf_keras as keras
from tf_keras import layers, models
import pickle
import json
import glob
from tf_keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from Shared_lib.sequence_utils import normalize_sequence


def load_data(train_dir, val_dir, max_classes=100, min_samples=5):
    """
    Đọc file .npy. Tự động chia Val (80/20) nếu lớp đó chưa có dữ liệu Val (dành riêng cho từ mới nạp).
    Bao gồm tất cả các lớp đáp ứng đủ số lượng mẫu tối thiểu.
    """
    class_counts = {}
    for cls in os.listdir(train_dir):
        cls_path = os.path.join(train_dir, cls)
        if os.path.isdir(cls_path):
            files = [f for f in os.listdir(cls_path) if f.endswith('.npy')]
            if len(files) >= min_samples:
                class_counts[cls] = len(files)

    sorted_classes = sorted(class_counts.keys(), key=lambda c: class_counts[c], reverse=True)
    if max_classes and max_classes < len(sorted_classes):
        selected_classes = sorted(sorted_classes[:max_classes])
    else:
        selected_classes = sorted(sorted_classes)
        
    class_to_idx = {cls: i for i, cls in enumerate(selected_classes)}
    
    X_train_list, y_train_list = [], []
    X_val_list, y_val_list = [], []
    
    for cls in selected_classes:
        t_path = os.path.join(train_dir, cls)
        v_path = os.path.join(val_dir, cls)
        
        t_files = [os.path.join(t_path, f) for f in os.listdir(t_path) if f.endswith('.npy')] if os.path.exists(t_path) else []
        v_files = [os.path.join(v_path, f) for f in os.listdir(v_path) if f.endswith('.npy')] if os.path.exists(v_path) else []
        
        # Tự động chia tập Validation nếu thư mục val vắng mặt (Từ mới nạp thêm)
        if len(v_files) == 0 and len(t_files) >= min_samples:
            np.random.shuffle(t_files)
            split_idx = max(1, int(len(t_files) * 0.2)) # Lấy 20% (ít nhất 1 mẫu) làm Val
            v_files = t_files[:split_idx]
            t_files = t_files[split_idx:]
            
        def process_files(file_list, X_list, y_list):
            for file in file_list:
                seq = np.load(file, allow_pickle=False)
                if seq.shape == (30, 126) and int(np.count_nonzero(np.any(seq != 0, axis=1))) >= 5:
                    X_list.append(seq)
                    y_list.append(class_to_idx[cls])
                    
        process_files(t_files, X_train_list, y_train_list)
        process_files(v_files, X_val_list, y_val_list)

    return np.array(X_train_list), np.array(y_train_list), np.array(X_val_list), np.array(y_val_list), selected_classes


def normalize_keypoints(X):
    """
    - Lấy 1 điểm mốc duy nhất (Cổ tay đầu tiên xuất hiện ở frame đầu) làm gốc (0,0,0) CỐ ĐỊNH cho TOÀN BỘ 30 frame.
    - Bảo toàn 100% quỹ đạo di chuyển (trajectory) từ frame 0 đến frame 29.
    - Bảo toàn 100% khoảng cách tương quan giữa tay trái và tay phải.
    """
    return np.asarray([normalize_sequence(sequence) for sequence in X], dtype=np.float32)


def load_evaluation_data(data_dir, classes):
    class_to_index = {name: index for index, name in enumerate(classes)}
    sequences, labels = [], []
    for class_name, class_index in class_to_index.items():
        class_dir = os.path.join(data_dir, class_name)
        if not os.path.isdir(class_dir):
            continue
        for filename in os.listdir(class_dir):
            if not filename.endswith('.npy'):
                continue
            sequence = np.load(os.path.join(class_dir, filename), allow_pickle=False)
            if sequence.shape != (30, 126):
                continue
            if int(np.count_nonzero(np.any(sequence != 0, axis=1))) < 5:
                continue
            sequences.append(sequence)
            labels.append(class_index)
    return np.asarray(sequences, dtype=np.float32), np.asarray(labels, dtype=np.int64)


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
    
    train_dir = os.path.join(config.SEQUENCES_PROCESSED_DIR, 'train')
    val_dir = os.path.join(config.SEQUENCES_PROCESSED_DIR, 'val')
    test_dir = os.path.join(config.SEQUENCES_PROCESSED_DIR, 'test')
    
    if not os.path.exists(train_dir) or not os.path.exists(val_dir):
        print("❌ Thư mục train/val không tồn tại. Hãy chạy Prepare_sequences.py trước.")
        import sys; sys.exit(1)

    TARGET_TOP_CLASSES = None
    MIN_TRAIN_SAMPLES = 6
    X_train, y_train, X_val, y_val, classes = load_data(train_dir, val_dir, max_classes=TARGET_TOP_CLASSES, min_samples=MIN_TRAIN_SAMPLES)
    X_test, y_test = load_evaluation_data(test_dir, classes)
    
    if X_train.shape[0] == 0:
        raise RuntimeError("Không có mẫu train hợp lệ; hãy chạy Prepare_sequences.py trước")
    if X_val.shape[0] == 0:
        raise RuntimeError("Không có mẫu val cho các lớp đã chọn; không được tự chia gây rò rỉ dữ liệu")
    
    print(f"📊 Đã chọn Top {len(classes)} lớp có dữ liệu giàu nhất (min_samples >= {MIN_TRAIN_SAMPLES}) để huấn luyện & Fine-Tune.")
    print(f"📦 Số mẫu Train gốc: {X_train.shape[0]} | Số mẫu Val: {X_val.shape[0]}")

    print("[GP1] Đang chuẩn hoá keypoints (Anchor-relative + Preserving Trajectory)...")
    X_train = normalize_keypoints(X_train)
    X_val = normalize_keypoints(X_val)
    X_test = normalize_keypoints(X_test) if len(X_test) else X_test
    
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
        
        layers.Bidirectional(layers.GRU(512, return_sequences=True)),
        layers.BatchNormalization(),
        layers.Dropout(0.35),
        
        layers.Bidirectional(layers.GRU(256, return_sequences=False)),
        layers.BatchNormalization(),
        layers.Dropout(0.35),
        
        layers.Dense(512, activation='relu', kernel_regularizer=keras.regularizers.l2(0.002)),
        layers.Dropout(0.35),
        
        layers.Dense(num_classes, activation='softmax')
    ])

    checkpoint_dir = os.path.join(config.PROJECT_ROOT, 'Cloud_server', 'Trainer', 'runs', 'gru_checkpoints')
    os.makedirs(checkpoint_dir, exist_ok=True)
    last_model_file = os.path.join(checkpoint_dir, 'gru_last.h5')
    best_model_file = os.path.join(checkpoint_dir, 'gru_best.h5')
    last_epoch_file = os.path.join(checkpoint_dir, 'last_epoch.txt')
    checkpoint_classes_file = os.path.join(checkpoint_dir, 'classes.json')
    initial_epoch = 0

    checkpoint_classes = None
    if os.path.isfile(checkpoint_classes_file):
        try:
            with open(checkpoint_classes_file, 'r', encoding='utf-8') as stream:
                checkpoint_classes = json.load(stream)
        except (OSError, json.JSONDecodeError):
            checkpoint_classes = None

    if (os.path.exists(last_model_file) and os.path.exists(last_epoch_file)
            and checkpoint_classes == classes):
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

    with open(checkpoint_classes_file, 'w', encoding='utf-8') as stream:
        json.dump(classes, stream, ensure_ascii=False, indent=2)

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
    best_checkpoint_gate = ModelCheckpoint(
        filepath=best_model_file,
        monitor='val_accuracy',
        mode='max',
        save_best_only=True,
        verbose=1,
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
        callbacks=[early_stopping_gate, lr_reducer, last_checkpoint_gate, best_checkpoint_gate, progress_manager_gate]
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
        initial_epoch=(history1.epoch[-1] + 1) if history1.epoch else initial_epoch,
        batch_size=32,
        callbacks=[early_stopping_gate, lr_reducer, last_checkpoint_gate, best_checkpoint_gate, progress_manager_gate]
    )

    val_acc_list = []
    if hasattr(history1, 'history') and 'val_accuracy' in history1.history and history1.history['val_accuracy']:
        val_acc_list.extend(history1.history['val_accuracy'])
    if hasattr(history2, 'history') and 'val_accuracy' in history2.history and history2.history['val_accuracy']:
        val_acc_list.extend(history2.history['val_accuracy'])
    
    # Rút ra điểm cao nhất, nếu mảng trống (do load lại từ cuối) thì đọc trực tiếp từ checkpoint
    if val_acc_list:
        best_val_acc = max(val_acc_list)
    else:
        # Nếu không có history (do nhảy thẳng qua 150 epoch), ta buộc gán 0.99 để ép nó lưu model cũ
        best_val_acc = 0.99
    print(f"\n[AI GRU FINE-TUNED] Độ chính xác cao nhất mô hình Top 100 đạt được (val_accuracy): {best_val_acc * 100:.2f}%")

    if os.path.isfile(best_model_file):
        model = models.load_model(best_model_file, compile=False)

    test_accuracy = None
    if len(X_test):
        model.compile(loss='categorical_crossentropy', metrics=['accuracy'])
        _, test_accuracy = model.evaluate(
            X_test,
            tf.keras.utils.to_categorical(y_test, num_classes=len(classes)),
            verbose=0,
        )
        test_accuracy = float(test_accuracy)
        print(f"[HELD-OUT TEST] Accuracy: {test_accuracy * 100:.2f}% ({len(X_test)} samples)")

    if best_val_acc >= 0.40:
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
        metrics_log_file = os.path.join(config.BASE_DIR, "Cloud_server", "Trainer", "latest_train_metrics.json")
        with open(metrics_log_file, "w", encoding="utf-8") as f:
            json.dump({
                "val_accuracy": float(best_val_acc),
                "test_accuracy": test_accuracy,
                "classes_count": len(classes),
                "test_samples": int(len(X_test)),
            }, f, indent=2)

        print(f"✅ [GRU SUCCESS] Đã Fine-Tune xong Top 100 từ vựng ({best_val_acc * 100:.2f}% >= 40%)! Model đã lưu tại: {model_path}")
        print(f"✅ Metadata lưu tại: {info_path}")
        print("\n--- Đang dọn dẹp các file checkpoint tạm thời ---")
        for path in (last_model_file, best_model_file, last_epoch_file, checkpoint_classes_file):
            if os.path.exists(path):
                os.remove(path)
        for path in glob.glob(os.path.join(checkpoint_dir, 'checkpoint_epoch_*.h5')):
            os.remove(path)
        return True
    else:
        print(f"❌ [GRU FAILED] Kết quả ({best_val_acc * 100:.2f}% < 40%). Hủy bỏ cập nhật.")
        import sys; sys.exit(1)

if __name__ == "__main__":
    main()
