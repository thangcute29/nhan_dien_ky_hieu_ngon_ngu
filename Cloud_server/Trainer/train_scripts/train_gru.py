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
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint


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
    
    print(f"Train: {X_train.shape}, Val: {X_val.shape}")
    print(f"Số lớp: {len(classes)}")

    # Chuyển nhãn sang one-hot
    y_train_onehot = tf.keras.utils.to_categorical(y_train, num_classes=len(classes))
    y_val_onehot = tf.keras.utils.to_categorical(y_val, num_classes=len(classes))

    # Tham số
    seq_len = X_train.shape[1]   # 30 frame
    input_dim = X_train.shape[2] # 63 (21 điểm * 3 tọa độ)
    num_classes = len(classes)

    # Xây dựng model GRU
    model = models.Sequential([
        layers.Masking(mask_value=0.0, input_shape=(seq_len, input_dim)),
        layers.GRU(128, return_sequences=True),
        layers.GRU(64),
        layers.Dropout(0.3),
        layers.Dense(64, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ])

    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
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

    # CHỐT CHẶN 1: TỰ ĐỘNG DỪNG SỚM CHỐNG OVERFITTING (PATIENCE = 5 EPOCHS)
    # Nếu qua 5 lần liên tiếp chỉ số chính xác trên tập kiểm thử (val_accuracy) không tăng,
    # mô hình sẽ tự động dừng ngay lập tức và giữ lại trọng số ở thời điểm đỉnh cao nhất.
    early_stopping_gate = EarlyStopping(
        monitor='val_accuracy',
        patience=5,
        restore_best_weights=True,
        verbose=1
    )

    # BỘ LƯU TRỌNG SỐ ĐỊNH KỲ VÀ LIÊN TỤC
    checkpoint_path = os.path.join(checkpoint_dir, 'checkpoint_epoch_{epoch:02d}.h5')
    
    save_period_gate = ModelCheckpoint(
        filepath=checkpoint_path,
        save_freq='epoch',
        period=5,
        save_weights_only=False,
        verbose=1
    )
    
    last_checkpoint_gate = ModelCheckpoint(
        filepath=last_model_file,
        save_best_only=False,
        verbose=0
    )

    class SaveEpochCallback(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            with open(last_epoch_file, 'w') as f:
                f.write(str(epoch + 1))
                
    epoch_saver = SaveEpochCallback()

    # Huấn luyện (Có nạp bộ gác cổng callbacks)
    history = model.fit(
        X_train, y_train_onehot,
        validation_data=(X_val, y_val_onehot),
        epochs=50,
        initial_epoch=initial_epoch,
        batch_size=32,
        callbacks=[early_stopping_gate, save_period_gate, last_checkpoint_gate, epoch_saver] # ĐƯA BỘ PHANH THÔNG MINH VÀO ĐÂY
    )

    # Đọc chỉ số chính xác cao nhất đạt được trên tập Validation từ bộ lịch sử train
    best_val_acc = max(history.history['val_accuracy'])
    print(f"\n[AI GRU] Độ chính xác cao nhất mô hình đạt được (val_accuracy): {best_val_acc * 100:.2f}%")

    # Dọn dẹp rác tiến trình
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