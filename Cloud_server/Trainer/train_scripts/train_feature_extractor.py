# cloud_server/trainer/train_scripts/train_feature_extractor.py
"""
Huấn luyện EfficientNetB0 để phân loại ảnh tĩnh (chữ cái ASL).
Sử dụng dữ liệu từ Features/train/ và Features/val/
"""
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
import config
from Shared_lib.Constants import ASL_29_CLASSES
import tensorflow as tf
import tf_keras as keras
from tf_keras import layers, models
from tf_keras.applications import EfficientNetB0
from tf_keras.preprocessing.image import ImageDataGenerator
from tf_keras.callbacks import EarlyStopping, ModelCheckpoint
from tf_keras.applications.efficientnet import preprocess_input
from tf_keras.layers import Dense, GlobalAveragePooling2D
from tf_keras.models import Model
from tf_keras.optimizers import Adam

def main():
    print("=== Huấn luyện EfficientNet phân loại chữ cái ===")
    # Đường dẫn thư mục train/val
    train_dir = os.path.join(config.FEATURES_DIR, 'train')
    val_dir = os.path.join(config.FEATURES_DIR, 'val')
    
    print(f"-> Thư mục Train: {train_dir}")
    print(f"-> Thư mục Val: {val_dir}")

    # Nạp danh sách 29 chữ cái chuẩn ASL
    classes = ASL_29_CLASSES
    num_classes = len(classes)
    print(f"Số lớp bắt buộc: {num_classes} - {classes}")

    # Đảm bảo tạo thư mục train/val nếu chưa có đúng 29 lớp
    for cls in classes:
        os.makedirs(os.path.join(train_dir, cls), exist_ok=True)
        os.makedirs(os.path.join(val_dir, cls), exist_ok=True)

    if not os.path.exists(train_dir) or not os.path.exists(val_dir):
        print("❌ Không tìm thấy thư mục dữ liệu train hoặc val trên đĩa cứng!")
        return False
    
    IMG_SIZE = 224
    BATCH_SIZE = 32
    EPOCHS = 30

    # Data augmentation
    train_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
        horizontal_flip=False # Thủ ngữ không nên lật gương ảnh làm đổi chiều tay trái/phải
    )
    val_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='categorical'
    )
    val_generator = val_datagen.flow_from_directory(
        val_dir,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='categorical'
    )

    # Nạp toàn bộ ảnh từ thư mục val vật lý
    val_generator = val_datagen.flow_from_directory(
        val_dir,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='categorical'
    )

    num_classes = train_generator.num_classes
    if num_classes == 0:
        print("❌ Thất bại: Không tìm thấy nhãn chữ cái nào trong thư mục train!")
        return False

    print(f"✅ Đồng bộ thành công: Phát hiện {num_classes} nhãn ký tự.")

    # Khởi tạo mô hình nền tảng EfficientNetB0
    base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    base_model.trainable = True # Cho phép fine-tuning toàn bộ mô hình 
    
    # Xây dựng mô hình phân loại
    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.3),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation='softmax')
    ])

    # Ép tốc độ học chuẩn 0.001 theo đúng chiến thuật
    model.compile(
        optimizer=Adam(learning_rate=0.0001),#tuyệt đối không được bật base_model.trainable = True khi tốc độ học còn cao (learning_rate=0.001).
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    checkpoint_dir = os.path.join(config.PROJECT_ROOT, 'Cloud_server', 'Trainer', 'runs', 'feature_extractor_checkpoints')
    os.makedirs(checkpoint_dir, exist_ok=True)
    last_model_file = os.path.join(checkpoint_dir, 'feature_extractor_last.h5')
    last_epoch_file = os.path.join(checkpoint_dir, 'last_epoch.txt')
    initial_epoch = 0

    # --- KHÔI PHỤC TIẾN TRÌNH (RESUME) ---
    if os.path.exists(last_model_file) and os.path.exists(last_epoch_file):
        print("[*] Tìm thấy Checkpoint cũ. Đang khôi phục quá trình huấn luyện...")
        model = models.load_model(last_model_file) 
        with open(last_epoch_file, 'r') as f:
            initial_epoch = int(f.read())
        print(f"[*] Đã khôi phục thành công! Tiếp tục từ Epoch {initial_epoch + 1}")

    # Bộ phanh dừng sớm khi val_loss dậm chân 5 lần liên tiếp
    early_stopping_gate = EarlyStopping(
        monitor='val_accuracy',
        patience=5,
        restore_best_weights=True,
        verbose=1
    )

    # BỘ LƯU TRỌNG SỐ ĐỊNH KỲ VÀ LIÊN TỤC
    checkpoint_path = os.path.join(checkpoint_dir, 'checkpoint_epoch_{epoch:02d}.h5')
    
    # BỘ LƯU TRỌNG SỐ ĐỊNH KỲ VÀ GHI TIẾN TRÌNH TỰ VIẾT (VÁ LỖI KERAS)
    class SmartProgressCallback(keras.callbacks.Callback):
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

    # Khởi tạo bộ gác cổng thông minh mới thay thế cho 2 callback cũ
    progress_manager_gate = SmartProgressCallback(
        last_file=last_model_file,
        epoch_file=last_epoch_file,
        checkpoint_fmt=checkpoint_path,
        period=5 # Kích hoạt lưu mỗi 5 epoch cực kỳ an toàn
    )
    
    # Bộ lưu đè liên tục phục vụ tính năng Resume
    last_checkpoint_gate = ModelCheckpoint(
        filepath=last_model_file,
        save_best_only=False,
        verbose=0
    )

    print(f"-> Trạng thái lò: LR=0.001 | Patience=5 | Save Period=5 | Đang tiến hành fit...")
    
    history = model.fit(
        train_generator,
        epochs=EPOCHS,
        initial_epoch=initial_epoch,
        validation_data=val_generator,
        steps_per_epoch=len(train_generator),
        validation_steps=len(val_generator),
        callbacks=[early_stopping_gate, last_checkpoint_gate, progress_manager_gate], # Đã làm sạch đường ống
        verbose=1
    )

    # Đọc chỉ số chính xác cao nhất đạt được trên tập khảo thí thực tế
    best_val_acc = max(history.history['val_accuracy'])
    print(f"\n[AI Feature] Độ chính xác cao nhất mô hình đạt được (val_accuracy): {best_val_acc * 100:.2f}%")

    # Dọn dẹp rác tiến trình
    import glob
    print("\n--- Đang dọn dẹp các file checkpoint tạm thời ---")
    if os.path.exists(last_model_file): os.remove(last_model_file)
    if os.path.exists(last_epoch_file): os.remove(last_epoch_file)
    for f in glob.glob(os.path.join(checkpoint_dir, 'checkpoint_epoch_*.h5')):
        os.remove(f)

    # MÀNG LỌC GATE 90% ĐỂ XUẤT FILE ĐỒNG BỘ
    if best_val_acc >= 0.90:
        dest_path = os.path.join(config.SHARED_ASSETS_DIR, 'feature_extractor.h5')
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
        model.save(dest_path)
        print(f"✅ Kết quả đạt chuẩn chất lượng! Bộ trích xuất đã lưu tại: {dest_path}")
        return True
    else:
        print(f"❌ Kết quả huấn luyện THẤT BẠI ({best_val_acc * 100:.2f}% < 90%). Hủy bỏ cập nhật.")
        return False
    
if __name__ == "__main__":
    main()