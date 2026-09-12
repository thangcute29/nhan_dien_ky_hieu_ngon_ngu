import os
import tensorflow as tf
import tf_keras
from tf_keras import layers, models, applications
from tf_keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import sys
import shutil

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)
import config
dataset_dir = os.path.join(BASE_DIR, "Research_and_Data", "Dataset", "Features")

# Khu vực nháp chạy MLOps
RUNS_DIR = os.path.join(BASE_DIR, "Cloud_server", "Trainer", "runs", "cnn_checkpoints")
os.makedirs(RUNS_DIR, exist_ok=True)

def main():
    print("=== HUẤN LUYỆN CNN (TỪ DATASET ẢNH VẼ BỘ XƯƠNG) ===")
    IMG_SIZE = (128, 128)
    BATCH_SIZE = 32

    # Vẫn dùng tf.keras.preprocessing để đọc data (đây là API load data độc lập)
    train_dataset = tf.keras.preprocessing.image_dataset_from_directory(
        dataset_dir, validation_split=0.2, subset="training", seed=123,
        image_size=IMG_SIZE, batch_size=BATCH_SIZE
    )
    val_dataset = tf.keras.preprocessing.image_dataset_from_directory(
        dataset_dir, validation_split=0.2, subset="validation", seed=123,
        image_size=IMG_SIZE, batch_size=BATCH_SIZE
    )

    class_names = train_dataset.class_names
    num_classes = len(class_names)
    print(f"✅ Đã tìm thấy {num_classes} lớp.")

    AUTOTUNE = tf.data.AUTOTUNE
    train_dataset = train_dataset.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    val_dataset = val_dataset.cache().prefetch(buffer_size=AUTOTUNE)

    # ĐẢM BẢO TẤT CẢ DÙNG tf_keras ĐỂ TRÁNH LỖI XUNG ĐỘT PHIÊN BẢN (Keras 2 vs Keras 3)
    base_model = applications.MobileNetV2(input_shape=IMG_SIZE + (3,), include_top=False, weights='imagenet')
    base_model.trainable = False

    inputs = tf_keras.Input(shape=IMG_SIZE + (3,))
    x = applications.mobilenet_v2.preprocess_input(inputs)
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    model = models.Model(inputs, outputs)
    
    model.compile(optimizer=tf_keras.optimizers.Adam(learning_rate=0.001),
                  loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    best_model_path = os.path.join(RUNS_DIR, "best_cnn.h5")
    
    callbacks = [
        ModelCheckpoint(best_model_path, monitor='val_accuracy', save_best_only=True, verbose=1),
        EarlyStopping(monitor='val_accuracy', patience=15, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1)
    ]

    history = model.fit(train_dataset, validation_data=val_dataset, epochs=50, callbacks=callbacks)
    
    best_val_acc = max(history.history['val_accuracy'])
    if best_val_acc < 0.90:
        print(f"\n❌ [THẤT BẠI CHỐT CHẶN] Độ chính xác cao nhất chỉ đạt {best_val_acc*100:.2f}%. KHÔNG ĐẠT NGƯỠNG 90%!")
        if os.path.exists(best_model_path):
            os.remove(best_model_path)
    else:
        production_path = os.path.join(config.SHARED_ASSETS_DIR, "feature_extractor.h5")
        os.makedirs(config.SHARED_ASSETS_DIR, exist_ok=True)
        shutil.copy2(best_model_path, production_path)
        print(f"\n🎉 HOÀN TẤT! Mô hình CNN đã vượt chuẩn và được xuất tại: {production_path}")

if __name__ == "__main__":
    main()
