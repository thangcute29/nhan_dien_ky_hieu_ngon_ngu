# Data_preparation/prepare_features.py
import os
import pandas as pd
import shutil
from sklearn.model_selection import train_test_split
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import config

def copy_images_from_dir(src_dir, dst_dir, class_name):
    """Copy tất cả ảnh từ src_dir vào dst_dir/class_name."""
    if not os.path.exists(src_dir):
        return 0
    dst_class_dir = os.path.join(dst_dir, class_name)
    os.makedirs(dst_class_dir, exist_ok=True)
    count = 0
    for f in os.listdir(src_dir):
        src_file = os.path.join(src_dir, f)
        if os.path.isfile(src_file) and f.lower().endswith(('.jpg', '.jpeg', '.png')):
            shutil.copy2(src_file, dst_class_dir)
            count += 1
    return count

def main():
    print("=== Chuẩn bị dữ liệu Features (Ảnh ký hiệu tĩnh) ===")
    
    source_dir = config.FEATURES_RAW_DIR
    train_output = os.path.join(config.FEATURES_DIR, 'train')
    val_output = os.path.join(config.FEATURES_DIR, 'val')

    # KỊCH BẢN A: Đã có sẵn folder train/ và val/ chuẩn trong Features
    if os.path.exists(train_output) and os.path.exists(val_output) and train_output != source_dir:
        train_classes = [d for d in os.listdir(train_output) if os.path.isdir(os.path.join(train_output, d))]
        if len(train_classes) > 0:
            print(f"🟢 [ADAPTER] Phát hiện dữ liệu Features train/val đã sẵn sàng ({len(train_classes)} lớp)!")
            print("✅ DỮ LIỆU FEATURES ĐÃ SẴN SÀNG TRAIN!")
            return

    # KỊCH BẢN B: Nếu chưa chia, kiểm tra dọn dẹp và chia 80/20
    if not os.path.exists(source_dir):
        print(f"❌ Không tìm thấy thư mục ảnh nguồn tại: {source_dir}")
        return

    # 2. Kiểm tra xem có sẵn cấu trúc train/val trong source_dir không
    if os.path.exists(os.path.join(source_dir, 'train')) and os.path.exists(os.path.join(source_dir, 'val')):
        print("✅ Phát hiện cấu trúc train/val có sẵn. Sẽ copy trực tiếp...")
        src_train = os.path.join(source_dir, 'train')
        src_val = os.path.join(source_dir, 'val')
        
        for class_name in os.listdir(src_train):
            class_path = os.path.join(src_train, class_name)
            if os.path.isdir(class_path):
                copy_images_from_dir(class_path, train_output, class_name)
                
        for class_name in os.listdir(src_val):
            class_path = os.path.join(src_val, class_name)
            if os.path.isdir(class_path):
                copy_images_from_dir(class_path, val_output, class_name)
                
    else:
        print("ℹ️ Không tìm thấy cấu trúc train/val có sẵn. Tiến hành tự chia 80/20...")
        # Tìm tất cả các lớp (thư mục con) trong source_dir
        classes = [d for d in os.listdir(source_dir) if os.path.isdir(os.path.join(source_dir, d))]
        
        for class_name in classes:
            class_path = os.path.join(source_dir, class_name)
            images = [f for f in os.listdir(class_path) if os.path.isfile(os.path.join(class_path, f)) and f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            
            if len(images) == 0:
                continue
                
            # Chia 80/20
            train_imgs, val_imgs = train_test_split(images, test_size=0.2, random_state=42)
            
            # Copy ảnh train
            for img in train_imgs:
                src = os.path.join(class_path, img)
                dst_dir = os.path.join(train_output, class_name)
                os.makedirs(dst_dir, exist_ok=True)
                shutil.copy2(src, dst_dir)
                
            # Copy ảnh val
            for img in val_imgs:
                src = os.path.join(class_path, img)
                dst_dir = os.path.join(val_output, class_name)
                os.makedirs(dst_dir, exist_ok=True)
                shutil.copy2(src, dst_dir)
                
            print(f"   Đã xử lý lớp '{class_name}': {len(train_imgs)} train, {len(val_imgs)} val")

    print("✅ Hoàn tất chuẩn bị dữ liệu Features.")
    print(f"   - Train: {train_output}")
    print(f"   - Val: {val_output}")

if __name__ == "__main__":
    main()