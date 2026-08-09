import os
import shutil
import pandas as pd
from sklearn.model_selection import train_test_split
import sys

# Đưa thư mục chứa file hiện tại và thư mục gốc vào đường dẫn hệ thống để import config
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import config

def main():
    print("=== Chuẩn bị dữ liệu Classification (Multi-Label) ===")
    
    # Tạo các thư mục cần thiết cho Classification
    os.makedirs(config.CLASSIFICATION_DIR, exist_ok=True)
    os.makedirs(config.CLASSIFICATION_IMAGES_DIR, exist_ok=True)
    
    # KỊCH BẢN A: Đã có sẵn tập Classification train.csv và val.csv chuẩn
    if os.path.exists(config.CLASSIFICATION_TRAIN_CSV) and os.path.exists(config.CLASSIFICATION_VAL_CSV):
        train_df_exist = pd.read_csv(config.CLASSIFICATION_TRAIN_CSV)
        val_df_exist = pd.read_csv(config.CLASSIFICATION_VAL_CSV)
        print(f"🟢 [ADAPTER] Đã phát hiện dữ liệu Classification có sẵn! Train: {len(train_df_exist)} dòng | Val: {len(val_df_exist)} dòng.")
        print("✅ DỮ LIỆU CLASSIFICATION ĐÃ SẴN SÀNG TRAIN!")
        return

    # KỊCH BẢN B: Tự động quét bất kỳ file CSV mới nào được nạp vào thư mục Classification
    import glob
    csv_candidates = glob.glob(os.path.join(config.CLASSIFICATION_DIR, "*.csv"))
    csv_candidates = [c for c in csv_candidates if not c.endswith("train.csv") and not c.endswith("val.csv")]

    if not csv_candidates:
        print(f"❌ Không tìm thấy file nhãn CSV mới nào trong thư mục: {config.CLASSIFICATION_DIR}")
        return

    source_csv = csv_candidates[0]
    print(f"📂 [ADAPTER] Đang tự động quét và phân tích file nhãn mới: {os.path.basename(source_csv)}...")
    df = pd.read_csv(source_csv)
    print(f"Tổng số dòng ban đầu trong CSV: {len(df)}")
    
    # Loại bỏ dòng trùng lặp dựa trên imageName để đảm bảo sạch dữ liệu
    df_unique = df.drop_duplicates(subset=['imageName'])
    print(f"Số ảnh duy nhất sau khi lọc trùng: {len(df_unique)}")
    
    # 2. Lọc ra những ảnh thực sự tồn tại trong ổ cứng
    print("\nĐang kiểm tra sự tồn tại của các file ảnh gốc...")
    valid_rows = []
    for _, row in df_unique.iterrows():
        img_path = os.path.join(config.DETECTION_IMAGES_DIR, row['imageName'])
        if os.path.exists(img_path):
            valid_rows.append(row)
            
    df_valid = pd.DataFrame(valid_rows)
    print(f"Số ảnh hợp lệ (tồn tại trên ổ cứng): {len(df_valid)}")
    
    if len(df_valid) == 0:
        print("Không có ảnh hợp lệ. Dừng quá trình.")
        return

    # 3. Chia train/val (80/20) dựa trên ID người dùng 
    # (Việc chia theo ID giúp tránh rò rỉ dữ liệu - Data Leakage giữa train và test)
    unique_ids = df_valid['id'].unique()
    train_ids, val_ids = train_test_split(unique_ids, train_size=0.8, random_state=42)
    
    train_df = df_valid[df_valid['id'].isin(train_ids)]
    val_df = df_valid[df_valid['id'].isin(val_ids)]
    
    print(f"\nPhân chia dữ liệu:")
    print(f" - Train set: {len(train_df)} ảnh")
    print(f" - Val set: {len(val_df)} ảnh")
    
    # 4. Lưu ra 2 file CSV (chứa TOÀN BỘ các nhãn phong phú: tuổi, giới tính, hướng tay, màu da...)
    train_df.to_csv(config.CLASSIFICATION_TRAIN_CSV, index=False)
    val_df.to_csv(config.CLASSIFICATION_VAL_CSV, index=False)
    print(f"\nĐã lưu file: {config.CLASSIFICATION_TRAIN_CSV}")
    print(f"Đã lưu file: {config.CLASSIFICATION_VAL_CSV}")
    
    # 5. Copy ảnh sang thư mục Classification/images (Gom chung một chỗ)
    print("\nĐang copy ảnh sang thư mục Classification... (Vui lòng đợi)")
    total_imgs = len(df_valid)
    count = 0
    copied = 0
    
    for img_name in df_valid['imageName'].tolist():
        count += 1
        if count % 1000 == 0:
            print(f"  Đã kiểm tra/copy {count}/{total_imgs} ảnh...")
            
        src = os.path.join(config.DETECTION_IMAGES_DIR, img_name)
        dst = os.path.join(config.CLASSIFICATION_IMAGES_DIR, img_name)
        
        # Chỉ copy nếu file đích chưa tồn tại (giúp chạy lại script nhanh hơn nếu bị ngắt quãng)
        if not os.path.exists(dst):
            shutil.copy2(src, dst)
            copied += 1
            
    print(f"  Đã copy mới thêm {copied} ảnh.")
            
    print("\n✅ Hoàn tất chuẩn bị dữ liệu Classification Đa nhãn (Multi-Label)!")
    print("-> Tất cả ảnh đã quy về một mối.")
    print("-> Nhãn được quản lý tập trung và an toàn bằng train.csv và val.csv.")

if __name__ == "__main__":
    main()
