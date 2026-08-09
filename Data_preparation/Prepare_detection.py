# SIGN_LANGUAGE_MARKET_READY/data_preparation/prepare_detection.py
import os
import cv2
import pandas as pd
from sklearn.model_selection import train_test_split
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import config

try:
    from mediapipe.python.solutions import hands as mp_hands
except ImportError:
    print("Đang cài đặt thư viện MediaPipe để tạo nhãn tự động...")
    os.system("pip install mediapipe opencv-python")
    from mediapipe.python.solutions import hands as mp_hands

def get_hand_bounding_box(image_path, hands_detector):
    """Sử dụng MediaPipe để tìm khung bao quanh bàn tay trong ảnh."""
    img = cv2.imread(image_path)
    if img is None:
        return None
    
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands_detector.process(img_rgb)
    
    if not results.multi_hand_landmarks:
        return None
    
    h, w, _ = img.shape
    x_min, y_min = w, h
    x_max, y_max = 0, 0
    
    # Lấy tọa độ nhỏ nhất và lớn nhất của các khớp tay
    for hand_landmarks in results.multi_hand_landmarks:
        for landmark in hand_landmarks.landmark:
            x, y = int(landmark.x * w), int(landmark.y * h)
            if x < x_min: x_min = x
            if y < y_min: y_min = y
            if x > x_max: x_max = x
            if y > y_max: y_max = y
            
    # Thêm một chút lề (padding) 5% để khung không bị quá sát tay
    pad_x = int(w * 0.05)
    pad_y = int(h * 0.05)
    
    x_min = max(0, x_min - pad_x)
    y_min = max(0, y_min - pad_y)
    x_max = min(w, x_max + pad_x)
    y_max = min(h, y_max + pad_y)
    
    # Chuyển đổi sang định dạng YOLO: x_center, y_center, width, height (chuẩn hóa 0->1)
    box_w = (x_max - x_min) / w
    box_h = (y_max - y_min) / h
    x_center = (x_min + x_max) / 2.0 / w
    y_center = (y_min + y_max) / 2.0 / h
    
    # 0 là ID của class 'hand'
    return f"0 {x_center:.6f} {y_center:.6f} {box_w:.6f} {box_h:.6f}"

def main():
    print("=== Chuẩn bị dữ liệu Detection (Hands and Palm) với Auto-labeling ===")
    
    # Khởi tạo MediaPipe Hands trực tiếp từ mp_hands
    # min_detection_confidence thấp một chút để đảm bảo bắt được nhiều ảnh tay nhất có thể
    hands_detector = mp_hands.Hands(static_image_mode=True, max_num_hands=2, min_detection_confidence=0.3)
    
    # Đọc CSV
    df = pd.read_csv(config.DETECTION_CSV)
    print(f"Tổng số dòng trong CSV: {len(df)}")

    # Loại bỏ dòng trùng ảnh
    df_unique = df.drop_duplicates(subset=['imageName'])
    print(f"Số ảnh duy nhất: {len(df_unique)}")

    # Chia train/val theo id người dùng (80/20)
    unique_ids = df_unique['id'].unique()
    train_ids, val_ids = train_test_split(unique_ids, train_size=0.8, random_state=42)

    train_df = df_unique[df_unique['id'].isin(train_ids)]
    val_df = df_unique[df_unique['id'].isin(val_ids)]
    print(f"Train dự kiến: {len(train_df)} ảnh | Val dự kiến: {len(val_df)} ảnh")

    def process_and_write(dataframe, output_path):
        lines = []
        missing = 0
        no_hands = 0
        
        print(f"\nĐang xử lý và tạo nhãn Bounding Box cho file: {os.path.basename(output_path)}...")
        total = len(dataframe)
        count = 0
        
        for _, row in dataframe.iterrows():
            count += 1
            if count % 500 == 0:
                print(f"  Đã xử lý {count}/{total} ảnh...")
                
            img_path = os.path.join(config.DETECTION_IMAGES_DIR, row['imageName'])
            
            if not os.path.exists(img_path):
                missing += 1
                continue
                
            # Dùng AI sinh nhãn tự động
            yolo_bbox = get_hand_bounding_box(img_path, hands_detector)
            if yolo_bbox is None:
                no_hands += 1
                continue # Bỏ qua ảnh này vì không thấy tay
                
            # Lưu file .txt nhãn nằm cùng thư mục với ảnh
            txt_path = img_path.rsplit('.', 1)[0] + '.txt'
            with open(txt_path, 'w', encoding='utf-8') as f:
                f.write(yolo_bbox + '\n')
                
            # LƯU Ý QUAN TRỌNG: Chỉ ghi đường dẫn ảnh, XÓA BỎ SỐ 0 ở cuối
            lines.append(f"{os.path.abspath(img_path)}\n")
            
        with open(output_path, 'w', encoding='utf-8') as f:
            f.writelines(lines)
            
        print(f"-> Hoàn tất ghi {len(lines)} dòng vào {output_path}")
        print(f"   (Bỏ qua do thiếu ảnh: {missing} | Bỏ qua do AI không nhận diện được tay: {no_hands})")

    # Ghi file danh sách và sinh file txt
    process_and_write(train_df, os.path.join(config.DETECTION_DIR, 'train.txt'))
    process_and_write(val_df, os.path.join(config.DETECTION_DIR, 'val.txt'))

    # Tạo file data.yaml cho YOLO
    yaml_content = f"""
path: {config.DETECTION_DIR}
train: train.txt
val: val.txt
nc: 1
names: ['hand']
"""
    with open(os.path.join(config.DETECTION_DIR, 'data.yaml'), 'w', encoding='utf-8') as f:
        f.write(yaml_content)

    print("\n✅ Hoàn tất chuẩn bị dữ liệu và sinh nhãn tự động!")
    print(f"   - File train.txt: {os.path.join(config.DETECTION_DIR, 'train.txt')}")
    print(f"   - File val.txt: {os.path.join(config.DETECTION_DIR, 'val.txt')}")
    print(f"   - File data.yaml: {os.path.join(config.DETECTION_DIR, 'data.yaml')}")
    
    hands_detector.close()

if __name__ == "__main__":
    main()