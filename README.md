# Hướng dẫn Chạy Huấn Luyện (Training) Hệ Thống Nhận Diện Ngôn Ngữ Ký Hiệu

Đây là danh sách các câu lệnh chuẩn xác nhất để chạy quá trình huấn luyện các mô hình AI. 
**Lưu ý quan trọng:** Bạn phải mở Terminal ở thư mục gốc của dự án (`D:\THUC_TAP_CCVI\Sign_language`), sau đó copy và dán các lệnh dưới đây vào rồi ấn Enter.

---

### 1. Huấn luyện Mô hình YOLO (Nhận diện bàn tay)
Lệnh này sẽ huấn luyện mạng YOLOv8 để tìm ra vị trí bàn tay trong khung hình (Bounding Box).
```bash
python Cloud_server/Trainer/train_scripts/train_yolo.py
```

### 2. Huấn luyện Mô hình Trích xuất Đặc trưng (Feature Extractor)
Lệnh này sẽ huấn luyện mạng EfficientNetB0 để trích xuất các đặc trưng tĩnh của tay.
```bash
python Cloud_server/Trainer/train_scripts/train_feature_extractor.py
```

### 3. Huấn luyện Mô hình GRU Đô Con (Nhận diện Hành động/Từ vựng)
Lệnh này sẽ huấn luyện mô hình phân tích chuỗi thời gian GRU để dịch các cử chỉ liên tiếp thành một từ vựng cụ thể (300 lớp).
```bash
python Cloud_server/Trainer/train_scripts/train_gru.py
```

### 4. Huấn luyện Mô hình Phân loại Đa nhãn (Multi-Label Classification)
Lệnh này huấn luyện AI phân loại đa nhãn (Multi-head Network) tách biệt dựa trên các đặc điểm của tay.
```bash
python Cloud_server/Trainer/train_scripts/train_classification.py
```

---
*💡 **Mẹo nhỏ:** Để tránh bị lỗi đường dẫn hoặc lỗi môi trường, hãy luôn ưu tiên copy-paste dòng lệnh thay vì dùng nút Run (Play) trên các phần mềm như VS Code.*
