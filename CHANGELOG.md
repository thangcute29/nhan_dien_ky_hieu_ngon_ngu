# Changelog / Session Log

## [2026-07-08] Xây dựng Hệ thống Classification Đa nhãn (Multi-Label)

### 1. `Data_preparation/Prepare_classification.py` (MỚI)
- Đã thiết lập hoàn chỉnh quy trình tách dữ liệu Classification ra khỏi luồng Detection.
- Dữ liệu ảnh được quy về `Dataset/Classification/images` và nhãn được lưu tại `train.csv` (Multi-Label) theo phương pháp chia tệp khách hàng 80/20 chuẩn xác bằng mã User ID.

### 2. `Cloud_server/Trainer/train_scripts/train_classification.py` (MỚI)
- Đã lập trình xong file huấn luyện phân loại đa nhãn (Multi-head Network).
- **Kiến trúc:** Sử dụng `EfficientNet-B0` làm backbone (đồng bộ với file Features) chia thành 5 đầu ra độc lập.
- **Tính năng Đột phá:** Tích hợp Early Stopping (bằng cơ chế đếm lùi `patience_counter` chuẩn PyTorch thay vì gọi Keras) và vòng lặp tự động học lại nếu Validation Accuracy < 90%.
- **Chia sẻ Assets:** Cập nhật đường dẫn lưu Model (`multi_head_best.pth`) và từ điển nhãn (`classification_classes.json`) vào đúng thư mục `config.SHARED_ASSETS_DIR` (`Shared_lib/assets/`) để App/Web dùng chung.
- **Trạng thái hiện tại:** Hệ thống Training đã sẵn sàng 100%. Bước tiếp theo có thể chạy huấn luyện mô hình này hoặc tiếp tục hoàn thiện `train_detection.py` (YOLO).

### 3. Cập nhật `config.py`
- Dọn dẹp biến `DETECTION_CSV` bị thừa thãi do logic dữ liệu `HandInfo.csv` đã được tách sang cho bài toán Classification.
- Loại bỏ các đường dẫn cấu hình bị trùng lặp, tối ưu lại biến `SHARED_ASSETS_DIR`.


## [2026-07-08] Sửa Lỗi Dataset YOLO, Bounding Box và Tích hợp AI Auto-Labeling

### 1. `Data_preparation/Prepare_detection.py`
- Khắc phục sự hiểu nhầm cấu trúc dữ liệu giữa bài toán Phân loại (Classification) và Nhận diện (Detection).
- **Lỗi cũ:** Code cũ thêm hậu tố ` 0` vào cuối file `train.txt` khiến YOLO không đọc được ảnh, và sau khi đọc được ảnh thì lại không có file tọa độ Bounding Box, khiến YOLO coi 11.000 bức ảnh là ảnh nền (Backgrounds).
- **Giải pháp Đột phá:** Viết lại toàn bộ file. Nhúng thư viện AI `MediaPipe` vào để tự động quét 11.000 tấm ảnh, tự động đóng khung bàn tay và sinh ra các file nhãn `.txt` (Auto-labeling) đúng chuẩn YOLO. Xóa bỏ hoàn toàn số `0` lỗi ở cuối đường dẫn trong `train.txt` và `val.txt`.

### 2. Dữ liệu huấn luyện (Dataset YOLO)
- Sửa lỗi văng chương trình (FileNotFound) khi load dữ liệu: Xử lý tận gốc vấn đề ký tự tiếng Việt có dấu (`THỰC TẬP CCVI` -> `THUC_TAP_CCVI`). Cập nhật đồng loạt toàn bộ các đường dẫn tuyệt đối trong `data.yaml`, `train.txt` và `val.txt` sang đường dẫn không dấu.
- Giải cứu sự cố 0 bytes cho 2 file cấu hình dataset, tái tạo danh sách 8860 ảnh train và 2216 ảnh val hoàn chỉnh.

---

## [2026-07-01] Sửa lỗi cấu hình Assets và đồng bộ quá trình lưu Model

### 1. `config.py`
- Khắc phục lỗi văng hệ thống (`UnicodeEncodeError`) do không in được ký tự tiếng Việt trên CMD của Windows. Lỗi này khiến các file train (như `train_feature_extractor.py`) bị crash ngay lập tức ở câu lệnh `import config`.
- **Giải pháp**: Bổ sung cờ ép mã hóa `utf-8` cho `sys.stdout` ở đầu file.

### 2. `Cloud_server/Trainer/train_scripts/train_yolo.py`
- Khắc phục tình trạng YOLO tự lưu model riêng lẻ vào thư mục `Trainer/runs` thay vì dùng chung kho lưu trữ với các AI khác.
- **Giải pháp**: Cập nhật mã nguồn để khi huấn luyện thành công (độ chính xác >90%), YOLO tự động lưu file `hand_det_yolo.pt` vào khu vực dùng chung `config.SHARED_ASSETS_DIR` (thư mục `assets`).

### 3. `Cloud_server/Trainer/train_scripts/train_gru.py`
- Khắc phục lỗi ngầm `NameError` khi mô hình huấn luyện vượt ngưỡng 90% (hệ thống cố gắng lưu biến `model_info` chưa được khai báo trước đó, dẫn đến crash toàn bộ quá trình train).
- **Giải pháp**: Đã khai báo bổ sung biến từ điển `model_info` (chứa `classes`, `seq_len`, `input_dim`) trước bước lưu ra thư mục `assets`.
