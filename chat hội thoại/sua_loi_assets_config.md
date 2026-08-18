# 🛠️ Tóm Tắt Sửa Lỗi Assets, Config & Tiến Trình Huấn Luyện Dự Án

Tài liệu ghi chép tóm tắt các điểm điều chỉnh cấu hình hệ thống, luồng lưu trữ mô hình và cơ chế tự động dọn rác bộ nhớ.

---

## 📌 1. Đơn Giản Hóa Bộ Trích Xuất Đặc Trưng (`train_classification.py`)
* **Thay đổi:** Loại bỏ hoàn toàn khối `if-else` chọn lựa giữa ResNet50, MobileNet và EfficientNet.
* **Tối ưu:** Giữ duy nhất kiến trúc **EfficientNet-B0** vì dung lượng siêu nhẹ, tối ưu 100% cho thiết bị di động và tránh chọn nhầm mô hình nặng.

---

## 📌 2. Chuẩn Hóa Luồng Lưu Trữ Kho Mô Hình (`Shared_lib/assets/`)
* **Cơ chế:** Cả 4 file huấn luyện (`train_yolo.py`, `train_feature_extractor.py`, `train_gru.py`, `train_classification.py`) đều lưu trọng số thành phẩm (`.pt`, `.h5`) trực tiếp vào kho dùng chung `Shared_lib/assets/`.
* **Đồng bộ:** Sửa script `Tools/convert_to_mobile.py` trỏ đọc từ `Shared_lib/assets/` thay vì thư mục `models/` cũ, triệt tiêu lỗi `FileNotFoundError`.

---

## 📌 3. Luồng 3 Bước Chạy Hệ Thống Sau Huấn Luyện
1. **Bước 1 (Chuyển đổi TFLite):** Chạy `Tools/convert_to_mobile.py` chuyển trọng số PyTorch/Keras sang `.tflite` đẩy vào `mobile_app/assets`.
2. **Bước 2 (Khởi động Backend API):** Chạy `Cloud_server/Api/Main.py` nạp `ContextAgent` và `LLMCorrector`.
3. **Bước 3 (Khởi động Client UI):** Chạy `Demo_ui/App.py` (trình diễn phụ đề Netflix) hoặc `Mobile_app` trên điện thoại.

---

## 📌 4. Tính Năng Checkpoint Định Kỳ, Resume & Dọn Rác Tự Động
* **Save Period = 5:** Lưu trọng số dự phòng mỗi 5 epoch (`epoch_5.pth`).
* **Resume (Tự động khôi phục):** Tự động nạp file `_last.pth` gần nhất để tiếp tục train nếu bị ngắt điện/dừng đột ngột.
* **Tự động dọn rác ổ cứng (Auto Garbage Collection):** Khi huấn luyện xong, hệ thống dùng `glob` tự xóa toàn bộ các file dự phòng tạm `_epoch_*.pth`/`.h5` và `_last`, giữ ổ cứng luôn sạch đĩa.

---

## 📌 5. Sửa Lỗi Tương Thích Keras 2.16+ & Tối Ưu Mạng BiGRU
* **Vá lỗi `period` Keras:** Thay thế tham số `period` bị khai tử của `ModelCheckpoint` bằng lớp tùy chỉnh `SmartProgressCallback` xử lý chu kỳ `% 5 == 0`.
* **Kiến trúc BiGRU "Đô Con":** Mở rộng tầng Dense đệm lên 256 chiều, rút gọn về 1 tầng `Bidirectional(GRU(128))`, bổ sung `Dropout(0.5)` + штраф L2 (`0.0001`) để triệt tiêu học vẹt.
* **Quy trình Thu gọn Từ vựng (Top 100 từ):** Đổi cấu hình `config.py` sang `nslt_100.json`, yêu cầu dọn sạch cache cũ trước khi chạy `Prepare_sequences.py` để trích xuất ma trận dữ liệu mới.
