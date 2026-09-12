# 🤟 Hướng dẫn Hệ Thống Nhận Diện Ngôn Ngữ Ký Hiệu

## 🚀 CÁCH CHẠY TOÀN BỘ BÀI (QUAN TRỌNG NHẤT)
Để khởi chạy toàn bộ hệ thống một cách chính xác và không gặp lỗi, bạn **bắt buộc** phải thực hiện theo thứ tự các lệnh sau trong Terminal (PowerShell):

```powershell
# Bước 1: Di chuyển vào đúng thư mục dự án
cd Nhan_dien_ngon_ngu_ky_hieu

# Bước 2: Chạy script sửa lỗi thư viện (chống xung đột Protobuf/MediaPipe)
.\fix_protobuf_error.bat
```

> **Lưu ý:** Sau khi chạy xong 2 lệnh trên, môi trường của bạn đã hoàn toàn sẵn sàng. Bạn có thể bắt đầu chạy dự án theo quy trình bên dưới.

---

## 🛤️ QUY TRÌNH CHẠY TỪNG BƯỚC TỪ A-Z (STEP-BY-STEP WORKFLOW)

Nếu bạn thiết lập dự án từ đầu, hãy chạy lần lượt các bước sau để đi từ chuẩn bị dữ liệu đến khi chạy demo hệ thống:

### 1️⃣ Bước 1: Chuẩn bị Dữ liệu (Data Preparation)
Trích xuất tọa độ khớp tay (Keypoints) từ các video đầu vào:
```powershell
python Data_preparation/Prepare_sequences.py
```

### 2️⃣ Bước 2: Huấn luyện Mô hình AI (Training)
Lần lượt huấn luyện các mô hình AI từ cơ bản đến phức tạp (tuần tự):
```powershell
python Cloud_server/Trainer/train_scripts/train_dense_static.py
python Cloud_server/Trainer/train_scripts/train_feature_extractor.py
python Cloud_server/Trainer/train_scripts/train_gru.py
```

### 3️⃣ Bước 3: Chuẩn bị Môi trường cho Ứng dụng & Demo (Deployment Prep)
Đóng gói mô hình sang định dạng nhẹ (TFLite) cho ứng dụng:
```powershell
python Tools/convert_to_mobile.py
python Tools/auto_dataset_builder.py
```

### 4️⃣ Bước 4: Khởi chạy Máy chủ Backend (Cloud Server AI)
Mở một cửa sổ Terminal mới (nhớ chạy `cd Nhan_dien_ngon_ngu_ky_hieu`), khởi động server API kết nối với Gemini để sửa lỗi ngữ pháp văn bản:
```powershell
uvicorn Cloud_server.Api.Main:app --reload --host 0.0.0.0 --port 8000
```

### 5️⃣ Bước 5: Khởi chạy Ứng dụng Trung tâm (Giao diện người dùng)
Mở một cửa sổ Terminal khác, khởi chạy giao diện tương tác:
```powershell
python Main.py
```

---

## ⚡ CÁC TÍNH NĂNG CHÍNH (FUNCTIONS)

Hệ thống có 4 chức năng chính được tích hợp sẵn. Để khởi chạy giao diện điều khiển trung tâm, sử dụng lệnh:
```powershell
python Main.py
```

### 🎥 1. Dịch thuật Trực tiếp qua Webcam (Live Camera Translation)
- **Mô tả:** Nhận diện ngôn ngữ ký hiệu theo thời gian thực và dịch ra các ngôn ngữ khác bằng AI.
- **Cách hoạt động:**
  - Bắt dính tay trái (Khung Xanh Lá) và tay phải (Khung Xanh Dương).
  - Khi bạn dừng cử chỉ > 2 giây, AI sẽ tự động dịch câu qua Gemini LLM và phát giọng đọc (TTS).
- **Phím tắt:**
  - `[V]` Tiếng Việt, `[E]` Tiếng Anh, `[J]` Tiếng Nhật, `[K]` Tiếng Hàn.
  - `[Q]`: Thoát.

### 📁 2. Dịch thuật từ Video (MP4/AVI)
- **Mô tả:** Chọn một video ngôn ngữ ký hiệu có sẵn trên máy tính để hệ thống phân tích và dịch.
- **Cách hoạt động:** Quét từng khung hình video để trích xuất chuỗi cử chỉ, sau đó hiển thị kết quả dịch thuật ra màn hình cùng âm thanh đọc câu.

### ✍️ 3. Bảng Soạn Thảo Văn Bản Ký Hiệu (Sign Text Editor)
- **Mô tả:** Chức năng dành cho người khiếm thính muốn soạn thảo văn bản hoặc viết thư bằng thủ ngữ.
- **Phím tắt điều khiển:**
  - `[Space]`: Chèn từ hiện tại vào văn bản.
  - `[D]`: Xóa từ vừa nhập gần nhất.
  - `[C]`: Xóa toàn bộ đoạn văn bản đã soạn.
  - `[S]`: Lưu văn bản đã soạn thành file `.txt`.
  - `[T]`: Kích hoạt AI đọc toàn bộ đoạn văn bản.

### 🎓 4. Gia Sư AI Chấm Điểm & Trợ Lý Ảo (AI Avatar)
- **Luyện tập (Tùy chọn A):** Bạn nhập từ vựng muốn luyện tập. Trợ lý ảo AI sẽ xuất hiện trên màn hình, hướng dẫn và chấm điểm phần trăm độ chính xác của cử chỉ. Trợ lý sẽ nhắc nhở nếu bạn làm sai (ví dụ: để tay quá thấp).
- **Thêm từ mới (Tùy chọn B):** Nạp video chứa cử chỉ mới. Hệ thống sẽ tự động học và thêm từ vựng này vào thư viện (Dataset) để nhận diện sau này.

---

## 🛠️ CÔNG CỤ HỖ TRỢ (TOOLS)

Các công cụ xử lý chuyên sâu được đặt trong thư mục `Tools/`. Bạn có thể chạy độc lập từng công cụ tùy theo nhu cầu:

### 1. Công Cụ Dịch Video Độc Lập
- **Chức năng:** Dịch thuật hàng loạt các tệp video MP4/AVI mà không cần mở giao diện chính.
- **Lệnh chạy:**
  ```powershell
  python Tools/video_translator.py
  ```

### 2. Động Cơ Gia Sư AI (AI Tutor Engine)
- **File:** `Tools/ai_tutor_engine.py`
- **Chức năng:** Chứa logic cốt lõi của tính năng Gia sư chấm điểm. Xử lý hoạt ảnh của nhân vật Trợ lý Ảo, phát âm thanh nhắc nhở và tự động lưu video mẫu mới vào thư mục `Sequences/custom_enrollment`.

### 3. Đóng Gói Mô Hình Cho Điện Thoại (Mobile Converter)
- **Chức năng:** Nén và chuyển đổi mô hình gốc sang định dạng `.tflite` siêu nhẹ để tích hợp vào ứng dụng di động.
- **Lệnh chạy:**
  ```powershell
  python Tools/convert_to_mobile.py
  ```

### 4. Xây Dựng Dữ Liệu Tự Động (Auto Dataset Builder)
- **Chức năng:** Tự động cấu trúc lại tập dữ liệu (dataset) khi có các file video mới được nạp vào, giúp hệ thống không bị lỗi đường dẫn.
- **Lệnh chạy:**
  ```powershell
  python Tools/auto_dataset_builder.py
  ```

---

## 🧠 CÁC TÁC VỤ NÂNG CAO & HUẤN LUYỆN LẠI (RETRAIN)

### 1. Chuẩn bị dữ liệu (Data Preparation)
Nếu bạn cập nhật video mới và cần trích xuất lại tọa độ (Keypoints):
```powershell
python Data_preparation/Prepare_sequences.py
```

### 2. Huấn luyện lại mô hình (Auto Retrain)
Nếu mô hình nhận diện chưa tốt hoặc bạn vừa thêm từ vựng mới qua Gia Sư AI, hãy chạy lệnh sau để máy học lại:
```powershell
python Cloud_server/Trainer/Retrain.py
```

### 3. Khởi chạy Giao diện Phụ đề Cao cấp & Server API
- **Khởi chạy Cloud Server Backend (Gemini AI):**
  ```powershell
  uvicorn Cloud_server.Api.Main:app --reload --host 0.0.0.0 --port 8000
  ```
- **Khởi chạy Giao diện Demo Phụ đề (chuẩn Netflix):** Phục vụ thuyết trình, xuất luồng Webcam ảo.
  ```powershell
  python Demo_ui/App.py
  ```
