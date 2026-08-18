# 🤟 Hướng dẫn Hệ Thống Nhận Diện Ngôn Ngữ Ký Hiệu

> **Thư mục làm việc gốc:** `D:\THUC_TAP_CCVI\Nhan_dien_ngon_ngu_ky_hieu`  
> 💡 **MẸO QUAN TRỌNG (Khắc phục lỗi không tìm thấy file):**  
> Khi bạn mở Terminal và thấy dòng lệnh hiển thị `PS D:\THUC_TAP_CCVI>`, hãy gõ câu lệnh sau để di chuyển con trỏ vào thư mục bài làm chính:
> ```powershell
> cd Nhan_dien_ngon_ngu_ky_hieu
> ```
> *Công dụng:* Giúp Terminal trỏ chính xác vào thư mục chứa mã nguồn bài làm của bạn, tránh bị lỗi `[Errno 2] No such file or directory` khi chạy `python Mobile_app/Src/Main.py`.

---

## ⚡ BẢNG TỔNG HỢP CÁC CÂU LỆNH CHẠY (QUICK RUN COMMANDS)

> 📌 **Copy & Paste trực tiếp các dòng lệnh dưới đây vào Terminal để chạy nhanh:**

### 🛠️ Phase 1: Chuẩn bị Dữ liệu (Data Preparation)

#### Option A: Lệnh chạy tuyệt đối trong PowerShell (Dùng trực tiếp trên máy của bạn)
```powershell
# 1. Prepare Detection
& C:\Users\Thang\AppData\Local\Programs\Python\Python311\python.exe d:/THUC_TAP_CCVI/Nhan_dien_ngon_ngu_ky_hieu/Data_preparation/Prepare_detection.py

# 2. Prepare Classification
& C:\Users\Thang\AppData\Local\Programs\Python\Python311\python.exe d:/THUC_TAP_CCVI/Nhan_dien_ngon_ngu_ky_hieu/Data_preparation/Prepare_classification.py

# 3. Prepare Features
& C:\Users\Thang\AppData\Local\Programs\Python\Python311\python.exe d:/THUC_TAP_CCVI/Nhan_dien_ngon_ngu_ky_hieu/Data_preparation/prepare_features.py

# 4. Prepare Sequences
& C:\Users\Thang\AppData\Local\Programs\Python\Python311\python.exe d:/THUC_TAP_CCVI/Nhan_dien_ngon_ngu_ky_hieu/Data_preparation/Prepare_sequences.py
```

#### Option B: Lệnh chạy tương đối (Terminal mở tại D:\THUC_TAP_CCVI\Nhan_dien_ngon_ngu_ky_hieu)
```powershell
python Data_preparation/Prepare_detection.py
python Data_preparation/Prepare_classification.py
python Data_preparation/prepare_features.py
python Data_preparation/Prepare_sequences.py
```


### 🧠 Phase 2: Huấn luyện Mô hình AI (Training Models)

#### Option A: Lệnh chạy tuyệt đối trong PowerShell (Dùng trực tiếp trên máy của bạn)
```powershell
# 1. Huấn luyện Mô hình YOLO (Nhận diện bàn tay trong khung hình)
& C:\Users\Thang\AppData\Local\Programs\Python\Python311\python.exe d:/THUC_TAP_CCVI/Nhan_dien_ngon_ngu_ky_hieu/Cloud_server/Trainer/train_scripts/train_yolo.py

# 2. Huấn luyện Mô hình Trích xuất Đặc trưng (Feature Extractor - EfficientNetB0)
& C:\Users\Thang\AppData\Local\Programs\Python\Python311\python.exe d:/THUC_TAP_CCVI/Nhan_dien_ngon_ngu_ky_hieu/Cloud_server/Trainer/train_scripts/train_feature_extractor.py

# 3. Huấn luyện Mô hình GRU (Nhận diện Cử chỉ / Từ vựng chuỗi thời gian)
& C:\Users\Thang\AppData\Local\Programs\Python\Python311\python.exe d:/THUC_TAP_CCVI/Nhan_dien_ngon_ngu_ky_hieu/Cloud_server/Trainer/train_scripts/train_gru.py

# 4. Huấn luyện Mô hình Phân loại Đa nhãn (Multi-Label Classification)
& C:\Users\Thang\AppData\Local\Programs\Python\Python311\python.exe d:/THUC_TAP_CCVI/Nhan_dien_ngon_ngu_ky_hieu/Cloud_server/Trainer/train_scripts/train_classification.py
```

#### Option B: Lệnh chạy tương đối (Terminal mở tại D:\THUC_TAP_CCVI\Nhan_dien_ngon_ngu_ky_hieu)
```powershell
python Cloud_server/Trainer/train_scripts/train_yolo.py
python Cloud_server/Trainer/train_scripts/train_feature_extractor.py
python Cloud_server/Trainer/train_scripts/train_gru.py
python Cloud_server/Trainer/train_scripts/train_classification.py
```

---

### 🚀 Phase 3: Vận hành Hệ thống & Demo Sản phẩm (Deployment & Live Demo)

> 💡 **Sau khi Huấn luyện (Training) xong, thực hiện các bước sau để vận hành & demo trực tiếp cho Sếp/Công ty:**

#### 1️⃣ Bước 1: Đóng gói & Chuyển đổi Mô hình sang TFLite cho App/Client
Chạy 1 lần duy nhất để nén các file mô hình gốc `.pt` và `.h5` thành 3 file `.tflite` siêu nhẹ trong `mobile_app/assets/`:
```powershell
python Tools/convert_to_mobile.py
```

#### 2️⃣ Bước 2: Khởi chạy Cloud Server Backend (AI Agent Gemini)
Mở **Terminal 1** tại thư mục `D:\THUC_TAP_CCVI\Nhan_dien_ngon_ngu_ky_hieu` và chạy:
```powershell
uvicorn Cloud_server.Api.Main:app --reload --host 0.0.0.0 --port 8000
```
*Tác dụng:* Chạy Backend API nhận diện ca khó, lắng nghe kết nối thời gian thực và tự động sửa lỗi ngữ cảnh câu bằng Gemini LLM.

#### 3️⃣ Bước 3: Khởi chạy Ứng dụng Client Nhận diện Thời gian thực (Webcam Live)
Mở **Terminal 2** tại thư mục `D:\THUC_TAP_CCVI\Nhan_dien_ngon_ngu_ky_hieu` và chạy:
```powershell
python Mobile_app/Src/Main.py
```
*Tác dụng:*
- Mở cửa sổ Webcam live $\rightarrow$ Khoanh vùng bàn tay (YOLO) $\rightarrow$ Đoán từ thô (BiGRU).
- Gửi sang AI Agent (Gemini) sửa thành câu tiếng Việt hoàn chỉnh.
- Tự động đọc câu bằng âm thanh (TTS) và phát hình ảnh qua Camera ảo (Virtual Cam).

#### 4️⃣ Bước 4: Chạy Retrain Tự động (Khi có Ca Khó người dùng phản hồi)
```powershell
python Cloud_server/Trainer/Retrain.py
```

---

🎬 **KỊCH BẢN DEMO NHANH CHO SẾP (QUICK DEMO CHEAT SHEET):**
1. Mở Terminal 1: `uvicorn Cloud_server.Api.Main:app --reload --host 0.0.0.0 --port 8000`
2. Mở Terminal 2: `python Mobile_app/Src/Main.py`
3. Thực hiện cử chỉ trước Webcam $\rightarrow$ Xem khung khoanh tay $\rightarrow$ Nghe âm thanh câu tiếng Việt được đọc ra loa!

---

## 📖 LOGIC CHI TIẾT CỦA TỪNG FILE CHUẨN BỊ DỮ LIỆU (`Data_preparation/`)

### 1. `Data_preparation/Prepare_detection.py`
* 🎯 **Mục đích:** Tự động tạo nhãn vị trí khung bao bàn tay (Bounding Box - YOLO format) trên tập ảnh.
* 🔄 **Logic xử lý:**
  1. Đọc dữ liệu từ file CSV nguồn (`HandInfo.csv` - `config.DETECTION_CSV`).
  2. Lọc bỏ các dòng bị trùng tên ảnh và chia tập Train/Val (80/20) theo `id` người dùng để chống rò rỉ dữ liệu (Data Leakage).
  3. Dùng **MediaPipe Hands** quét từng ảnh để tự động tính tọa độ khung bàn tay `[class x_center y_center w h]`.
  4. Xuất file nhãn `.txt` cho từng ảnh, tạo danh sách `train.txt`, `val.txt` và file cấu hình `data.yaml` cho YOLO.
* 📁 **Đầu ra:** `train.txt`, `val.txt`, `data.yaml` tại `Research_and_Data/Dataset/Detection`.

---

### 2. `Data_preparation/Prepare_classification.py`
* 🎯 **Mục đích:** Tạo tập dữ liệu quản lý nhãn thuộc tính bàn tay (tuổi, giới tính, hướng tay, màu da...).
* 🔄 **Logic xử lý:**
  1. Đọc file CSV `HandInfo.csv`.
  2. Kiểm tra sự tồn tại thực tế của file ảnh trên ổ cứng, loại bỏ đường dẫn ảo.
  3. Chia dữ liệu Train (80%) và Val (20%) dựa theo `id` người dùng.
  4. Tạo 2 file nhãn quản lý tập trung `train.csv` và `val.csv`.
  5. Gom copy toàn bộ các ảnh hợp lệ về một thư mục chung `Classification/images`.
* 📁 **Đầu ra:** `train.csv`, `val.csv` và ảnh trong `Research_and_Data/Dataset/Classification`.

---

### 3. `Data_preparation/prepare_features.py`
* 🎯 **Mục đích:** Tổ chức dữ liệu ảnh ký hiệu tĩnh (chữ cái A-Z) theo thư mục từng lớp phục vụ huấn luyện Feature Extractor.
* 🔄 **Logic xử lý:**
  1. Quét thư mục chứa ảnh gốc `asl-alphabet-train`.
  2. Nếu phát hiện đã có sẵn thư mục `train`/`val`, tự động copy nguyên trạng.
  3. Nếu chưa có, script tự động chia 80% Train / 20% Val bằng `train_test_split` cho từng ký hiệu.
  4. Lưu ảnh vào các thư mục con theo tên lớp (ví dụ: `train/A`, `train/B`,...).
* 📁 **Đầu ra:** Các thư mục lớp trong `Research_and_Data/Dataset/Features/train` và `val`.

---

### 4. `Data_preparation/Prepare_sequences.py`
* 🎯 **Mục đích:** Trích xuất chuỗi 126 tọa độ khớp tay (Keypoints sequence) từ các đoạn video cử chỉ động.
* 🔄 **Logic xử lý:**
  1. **Nguồn WLASL (JSON):** Đọc `nslt_100.json`, phân chia tập Train/Val dựa theo thuộc tính `subset` có sẵn trong file JSON gốc.
  2. **Nguồn Video tự quay (CSV):** Đọc `hand_gestures.csv`, sử dụng kỹ thuật **`GroupShuffleSplit`** nhóm theo `set_id` (người quay/buổi quay) giúp chống rò rỉ dữ liệu tuyệt đối.
  3. Dùng **MediaPipe Hands** trích xuất **126 số tọa độ** (21 điểm × 3 XYZ × 2 tay Trái & Phải) trên 30 khung hình cố định.
  4. Lưu dữ liệu dưới dạng mảng NumPy dạng file `.npy`.
* 📁 **Đầu ra:** File `.npy` lưu tại `Research_and_Data/Dataset/Sequences/processed/train` và `val`.

---

### 5. Cách Python Xử Lý Tự Động Gộp Ảnh Để Chia Tỷ Lệ (Auto-Merge & Re-split Dataset)
* 🎯 **Mục đích:** Gộp toàn bộ tập dữ liệu từ `test` vào `valid` và tự động điều chỉnh cấu hình nhãn về chuẩn Single-Class `hand` (80% Train / 20% Val).
* 🛠️ **Cách Python xử lý tự động:**
  1. **Lấy danh sách tệp từ `test`:** Dùng thư viện `glob` quét bốc toàn bộ 480 file ảnh `.jpg` và 480 file nhãn `.txt` đang nằm trong thư mục `test/`.
  2. **Di chuyển hàng loạt sang `valid`:** Dùng thư viện `shutil.move` để tự động đẩy toàn bộ số ảnh và nhãn đó nạp nối tiếp vào thư mục `valid/images` và `valid/labels`.
  3. **Dọn dẹp & cập nhật cấu hình `data.yaml`:** Xóa thư mục `test` đã rỗng, sau đó cập nhật lại file `data.yaml` loại bỏ nhánh `test` đi, chỉ giữ lại đúng 2 nhánh `train` (3,840 ảnh - 80%) và `val` (960 ảnh - 20%).

---

### 6. Chuẩn Hóa Tên Biến Trực Quan & Thẩm Mỹ Trong `config.py` (Self-Explanatory Variables)
* 🎯 **Mục đích:** Đơn giản hóa và tối ưu cấu hình đường dẫn bài toán YOLOv8 Object Detection cho các phiên bản bài làm tiếp theo.
* 💎 **Cấu trúc biến trực quan (Self-Explanatory Variables):**
  - **`DETECTION_YAML`**: Nhìn vào biết ngay là file cấu hình chính của YOLO (`data.yaml`).
  - **`DETECTION_TRAIN_DIR`**: Nhìn vào biết ngay là nơi chứa 80% ảnh huấn luyện (`train/images`).
  - **`DETECTION_VAL_DIR`**: Nhìn vào biết ngay là nơi chứa 20% ảnh kiểm thử (`valid/images`).

---

### 7. Hệ Thống 4 Menu Điều Khiển Trung Tâm (`Main.py`)
* 🎯 **Mục đích:** Cung cấp giao diện trung tâm đa tính năng cho người dùng, học viên và doanh nghiệp.
* 🚀 **Lệnh khởi chạy nhanh:**
  ```powershell
  python Main.py
  ```
  *(Hoặc `python Mobile_app/Src/Main.py`)*

#### 🎥 Chức năng `[1]`: Dịch thuật Trực tiếp qua Webcam (Live Camera Translation)
* **Quy trình:**
  1. Chọn ngôn ngữ dịch thuật mong muốn: `[V]` Tiếng Việt (mặc định), `[E]` Tiếng Anh, `[J]` Tiếng Nhật, `[K]` Tiếng Hàn.
  2. Bắt dính 2 bàn tay: Tay Trái khung **Xanh Lá** (`Left Hand`), Tay Phải khung **Xanh Dương** (`Right Hand`).
  3. AI gom từ vựng/chữ cái và khi dừng tay $> 2.0\text{s}$, hệ thống tự động dịch qua Gemini LLM và phát âm thanh TTS qua luồng ngầm (Threading) không làm đứng hình camera.
* **Phím tắt nhanh lúc chạy:**
  * `[V]`, `[E]`, `[J]`, `[K]`: Chuyển đổi trực tiếp ngôn ngữ đích (Việt/Anh/Nhật/Hàn).
  * `[Q]`: Thoát về Menu chính.

---

#### 📁 Chức năng `[2]`: Dịch thuật từ Tệp Video MP4/AVI
1. Nhập đường dẫn tệp video có sẵn trong máy (hoặc ấn **Enter** để dùng video mặc định `temp_edge.mp4`).
2. Hệ thống sẽ tự động quét từng khung hình, trích xuất chuỗi cử chỉ và xuất kết quả dịch thuật ra màn hình cùng âm thanh TTS.

---

#### ✍️ Chức năng `[3]`: Bảng Soạn Thảo Văn Bản Ký Hiệu (Sign Text Editor)
Dành cho người khiếm thính muốn soạn thảo một bức thư hoặc đoạn hội thoại bằng thủ ngữ:
* `[Phím Cách - Space]`: Xác nhận chèn từ/chữ cái đang hiển thị vào văn bản.
* `[D]`: Xóa từ vừa nhập gần nhất.
* `[C]`: Xóa toàn bộ đoạn văn bản đã soạn.
* `[S]`: Lưu văn bản đã soạn thành file `composed_sign_text.txt` trên máy tính.
* `[T]`: Kích hoạt giọng đọc AI đọc toàn bộ bức thư/văn bản ra loa.
* `[Q]`: Thoát về Menu chính.

---

#### 🎓 Chức năng `[4]`: Gia Sư AI Chấm Điểm & Người Ảo AI Trợ Lý (AI Avatar)
* **Lựa chọn `[A]` (Luyện tập):** Nhập từ bạn muốn luyện (ví dụ: `apple`, `book`, `hello`...).
  * Góc phải màn hình xuất hiện **Người Ảo AI Trợ Lý (AI Robot Avatar)** với nét mặt cảm xúc và bảng điểm % độ chính xác.
  * Nếu hạ tay quá thấp: Gia sư AI nhắc nhở qua loa *"Hãy nâng tay cao hơn ngang ngực!"*.
  * Nếu làm đúng cử chỉ: Robot mỉm cười và chấm điểm 100%.
* **Lựa chọn `[B]` (Nạp Video Mẫu mới):** Nhập đường dẫn video `.mp4` mới của bạn $\rightarrow$ Gia sư AI tự động copy và nạp bổ sung vào kho Dataset `Sequences/custom_enrollment` để hệ thống tự học thêm từ mới!

---

### 8. Giao Diện Demo Phụ Đề Netflix Cao Cấp (`Demo_ui/App.py`)
* 🎯 **Mục đích:** Giao diện trình diễn phụ đề điện ảnh chuẩn Netflix/YouTube (chữ Tiếng Việt có dấu nét căng, thẻ Badge bo góc, xuất luồng Webcam ảo cho Google Meet/Zoom). Phục vụ quay video báo cáo, thuyết trình đồ án hoặc demo cho đối tác.
* 🚀 **Lệnh khởi chạy:**
  ```powershell
  python Demo_ui/App.py
  ```

---

### 9. Các Module Xử Lý Độc Lập Chuyên Nghiệp (`Tools/`)
* 🎬 **[`Tools/video_translator.py`](file:///D:/THUC_TAP_CCVI/Nhan_dien_ngon_ngu_ky_hieu/Tools/video_translator.py):** Module chuyên trách dịch thuật hàng loạt các tệp video MP4/AVI từ ổ cứng.
* 🎓 **[`Tools/ai_tutor_engine.py`](file:///D:/THUC_TAP_CCVI/Nhan_dien_ngon_ngu_ky_hieu/Tools/ai_tutor_engine.py):** Module Gia sư AI Chấm điểm Thực hành & **Người Ảo AI Trợ Lý (AI Virtual Avatar)**:
  - Nạp Video Mẫu mới và tự động lưu bổ sung vào Dataset `Sequences/custom_enrollment`.
  - Hiển thị Nhân vật Trợ lý AI ảo ở góc phải camera, phát lời nhắc âm thanh TTS và hiển thị bong bóng hội thoại chỉ vị trí sai.

---

### 10. Nâng Cấp Bộ Điều Phối Dữ Liệu Thông Minh 100% (Universal Dynamic Data Adapters)
* 🎯 **Mục đích:** Loại bỏ 100% đường dẫn gán cứng trong `config.py` và tự động nhận diện dữ liệu chuẩn bị trong cả 4 file `Prepare_*.py`.
* 💎 **Chi tiết nâng cấp 4 File Chuẩn bị dữ liệu:**
  1. **`Prepare_detection.py`:** Tự động phát hiện dataset Roboflow/YOLO, gộp `test` sang `valid` chuẩn 80/20 và ép Single-Class `hand` (`nc: 1`).
  2. **`Prepare_classification.py`:** Tự động nhận diện dữ liệu thuộc tính đã gộp, tránh ghi đè làm hỏng file `train.csv` / `val.csv`.
  3. **`prepare_features.py`:** Tự động quét folder lớp ảnh ký hiệu tĩnh `train/` và `val/`.
  4. **`Prepare_sequences.py`:** Tự động quét động bằng `os.walk` trích xuất keypoints từ kho Video Mẫu Gia sư AI (`custom_enrollment`) và nạp bổ sung vào mảng NumPy `.npy`.

---

## 🔄 PHASE 5: Các Tác Vụ Nâng Cao (Retrain & Module Độc Lập)

* **Tự động Retrain lại mô hình khi có ca khó hoặc dữ liệu mới:**
  ```powershell
  python Cloud_server/Trainer/Retrain.py
  ```
* **Chạy dịch video độc lập:**
  ```powershell
  python Tools/video_translator.py
  ```

---

*💡 **Mẹo nhỏ:** Để tránh bị lỗi đường dẫn hoặc lỗi môi trường, hãy luôn ưu tiên copy-paste dòng lệnh từ phần "BẢNG TỔNG HỢP CÁC CÂU LỆNH CHẠY" ở trên thay vì bấm nút Run (Play) trên các phần mềm như VS Code.*






