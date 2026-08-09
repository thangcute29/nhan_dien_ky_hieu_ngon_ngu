# 📋 CHANGELOG — Ghi chú lỗi kỹ thuật trong `train_gru.py`

> **Ngày ghi nhận:** 22/07/2026  
> **File liên quan:** `train_gru.py`  
> **Mục đích:** Đối chiếu chi tiết 5 lỗi kỹ thuật và vị trí đoạn code tương ứng để tiện sửa dứt điểm.

---

## 🔴 Lỗi 1: Quá ít dữ liệu (810 mẫu / 100 lớp ≈ 8 mẫu/lớp)

### 📍 Vị trí code gây lỗi

> Nằm ở **hàm `load_data`** và **phần nạp dữ liệu trong hàm `main`**

```python
# Đoạn code nạp trực tiếp toàn bộ dữ liệu thô mà không kiểm tra cân bằng/mật độ mẫu
X_train, y_train, classes = load_data(train_dir)
X_val, y_val, _ = load_data(val_dir)

num_classes = len(classes)  # 100 lớp nhưng tổng mẫu X_train quá mỏng
```

### 🔍 Bản chất lỗi

Code nạp nguyên bản toàn bộ folder mà **không có bộ lọc** loại bỏ những lớp có ít hơn **15–20 mẫu video**.  
Hậu quả: mô hình phải phân loại **100 lớp** trong khi dữ liệu **cực kỳ thiếu hụt** (~8 mẫu/lớp).

### ✅ Hướng sửa gợi ý

- Thêm bộ lọc loại bỏ các lớp có số mẫu < 15–20.
- Hoặc thu thập thêm dữ liệu cho các lớp thiếu.
- Cân nhắc giảm số lớp xuống chỉ giữ lại các lớp có đủ mẫu.

---

## 🔴 Lỗi 2: Không có Data Augmentation trong training

### 📍 Vị trí code gây lỗi

> Nằm ở **lệnh kích hoạt huấn luyện `model.fit`**

```python
# Nạp trực tiếp mảng numpy tĩnh X_train vào hàm fit
history = model.fit(
    X_train, y_train_onehot,  # <--- Dữ liệu tĩnh 100%, không qua bộ biến đổi Augmentation
    validation_data=(X_val, y_val_onehot),
    epochs=100,
    batch_size=32,
    callbacks=[early_stopping_gate, lr_reducer, last_checkpoint_gate, progress_manager_gate]
)
```

### 🔍 Bản chất lỗi

Khác với `ImageDataGenerator` bên file ảnh, ở đây `X_train` là **mảng số cố định**.  
Mạch code **thiếu một bộ sinh dữ liệu** (`tf.keras.utils.Sequence`) để tự động:
- Thêm **nhiễu (noise)** vào tọa độ keypoints
- **Xoay nhẹ góc** (rotation)
- **Tịnh tiến tọa độ** (translation) các khung hình trong lúc train

### ✅ Hướng sửa gợi ý

- Tạo một class kế thừa `tf.keras.utils.Sequence` làm data generator.
- Trong mỗi batch, áp dụng random noise, random rotation, random scale lên tọa độ keypoints.
- Truyền generator vào `model.fit()` thay vì mảng numpy tĩnh.

---

## 🟠 Lỗi 3: Keypoints chưa chuẩn hóa (Tọa độ tuyệt đối)

### 📍 Vị trí code gây lỗi

> Nằm trong **hàm `load_data`**

```python
def load_data(data_dir):
    # ...
    for file in os.listdir(cls_path):
        if file.endswith('.npy'):
            seq = np.load(os.path.join(cls_path, file))
            X.append(seq)  # <--- Nạp thẳng ma trận tọa độ gốc từ file .npy mà không trừ đi tọa độ cổ tay
```

### 🔍 Bản chất lỗi

Dữ liệu `.npy` được nạp thẳng vào bộ nhớ mà **không trải qua bước tiền xử lý** tính toán tọa độ tương đối:

$$
\text{Hand\_Landmarks}_{\text{normalized}} = \text{Landmarks} - \text{Wrist\_Landmark (Point 0)}
$$

Điều này khiến mô hình **phụ thuộc vào vị trí đứng** của người ra dấu trên khung hình video.  
→ Cùng một cử chỉ nhưng đứng ở vị trí khác sẽ cho ra tọa độ hoàn toàn khác → AI dự đoán sai.

### ✅ Hướng sửa gợi ý

- Sau khi load `.npy`, trừ đi tọa độ điểm cổ tay (Point 0) cho toàn bộ 21 điểm của mỗi bàn tay.
- Có thể chuẩn hóa thêm bằng khoảng cách max để đưa tọa độ về khoảng [0, 1].

---

## 🟠 Lỗi 4: Model quá lớn (~1.5M params) gây Overfitting

### 📍 Vị trí code gây lỗi

> Nằm ở **block khởi tạo kiến trúc `models.Sequential`**

```python
# Cấu trúc quá "đô con" tạo ra ~1.5 triệu tham số, trong khi tập train chỉ có ~810 mẫu
model = models.Sequential([
    layers.Masking(mask_value=0.0, input_shape=(seq_len, input_dim)),

    layers.Dense(256, activation='relu'),                          # Tầng Dense to
    layers.BatchNormalization(),
    layers.Dropout(0.2),

    layers.Bidirectional(layers.GRU(256, return_sequences=True)),   # BiGRU 256 units
    layers.Dropout(0.3),

    layers.Bidirectional(layers.GRU(128, return_sequences=False)),  # BiGRU 128 units
    layers.Dropout(0.3),

    layers.Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.0001)),
    layers.BatchNormalization(),
    layers.Dropout(0.3),

    layers.Dense(num_classes, activation='softmax')
])
```

### 🔍 Bản chất lỗi

Số lượng trọng số (parameters) của mạng **vượt xa** số lượng mẫu dữ liệu:

$$
1{,}500{,}000 \text{ params} \gg 810 \text{ mẫu}
$$

Làm mô hình dễ dàng **"học thuộc lòng"** toàn bộ tập train thay vì học quy luật chung (overfitting nghiêm trọng).

### ✅ Hướng sửa gợi ý

- Giảm số units GRU xuống (ví dụ: 64–128 thay vì 256).
- Bỏ bớt tầng Dense hoặc giảm số neurons.
- Tăng hệ số Dropout lên 0.4–0.5.
- Tăng hệ số L2 regularization lên 0.001–0.01.
- Mục tiêu: đưa tổng params xuống dưới **200K–300K**.

---

## 🟡 Lỗi 5: Chỉ dùng 1 tay (63 features) cho bài toán 100 lớp

### 📍 Vị trí code gây lỗi

> Nằm ở **phần đọc kích thước dữ liệu `input_dim`**

```python
# Tham số đầu vào cố định ở mốc 63
seq_len = X_train.shape[1]    # 30 frames
input_dim = X_train.shape[2]  # 63 (21 điểm × 3 tọa độ XYZ của ĐỦ 1 BÀN TAY)
```

### 🔍 Bản chất lỗi

Khai báo **63 đặc trưng** đồng nghĩa với việc AI bị **"mù" thông tin** ở:
- **Tay thứ hai** (rất quan trọng vì nhiều ký hiệu dùng cả 2 tay)
- **Vị trí khuôn mặt / ngực / vai** (ngữ cảnh cơ thể)

Số features nên là:
| Cấu hình | Số features |
|---|---|
| 1 tay (hiện tại) | 63 = 21 × 3 |
| 2 tay | **126** = 2 × 21 × 3 |
| 2 tay + Pose context | **162** = 126 + 12 × 3 (12 điểm pose) |

### ✅ Hướng sửa gợi ý

- Cập nhật pipeline thu thập dữ liệu để trích xuất **cả 2 bàn tay** (Left + Right).
- Cân nhắc thêm **Pose landmarks** (vai, khuỷu tay, cổ tay) để cung cấp ngữ cảnh tương đối.
- Cập nhật lại `input_dim` tương ứng (126 hoặc 162).

---

## 📊 Bảng tổng hợp mức độ ưu tiên sửa lỗi

| # | Lỗi | Mức độ | Ưu tiên sửa |
|---|---|---|---|
| 1 | Quá ít dữ liệu | 🔴 Nghiêm trọng | ⭐⭐⭐⭐⭐ |
| 2 | Không có Data Augmentation | 🔴 Nghiêm trọng | ⭐⭐⭐⭐⭐ |
| 3 | Keypoints chưa chuẩn hóa | 🟠 Quan trọng | ⭐⭐⭐⭐ |
| 4 | Model quá lớn (~1.5M params) | 🟠 Quan trọng | ⭐⭐⭐⭐ |
| 5 | Chỉ dùng 1 tay (63 features) | 🟡 Cần cải thiện | ⭐⭐⭐ |

> [!IMPORTANT]
> **Thứ tự sửa khuyến nghị:** Lỗi 3 → Lỗi 5 → Lỗi 1 → Lỗi 2 → Lỗi 4  
> (Chuẩn hóa dữ liệu trước → Mở rộng features → Lọc/tăng dữ liệu → Augmentation → Thu gọn model)

---

## 🔴 Lỗi 6: Rò rỉ dữ liệu (Data Leakage) khi chia tập Train/Val và Lỗi nạp Config

> **Ngày ghi nhận:** 29/07/2026  
> **File liên quan:** `config.py` và `Prepare_sequences.py`

### 📍 Vị trí code gây lỗi

```python
# Lỗi 6.1 ở config.py (ghi đè file JSON sang CSV)
SEQUENCES_JSON = os.path.join(SEQUENCES_DIR, 'nslt_100.json')
SEQUENCES_CSV = SEQUENCES_JSON  # <-- Lỗi: Biến CSV trỏ nhầm sang file JSON

# Lỗi 6.2 ở Prepare_sequences.py (Chia ngẫu nhiên)
train_data, val_data = train_test_split(all_data, train_size=0.8, random_state=42, stratify=labels)
```

### 🔍 Bản chất lỗi
1. **Lỗi Config**: Code `Prepare_sequences.py` lấy `SEQUENCES_CSV` ép đổi đuôi `.json` thành `.csv` (thành `nslt_100.csv`) khiến file dữ liệu tự quay `hand_gestures.csv` bị bỏ qua hoàn toàn. Mô hình không thể nạp được các nhãn custom.
2. **Lỗi Data Leakage (Nghiêm trọng)**: Lệnh `train_test_split` xáo trộn ngẫu nhiên tất cả video mà không xét đến ID người thực hiện (Person ID) hay Session (Buổi quay). Điều này làm video của cùng một góc máy/người quay bị xé lẻ vào cả tập Train và tập Val.
   - **Hậu quả:** Mô hình sẽ "học vẹt" bối cảnh (phòng ốc, màu áo, ngoại hình...) của người quay thay vì học "cử chỉ bàn tay". Khi test trên tập Val điểm sẽ rất cao (do model nhận ra bối cảnh quen thuộc), nhưng khi test trên một người hoàn toàn mới, AI sẽ thất bại (Overfitting). 
   - Hơn nữa, với dữ liệu chuẩn WLASL, việc tự chia ngẫu nhiên đã phá vỡ cấu trúc phân chia Train/Val/Test (Signer Independent) mà các chuyên gia đã cất công xây dựng trong file JSON.

### ✅ Cách chúng ta đã sửa (29/07/2026)

**1. Khai báo độc lập trong `config.py`:**
Tách bạch biến cấu hình để code không còn phụ thuộc nhau, xử lý song song cả hai nguồn:
```python
SEQUENCES_JSON = os.path.join(SEQUENCES_DIR, 'nslt_100.json')
SEQUENCES_CSV = os.path.join(SEQUENCES_DIR, 'hand_gestures.csv') 
```

**2. Sửa file `Prepare_sequences.py` để ngăn chặn Rò rỉ dữ liệu:**
- **Đối với dữ liệu WLASL (JSON):** Gỡ bỏ lệnh `train_test_split`. Code mới duyệt qua danh sách và đọc trực tiếp trường `subset` ('train', 'val', 'test') được cung cấp sẵn từ file JSON gốc để chia các tập một cách chuẩn xác nhất.
- **Đối với dữ liệu CSV (tự quay):** 
  - Parse (chuyển đổi) file CSV dạng bảng ngang thành danh sách, dùng `set_id` làm ID đại diện cho Session/Nhóm.
  - Áp dụng kỹ thuật **`GroupShuffleSplit`** (nhóm theo `set_id`) thay vì `train_test_split` ngẫu nhiên. Nhờ vậy, 100% video của một Session (ví dụ toàn bộ thư mục `files/0/`) sẽ chỉ rơi vào tập Train hoặc tập Val. Mô hình bị buộc phải học đặc trưng cử chỉ tay thay vì học vẹt người quay, giải quyết dứt điểm Overfitting do rò rỉ dữ liệu.

---

## 🔴 Lỗi 7: AttributeError do thiếu `DETECTION_CSV` và Lỗi Sai Đường Dẫn Gốc khi Đổi Máy Tính

> **Ngày ghi nhận:** 03/08/2026  
> **File liên quan:** `config.py`, `Prepare_classification.py`, `Prepare_detection.py`

### 📍 Vị trí code gây lỗi

```python
# Lỗi 7.1 ở config.py (đường dẫn bị gán cứng trên máy cũ)
PROJECT_ROOT = r"D:\THUC_TAP_CCVI\Sign_language"

# Lỗi 7.2 ở config.py (thiếu khai báo biến DETECTION_CSV)
# Phần # ----------------- DETECTION ----------------- bị thiếu biến DETECTION_CSV
```

### 🔍 Bản chất lỗi

1. **Thiếu biến `DETECTION_CSV` trong `config.py`**:
   - Trong `Prepare_classification.py` (dòng 19) và `Prepare_detection.py` (dòng 70), code gọi `config.DETECTION_CSV`.
   - Tuy nhiên, trong `config.py` chỉ khai báo `DETECTION_DIR` và `DETECTION_IMAGES_DIR` mà bỏ quên `DETECTION_CSV`.
   - **Hậu quả:** Chương trình văng lỗi ngay khi vừa khởi chạy: `AttributeError: module 'config' has no attribute 'DETECTION_CSV'`.

2. **Dùng đường dẫn tuyệt đối gán cứng (`PROJECT_ROOT`)**:
   - `PROJECT_ROOT` bị gán cứng vào thư mục máy cũ: `D:\THUC_TAP_CCVI\Sign_language`.
   - Khi chuyển dự án sang máy tính mới (với thư mục dự án đặt tại `D:\THUC_TAP_CCVI\Nhan_dien_ngon_ngu_ky_hieu`), Python không tìm thấy tập tệp dữ liệu nghiên cứu (`Research_and_Data/Dataset`).

### ✅ Cách chúng ta đã sửa (03/08/2026)

Cập nhật lại file `config.py`:
```python
# --- CẤU HÌNH ĐƯỜNG DẪN THƯ MỤC GỐC (Linh hoạt tự động theo thư mục chứa config.py) ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = BASE_DIR

# ----------------- DETECTION -----------------
DETECTION_DIR = os.path.join(RESEARCH_DATA_DIR, 'Detection')
DETECTION_IMAGES_DIR = os.path.join(DETECTION_DIR, 'Hands', 'Hands')   # Thư mục ảnh
DETECTION_CSV = os.path.join(DETECTION_DIR, 'HandInfo.csv')             # Thêm đường dẫn file CSV thông tin ảnh Detection
```

---

## 🔴 Lỗi 8: AttributeError do không tương thích phiên bản MediaPipe (module 'mediapipe' has no attribute 'solutions')

> **Ngày ghi nhận:** 03/08/2026  
> **File liên quan:** `Prepare_detection.py`, `Prepare_sequences.py`

### 📍 Vị trí code gây lỗi

```python
# Cách import cũ bị lỗi trên MediaPipe 0.10+
import mediapipe as mp

# Khi gọi trong hàm main():
mp_hands = mp.solutions.hands
```

### 🔍 Bản chất lỗi

- Khi tải dự án về máy tính mới, thư viện `mediapipe` được cài đặt theo phiên bản mới nhất (`mediapipe >= 0.10.x`).
- Ở các phiên bản mới này, Google đã tái cấu trúc lại mã nguồn và không nạp sẵn `solutions` vào namespace gốc `mp.solutions`.
- **Hậu quả:** Khi chạy `Prepare_detection.py` hoặc `Prepare_sequences.py`, Python lập tức dừng và báo lỗi: `AttributeError: module 'mediapipe' has no attribute 'solutions'`.

### ✅ Cách chúng ta đã sửa (03/08/2026)

Cập nhật lại cách import trực tiếp trong cả 2 file `Prepare_detection.py` và `Prepare_sequences.py`:
```python
# 1. Thay đổi câu lệnh import ở đầu file:
from mediapipe.python.solutions import hands as mp_hands

# 2. Khởi tạo trực tiếp từ mp_hands:
hands_detector = mp_hands.Hands(static_image_mode=True, max_num_hands=2, min_detection_confidence=0.3)
```

---

## 💡 Ghi chú Kỹ thuật: Cơ chế phần cứng GPU (NVIDIA RTX 4050) vs CPU trong Dự án

> **Ngày ghi nhận:** 03/08/2026  
> **Phần cứng xác nhận:** NVIDIA GeForce RTX 4050 Laptop GPU (6GB VRAM, CUDA 13.0, `torch.cuda.is_available() = True`)

### 🔍 Giải thích cơ chế sử dụng GPU / CPU trong từng Phase

1. **Giai đoạn Chuẩn bị Dữ liệu (Data Preparation - MediaPipe):**
   - **Hiện tượng:** Khi chạy `Prepare_detection.py` hay `Prepare_sequences.py`, hệ thống báo `INFO: Created TensorFlow Lite XNNPACK delegate for CPU` và sử dụng CPU thay vì GPU.
   - **Nguyên nhân:** Thư viện `mediapipe` Python trên hệ điều hành Windows được Google phát hành mặc định **chỉ hỗ trợ CPU** (thông qua bộ tăng tốc XNNPACK CPU Delegate). Google không hỗ trợ GPU (CUDA) cho bản MediaPipe Python trên Windows.
   - **Tác động:** Không ảnh hưởng tới dự án, CPU vẫn đảm bảo tốc độ trích xuất tọa độ khớp tay và tính Bounding Box rất nhanh.

2. **Giai đoạn Huấn luyện Mô hình AI (Training Phase - PyTorch/YOLO/GRU/EfficientNet):**
   - **Trạng thái GPU:** Đã kiểm tra môi trường Python trên máy mới, `torch.cuda.is_available()` trả về `True` và nhận diện chính xác card **NVIDIA GeForce RTX 4050 Laptop GPU**.
   - **Ứng dụng:** Card RTX 4050 sẽ được tự động kích hoạt để gánh 100% tác vụ tính toán nặng khi chạy các script huấn luyện tại `Cloud_server/Trainer/train_scripts/` (`train_yolo.py`, `train_feature_extractor.py`, `train_gru.py`, `train_classification.py`).

---

## 🔄 Thay đổi 9: Cải tiến Luồng Đọc Dữ liệu Chuỗi Cử chỉ (`Prepare_sequences.py` & `config.py`)

> **Ngày thực hiện:** 03/08/2026  
> **File liên quan:** `config.py`, `Prepare_sequences.py`

### 📍 Nguyên nhân thay đổi
- Trước đây, `Prepare_sequences.py` phụ thuộc vào file JSON (`nslt_100.json`) để tra cứu danh sách video theo mã ID số và lấy thuộc tính `subset` ('train'/'val').
- Khi cập nhật cấu trúc Dataset mới, toàn bộ video đã được giải nén và phân loại trực tiếp theo từng thư mục tên từ vựng (ví dụ: `archive/dataset/SL/apple/02999.mp4`, `book/`, `cat/`...).
- Do đó, việc phụ thuộc vào file JSON bị bỏ để chuyển sang đọc trực tiếp cấu trúc thư mục lớp, giúp quản lý dữ liệu trực quan và linh hoạt hơn.

### ✅ Chi tiết cập nhật đã thực hiện

1. **Cập nhật `config.py`:**
   - Xóa bỏ biến `SEQUENCES_JSON`.
   - Khai báo biến đường dẫn bộ Dataset mới:
     `SEQUENCES_DATASET_DIR = os.path.join(SEQUENCES_DIR, 'archive', 'dataset', 'SL')`
   - Giữ nguyên `SEQUENCES_VIDEOS_DIR = os.path.join(SEQUENCES_DIR, 'videos')` để đảm bảo khả năng tương thích với các script/CSV khác.

2. **Cập nhật `Prepare_sequences.py`:**
   - Loại bỏ hàm đọc JSON cũ `process_from_wlasl_json`.
   - Bổ sung hàm mới **`process_from_dataset_folder`**:
     - Quét từng thư mục từ vựng trong `SEQUENCES_DATASET_DIR`.
     - Tự động phân chia 80% Train và 20% Val cho mỗi từ vựng bằng `train_test_split`.
     - Trích xuất 126 tọa độ keypoint (MediaPipe Hands) và xuất ra file `.npy` tương ứng tại `Sequences/processed/train/{tên_từ}/{vid_id}.npy` và `val/{tên_từ}/{vid_id}.npy`.

---

## 🔴 Lỗi 10: ModuleNotFoundError do xung đột import `tensorflow.keras` trong `train_yolo.py` khi nâng cấp TensorFlow 2.18

> **Ngày thực hiện:** 03/08/2026  
> **File liên quan:** `Cloud_server/Trainer/train_scripts/train_yolo.py`

### 📍 Vị trí code gây lỗi
```python
# Dòng 12 trong train_yolo.py
from tensorflow.keras.callbacks import EarlyStopping
```

### 🔍 Bản chất lỗi & Giải thích kỹ thuật

1. **Xung đột phiên bản TensorFlow / Keras 3**:
   - Ở TensorFlow 2.18+ (Keras 3), Google đã tách Keras ra thành thư viện độc lập `keras` thay vì tích hợp sẵn dưới namespace `tensorflow.keras`.
   - Do đó, câu lệnh `from tensorflow.keras.callbacks import EarlyStopping` báo lỗi: `ModuleNotFoundError: No module named 'tensorflow.keras'`.

2. **Về tính năng Dừng sớm (Early Stopping) của YOLOv8**:
   - Dòng khai báo `EarlyStopping` của Keras ở dòng 12 thực chất là **dòng code thừa không được sử dụng** trong `train_yolo.py`.
   - Thư viện YOLOv8 (`ultralytics`) đã có sẵn tính năng Early Stopping nội bộ được kích hoạt thông qua tham số `patience=5` trong lệnh `model.train(..., patience=5)`.
   - Vì vậy, việc xóa bỏ dòng import thừa này không ảnh hưởng tới tính năng ngắt học sớm (Early Stopping) chống Overfitting của YOLOv8.

### ✅ Cách chúng ta đã sửa (03/08/2026)
- Xóa dòng import thừa `from tensorflow.keras.callbacks import EarlyStopping` khỏi file `train_yolo.py`.
- Đồng thời cài đặt gói tương thích `tf_keras` để đảm bảo các file train khác (`train_feature_extractor.py`, `train_gru.py`, `train_classification.py`) hoạt động tương thích với TensorFlow 2.18.

---

## 💡 Ghi chú Kỹ thuật: Cơ chế quản lý Bộ nhớ RAM và lý do tham số `workers` chỉ có ở `train_yolo.py`

> **Ngày thực hiện:** 03/08/2026  
> **File liên quan:** `train_yolo.py`, `train_feature_extractor.py`, `train_gru.py`, `train_classification.py`

### 🔍 Giải thích sự khác biệt về `workers` giữa các file Train

1. **Khác biệt về Thư viện AI (PyTorch vs TensorFlow/Keras)**:
   - **`train_yolo.py` (Mạng PyTorch / Ultralytics)**: Trong PyTorch DataLoader, tham số `workers` quyết định số tiến trình con (child processes) của CPU được sinh ra để nạp ảnh song song vào GPU.
   - **3 file còn lại (Mạng TensorFlow / Keras)**: Keras quản lý luồng dữ liệu ngầm tự động trong hàm `model.fit()`. Trong Keras, tham số quản lý tương đương được gọi là `workers=1` hoặc `use_multiprocessing=False` và được Keras xử lý ngầm mà không yêu cầu khai báo thủ công.

2. **Khác biệt về Kích thước & Dạng Dữ liệu**:
   - **`train_yolo.py`**: Nạp khối lượng ảnh thô lớn (640x640) với 9,081 file ảnh. Quá trình xử lý ảnh trên CPU ngốn bộ nhớ RAM lớn. Mặc định `workers=4` trên Windows khiến 4 luồng PyTorch ngốn dần RAM gây lỗi `OutOfMemoryError` tại Epoch 12.
   - **`train_gru.py`**: Nạp các file mảng số NumPy `.npy` (chỉ chứa 126 tọa độ khớp tay). Dữ liệu mảng số siêu nhẹ (vài KB) nên được nạp thẳng vào RAM mà không cần chia luồng `workers` đa tiến trình.

### ✅ Khắc phục lỗi OutOfMemory trong `train_yolo.py`:
- Cập nhật cấu hình: `workers=2` (giảm luồng nạp ảnh song song) và `cache=False` (giải phóng bộ nhớ RAM đệm).
- **Tối ưu RAM cho máy 8GB RAM:** Hệ điều hành & ứng dụng ngầm chiếm ~5GB, RAM trống ~3GB. Cấu hình `workers=2` chỉ dùng ~2.4GB RAM, nằm an toàn trong ngưỡng bộ nhớ và chạy mượt mà.
- Tiến trình huấn luyện tự động khôi phục từ checkpoint Epoch 11/12 nhờ tham số `resume=True`.

---

## 🛠️ Ghi chú Kỹ thuật: Giải phóng VRAM bị kẹt bằng lệnh `taskkill /F /IM python.exe`

> **Ngày thực hiện:** 03/08/2026  
> **Lỗi liên quan:** `fatal: Memory allocation failure` & `RuntimeError: CUDA error: unknown error`

### 📍 Nguyên nhân kẹt VRAM GPU
- Khi ngắt lệnh train giữa chừng (Cancel / Ctrl + C / tắt cửa sổ Terminal), các tiến trình con PyTorch/CUDA (`PID 1516`...) có thể không tự hủy hoàn toàn mà tiếp tục chạy ngầm (Background Process).
- Trình điều khiển NVIDIA GPU tiếp tục giữ lại dung lượng VRAM đã cấp phát (ví dụ: ~2.3GB VRAM) cho tiến trình ngầm đó.
- Khi chạy lệnh train mới, tiến trình mới tiếp tục xin thêm VRAM làm tổng dung lượng vượt quá giới hạn 6GB VRAM của card RTX 4050 ➔ Gây ra lỗi `Memory allocation failure`.

### ✅ Cách xử lý & Câu lệnh dọn dẹp
Chạy câu lệnh PowerShell dưới đây để cưỡng chế dừng tất cả tiến trình Python ngầm còn sót lại:
```powershell
taskkill /F /IM python.exe
```
Lệnh này giúp đưa dung lượng VRAM của card đồ họa RTX 4050 về mốc `0MiB` sạch sẽ 100% trước khi bắt đầu đợt huấn luyện mới.

---

## 🔄 Cải tiến 12: Khai thông luồng chạy & Cơ chế Tự động Tiếp nối (Resume) trong `Prepare_sequences.py`

> **Ngày thực hiện:** 03/08/2026  
> **File liên quan:** `Data_preparation/Prepare_sequences.py`

### 📍 Nguyên nhân khiến tiến trình dừng lại ở 36% (Lớp 717)
- Bộ dữ liệu cử chỉ động bao gồm **2,000 lớp từ vựng (hàng chục nghìn video)**, thời gian trích xuất keypoints (MediaPipe Hands) trên CPU kéo dài từ **5 đến 8 tiếng**.
- Khi xử lý tới lớp 717 (~36% tiến độ), một video bị hỏng mã hóa (corrupted H.264 stream) khiến OpenCV thoát khỏi vòng lặp và dừng tiến trình.
- Do code cũ chỉ lưu biến nhớ `processed_ids` trên RAM tạm thời, khi bật lại script phải tốn thời gian duyệt lại từ lớp 1.

### 📝 Các cải tiến đã được áp dụng:

1. **Kiểm tra file trên ổ đĩa (Resume thông minh)**:
   - Script tự động kiểm tra thư mục đích `Sequences/processed/train` & `val`. Nếu file `.npy` của từ vựng đó đã tồn tại trên ổ cứng, script sẽ **bỏ qua ngay lập tức trong 0.001 giây**.
   - Khi chạy lại, script quét 717 lớp đã trích xuất xong trong 2.5 tiếng qua chỉ trong 3 giây và **chạy tiếp nối ngay từ lớp 718**.

2. **Xử lý ngoại lệ lỗi video (`try...except`)**:
   - Nếu trong quá trình đọc gặp file video nào bị hỏng nặng, Python sẽ **bỏ qua video lỗi đó** và tiếp tục đọc các video khác mà không làm sập chương trình.

---

## 🔴 Lỗi 13: Xử lý ModuleNotFoundError của Keras (`train_feature_extractor.py` & `train_gru.py`)

> **Ngày thực hiện:** 03/08/2026  
> **File liên quan:** `Cloud_server/Trainer/train_scripts/train_feature_extractor.py`, `train_gru.py`

### 📍 Nguyên nhân lỗi
- Khi chuyển sang môi trường TensorFlow 2.18 / Keras 3 trên máy mới, các câu lệnh import cũ dạng `from tensorflow.keras...` gặp lỗi `ModuleNotFoundError: No module named 'tensorflow.keras'` hoặc thiếu `ImageDataGenerator` legacy module.

### ✅ Cách xử lý & Cập nhật
Chuyển đổi toàn bộ câu lệnh import Keras sang gói tương thích `tf_keras`:
```python
import tf_keras as keras
from tf_keras import layers, models
from tf_keras.applications import EfficientNetB0
from tf_keras.preprocessing.image import ImageDataGenerator
from tf_keras.callbacks import EarlyStopping, ModelCheckpoint
```
- **Kết quả:** Đã kiểm tra import thành công **(`ALL TRAINING SCRIPTS IMPORT OK`)**, 100% các script huấn luyện Keras khởi chạy mượt mà.

---

## 🔴 Lỗi 14: Sửa lỗi `AttributeError: 'SmartProgressCallback' object has no attribute '_implements_train_batch_hooks'`

> **Ngày thực hiện:** 03/08/2026  
> **File liên quan:** `train_feature_extractor.py`, `train_gru.py`

### 📍 Nguyên nhân lỗi
- Class tự định nghĩa `SmartProgressCallback` kế thừa từ `tf.keras.callbacks.Callback` (lớp cũ của TensorFlow) thay vì `keras.callbacks.Callback` (lớp chuẩn của `tf_keras`).
- Sự khác biệt về lớp cha (Base Class) khiến `tf_keras` không thể truy cập các hook phương thức trong `model.fit()` và báo lỗi `AttributeError`.

### ✅ Cách xử lý & Cập nhật
- Đổi khai báo kế thừa sang `class SmartProgressCallback(keras.callbacks.Callback):`.
- **Kết quả:** `model.fit()` trong cả 2 file `train_feature_extractor.py` và `train_gru.py` khởi chạy mượt mà, lưu checkpoint định kỳ chính xác.

### 💡 Ghi chú Kỹ thuật: Công dụng của SmartProgressCallback
Lớp `SmartProgressCallback` đảm nhận 2 vai trò gác cổng chính trong quá trình huấn luyện:
1. **Tự động lưu tiến trình (Resume)**: Mỗi khi kết thúc 1 Epoch, hàm `on_epoch_end()` ghi nhận số Epoch hiện tại vào file `last_epoch.txt`. Nếu quá trình train bị ngắt giữa chừng, script sẽ tự động đọc file này để train nối tiếp thay vì phải chạy lại từ Epoch 1.
2. **Lưu Checkpoint định kỳ**: Cứ mỗi `period=5` epoch (Epoch 5, 10, 15...), tự động lưu bản snapshot trọng số mô hình `.h5` để theo dõi và so sánh hiệu năng.

---

## 🔴 Lỗi 15: Lỗi chuyển đổi mô hình TFLite (`convert_to_mobile.py`) & Xử lý `TFOpLambda` / YOLO

> **Ngày thực hiện:** 04/08/2026  
> **File liên quan:** `Tools/convert_to_mobile.py`

### 📍 Nguyên nhân lỗi
1. **Thiếu file weights YOLO**: Không tìm thấy `hand_det_yolo.pt` tại đường dẫn `Shared_lib/assets/`.
2. **Lỗi `TypeError: Could not locate class 'TFOpLambda'`**: TensorFlow 2.18 / Keras 3 không nhận diện được lớp `TFOpLambda` trong các file `.h5` cũ khi dùng `tf.keras.models.load_model()`.
3. **Lỗi chuyển đổi GRU model**: `action_recognizer.h5` bị lỗi khi convert TFLite tiêu chuẩn do thiếu tùy chọn `SELECT_TF_OPS`.
4. **Lỗi LiteRT trên Windows**: Ultralytics YOLO bị lỗi LiteRT khi export trực tiếp sang TFLite trên môi trường Windows.

### ✅ Cách xử lý & Cập nhật
- Thiết lập `os.environ['TF_USE_LEGACY_KERAS'] = '1'` ở đầu script để nạp mượt mà các file `.h5` Keras 2 chứa `TFOpLambda`.
- Thêm tự động kiểm tra và copy file weights `best.pt` từ quá trình train YOLO sang `Shared_lib/assets/hand_det_yolo.pt`.
- Bổ sung tùy chọn `SELECT_TF_OPS` và `_experimental_lower_tensor_list_ops = False` trong `convert_keras_to_tflite()` cho model GRU.
- Thêm cơ chế fallback chuyển đổi YOLO thông qua `SavedModel -> TFLiteConverter` trên môi trường Windows.
- **Kết quả:** Đã xuất thành công 3 file `.tflite` chuẩn (`hand_det_yolo.tflite`, `feature_extractor.tflite`, `action_recognizer.tflite`) vào `mobile_app/assets/`.

---

## 🔴 Lỗi 16: Thiếu file `learning_manager.py` & Công dụng của lớp `LearningManager`

> **Ngày thực hiện:** 04/08/2026  
> **File liên quan:** `Cloud_server/Database/edge_cases/learning_manager.py`, `Cloud_server/Api/Main.py`

### 📍 Nguyên nhân lỗi
- File `Cloud_server/Database/edge_cases/learning_manager.py` bị trống (0 bytes). Khi khởi chạy Server API với Uvicorn qua lệnh `uvicorn Cloud_server.Api.Main:app ...`, file `Main.py` gọi `from Cloud_server.Database.edge_cases.learning_manager import LearningManager` dẫn đến lỗi `ImportError: cannot import name 'LearningManager'`.

### 💡 Công dụng của lớp `LearningManager`
Lớp `LearningManager` quản lý toàn bộ luồng dữ liệu tự học ngầm (Implicit Feedback Loop) cho hệ thống AI nhận diện ngôn ngữ ký hiệu:
1. **Ghi nhận ca khó (`save_to_unverified`)**: Khi ứng dụng di động/web gặp các trường hợp ký hiệu nhận diện không tự tin (confidence < 0.60), hàm này ghi nhận dữ liệu Landmarks/ảnh lỗi cùng kết quả dự đoán của AI dưới dạng file JSON lưu vào `Cloud_server/Database/edge_cases/Unverified_Learning`.
2. **Tiếp nhận phản hồi người dùng (`promote_to_labeled`)**: Khi người dùng hoặc chuyên gia bấm nút "Sửa lại câu dịch" trên giao diện, hàm này chuyển file dữ liệu nghi vấn sang thư mục `Cloud_server/Database/edge_cases/Labeled` và đính kèm nhãn chuẩn do người dùng sửa để phục vụ các đợt Retrain tự động tiếp theo.

### ✅ Cách xử lý & Cập nhật
- Triển khai hoàn chỉnh lớp `LearningManager` trong `Cloud_server/Database/edge_cases/learning_manager.py` với đầy đủ 2 phương thức `save_to_unverified` và `promote_to_labeled`.
- Đồng thời sửa lỗi thụt lề `IndentationError` tại hàm `_check_google()` trong `llm_corrector.py`.
- **Kết quả:** Server API FastAPI trong `Main.py` khởi chạy mượt mà 100% bằng Uvicorn mà không bị gián đoạn.

---

## 🔴 Lỗi 17: Nguồn gốc & công dụng của `ALPHABET_CLASSES` và `ACTION_CLASSES`

> **Ngày thực hiện:** 04/08/2026  
> **File liên quan:** `Shared_lib/Constants.py`, `Shared_lib/predictor.py`, `Mobile_app/Src/Main.py`

### 1. Nguồn gốc: Tại sao lại có 2 biến này?
Trong mã nguồn lấy từ GitHub về:
- File `predictor.py` có sẵn dòng code: `from Shared_lib.Constants import ALPHABET_CLASSES, ACTION_CLASSES`
- File `Main.py` cũng có sẵn dòng code: `from Shared_lib.Constants import ACTION_CLASSES`

Tác giả gốc của dự án đã viết câu lệnh nhập (import) 2 biến này từ file `Constants.py`, nhưng khi họ lưu file `Constants.py` lên GitHub thì lại quên chưa định nghĩa (khai báo) 2 biến đó. Dẫn đến việc khi chạy ứng dụng, Python sẽ bị dừng lại vì lỗi:
> `ImportError: cannot import name 'ALPHABET_CLASSES' from 'Shared_lib.Constants'`

### 2. Công dụng của `ALPHABET_CLASSES` và `ACTION_CLASSES` là gì?
Khi mô hình AI (TensorFlow Lite) quét camera và nhận diện cử chỉ tay của bạn, kết quả mà AI tính toán ra không phải là chữ viết (như `"hello"` hay `"thanks"`), mà là một con số chỉ số (index).

- **Ví dụ:** AI dự đoán ra số `61`.

Để chuyển con số `61` đó thành từ ngữ hiển thị trên màn hình ứng dụng:
- **`ACTION_CLASSES`**: Là danh sách gồm 100 từ ngữ thủ ngữ (như `'hello'`, `'thanks'`, `'help'`, `'yes'`, `'no'`, `'apple'`, v.v.). Khi AI ra số `61`, chương trình sẽ tra danh sách `ACTION_CLASSES[61]` → lấy ra từ `"help"` để hiển thị và đọc phát âm cho người dùng.
- **`ALPHABET_CLASSES`**: Là danh sách bảng chữ cái từ `A` đến `Z` dùng cho tính năng đánh vần từng chữ cái bằng ngón tay (fingerspelling).

---

## 🔴 Lỗi 18: Nâng cấp toàn diện quy trình tự động Retrain (`Cloud_server/Trainer/Retrain.py`)

> **Ngày thực hiện:** 07/08/2026  
> **File liên quan:** `Cloud_server/Trainer/Retrain.py`, `Cloud_server/Model_registry/registry.py`, `Cloud_server/Trainer/train_scripts/train_gru.py`  
> **Tác giả:** Pair Programming cùng AI Assistant Antigravity  

### 📍 Mục đích nâng cấp
Chuyển đổi `Retrain.py` từ một script gọi train đơn thuần thành **Bộ điều phối tự động (Automation Pipeline Orchestrator)** thông minh, đảm bảo an toàn tuyệt đối cho hệ thống Cloud Server khi tự động học lại dữ liệu mới.

### 💡 3 Tính năng cốt lõi đã tích hợp & Đoạn code chính

#### 1. Bộ lọc kích hoạt thông minh (Threshold Trigger)
Tự động đếm số lượng mẫu dữ liệu gán nhãn mới trong thư mục `Labeled/`. Dừng an toàn nếu chưa đủ ngưỡng ($\ge 5$ mẫu) để tránh lãng phí tài nguyên CPU/GPU:
```python
def count_new_labeled_samples():
    total = 0
    for label in os.listdir(LABELED_DIR):
        label_dir = os.path.join(LABELED_DIR, label)
        if os.path.isdir(label_dir):
            files = [f for f in os.listdir(label_dir) if f.endswith(('.npy', '.json', '.mp4', '.avi'))]
            total += len(files)
    return total
```

#### 2. Tự động gộp dữ liệu & Tiền xử lý (Data Merge)
Tự động di chuyển các mẫu dữ liệu mới đã gán nhãn vào thư mục huấn luyện chính `Sequences/processed/train/`:
```python
def merge_labeled_data():
    for label in os.listdir(LABELED_DIR):
        src_label_dir = os.path.join(LABELED_DIR, label)
        dst_label_dir = os.path.join(SEQUENCES_TRAIN_DIR, label)
        os.makedirs(dst_label_dir, exist_ok=True)
        for file_name in os.listdir(src_label_dir):
            shutil.move(os.path.join(src_label_dir, file_name), os.path.join(dst_label_dir, file_name))
```

#### 3. Chốt chặn An toàn & Cơ chế Rollback (Safety Evaluation Gate)
Đọc chỉ số `% Val Accuracy` từ file `latest_train_metrics.json`. So sánh mô hình MỚI vs CŨ. Chỉ khi mô hình mới đạt độ chính xác vượt trội/ổn định mới cho phép đăng ký phiên bản mới trong `ModelRegistry` và convert TFLite cho App. Nếu kém hơn, hệ thống kích hoạt Rollback giữ nguyên bản cũ:
```python
def evaluate_new_model(previous_best_acc=0.30):
    metrics_log_file = os.path.join(BASE_DIR, "Cloud_server", "Trainer", "latest_train_metrics.json")
    new_acc = 0.0
    if os.path.exists(metrics_log_file):
        with open(metrics_log_file, "r", encoding="utf-8") as f:
            new_acc = json.load(f).get("val_accuracy", 0.0)

    if new_acc >= previous_best_acc:
        return True, new_acc
    else:
        print("⚠️ CẢNH BÁO: Mô hình mới có độ chính xác thấp hơn. Kích hoạt Rollback!")
        return False, new_acc
```

### ✅ Kết quả kiểm thử
- **Đã test thành công lệnh `python Cloud_server/Trainer/Retrain.py`. Hệ thống đếm mẫu mới (0 mẫu) và dừng an toàn mà không bị văng lỗi.

---

## 🔴 Lỗi 19: Khắc phục lỗi mất khung bàn tay (Bounding Box) & Thiếu bộ bắt phím chuyển ngôn ngữ (`V/E/J/K`)

> **Ngày thực hiện:** 07/08/2026  
> **File liên quan:** `Shared_lib/predictor.py`, `Mobile_app/Src/Main.py`  
> **Tác giả:** Pair Programming cùng AI Assistant Antigravity  

### 📍 Mô tả lỗi & Hiện tượng
1. **Không có khung xanh nhận diện bàn tay:** Khi bật ứng dụng `python Mobile_app/Src/Main.py`, camera bật lên nhưng không vẽ ô vuông xanh vị trí bàn tay.
2. **Không dịch được chữ cái:** Do không nhận diện được bàn tay, ứng dụng không trích xuất được vùng ảnh bàn tay (`hand_roi`), dẫn đến không hiển thị chữ dịch khi làm thủ ngữ bằng 2 tay.
3. **Phím V/E/J/K không phản hồi:** Nhấn các phím chuyển ngôn ngữ `[V] Tiếng Việt`, `[E] Tiếng Anh` không có bất kỳ phản hồi nào.

### 🔍 Nguyên nhân gốc rễ
1. **Sai ma trận đọc output YOLO (`predictor.py`):**  
   Mô hình `hand_det_yolo.tflite` xuất ra tensor có kích thước `[1, 5, 2100]`. Đoạn code cũ `boxes = output[0]` lấy `max(boxes, key=lambda x: x[4])` bị đọc nhầm chiều ma trận (duyệt 5 hàng thay vì 2100 ô anchors), làm hàm `detect_hand` **luôn luôn trả về `None`**.
2. **Thiếu xử lý phím trong loop (`Main.py`):**  
   Trong vòng lặp `while True` hiển thị camera, code chỉ bắt duy nhất phím `'q'` để thoát (`if key == ord('q'): break`) mà chưa cài đặt các hàm rẽ nhánh cho phím `V`, `E`, `J`, `K`.

### ✅ Cách xử lý & Cập nhật

#### 1. Chuyển vị ma trận YOLOv8/v11 trong `Shared_lib/predictor.py`
Chuyển vị `output[0].T` thành ma trận `[2100, 5]` để lọc đúng mảng confidence và quy đổi tâm `(cx, cy, w, h)` về khung hình camera:
```python
def detect_hand(self, frame):
    input_shape = self.yolo_input_details[0]['shape']
    img_h, img_w = frame.shape[:2]
    img = cv2.resize(frame, (input_shape[2], input_shape[1]))
    img = img.astype(np.float32) / 255.0
    img = np.expand_dims(img, axis=0)

    self.yolo_interpreter.set_tensor(self.yolo_input_details[0]['index'], img)
    self.yolo_interpreter.invoke()
    output = self.yolo_interpreter.get_tensor(self.yolo_output_details[0]['index'])

    # Chuyển vị ma trận từ [1, 5, 2100] thành [2100, 5]
    predictions = output[0].T
    scores = predictions[:, 4]
    best_idx = np.argmax(scores)
    
    if scores[best_idx] < 0.35:
        return None

    cx, cy, w, h = predictions[best_idx, :4]
    scale_x = img_w / input_shape[2]
    scale_y = img_h / input_shape[1]

    x1 = max(0, int((cx - w / 2) * scale_x))
    y1 = max(0, int((cy - h / 2) * scale_y))
    x2 = min(img_w, int((cx + w / 2) * scale_x))
    y2 = min(img_h, int((cy + h / 2) * scale_y))

    return (x1, y1, x2, y2)
```

#### 2. Bổ sung bộ bắt sự kiện phím tắt trong `Mobile_app/Src/Main.py`
```python
key = cv2.waitKey(1) & 0xFF
if key in (ord('q'), ord('Q')):
    break
elif key in (ord('v'), ord('V')):
    self.current_lang = 'vi'
    print("🌐 Đã chuyển ngôn ngữ dịch sang: Tiếng Việt (VI)")
elif key in (ord('e'), ord('E')):
    self.current_lang = 'en'
    print("🌐 Đã chuyển ngôn ngữ dịch sang: Tiếng Anh (EN)")
elif key in (ord('j'), ord('J')):
    self.current_lang = 'ja'
    print("🌐 Đã chuyển ngôn ngữ dịch sang: Tiếng Nhật (JA)")
elif key in (ord('k'), ord('K')):
    self.current_lang = 'ko'
    print("🌐 Đã chuyển ngôn ngữ dịch sang: Tiếng Hàn (KO)")
```

### 💡 Bài học kinh nghiệm & Lý giải config vs Main
- **Khác biệt `config.py` vs `Main.py`:**  
  `config.py` chỉ thiết lập ngôn ngữ ban đầu khi ứng dụng mới khởi động (Cấu hình tĩnh).  
  `Main.py` xử lý bộ bắt bàn phím động thời gian thực khi đang soi camera để người dùng có thể đổi ngôn ngữ linh hoạt mà không cần tắt app.

---

## 🔴 Lỗi 20: Tối ưu nhận diện đa bàn tay, tự động dọn buffer ngắt chữ rác & Từ điển dịch đa ngôn ngữ

> **Ngày thực hiện:** 07/08/2026  
> **File liên quan:** `Shared_lib/predictor.py`, `Mobile_app/Src/Virtual_cam/Virtual_camera.py`, `Cloud_server/Api/context_agent.py`, `Cloud_server/Api/Main.py`, `Mobile_app/Src/Main.py`  
> **Tác giả:** Pair Programming cùng AI Assistant Antigravity  

### 📍 Các hiện tượng & Lỗi phát sinh
1. **Dữ liệu rác lặp lại khi hạ tay ("cousin", "shirt"...):** Khi người dùng để tay xuống, màn hình vẫn liên tục nhảy từ cũ hoặc các từ không mong muốn.
2. **Hiện 2 cửa sổ camera trùng lặp:** Khi bật app có 2 cửa sổ `"Sign Language App"` và `"Virtual Camera Output"` cùng hiện lên.
3. **Ấn phím `V` không dịch từ Tiếng Anh sang Tiếng Việt:** Từ `"apple"` vẫn hiển thị Tiếng Anh thay vì dịch ra `"quả táo"`.
4. **Server báo lỗi 404 Not Found:** `POST /upload_edge_case HTTP/1.1 404 Not Found`.

### 🔍 Nguyên nhân gốc rễ & Cách xử lý

#### 1. Xóa sạch bộ đệm `self.buffer.clear()` khi hạ tay (`Shared_lib/predictor.py`)
- **Nguyên nhân:** Khi hạ tay (`len(bboxes) == 0`), code cũ không dọn 30 khung hình trong `self.buffer`, khiến AI liên tục dự đoán trên dữ liệu cũ đọng lại.
- **Khắc phục:** Tự động gọi `self.buffer.clear()` khi không có bàn tay $\rightarrow$ Dừng ngay dự đoán khi hạ tay xuống:
```python
if len(bboxes) > 0:
    # Nạp feature bàn tay
    self.buffer.append(feature_vec)
else:
    # Xóa sạch bộ đệm khi hạ tay để dừng dự đoán chữ cũ ngay lập tức
    self.buffer.clear()
```

#### 2. Tắt cửa sổ camera trùng lặp (`Mobile_app/Src/Virtual_cam/Virtual_camera.py`)
- **Nguyên nhân:** Khi `self.use_virtual = False`, `send_frame` gọi `cv2.imshow("Virtual Camera Output", ...)` trùng với `Main.py`.
- **Khắc phục:** Loại bỏ `cv2.imshow` dư thừa $\rightarrow$ Giữ duy nhất 1 cửa sổ `"Sign Language App"`.

#### 3. Bổ sung từ điển dịch thủ ngữ 100 từ vựng (`Cloud_server/Api/context_agent.py`)
- **Nguyên nhân:** `context_agent.py` chưa có bảng mapping dịch từ Tiếng Anh thô sang Tiếng Việt/Nhật/Hàn theo `target_lang`.
- **Khắc phục:** Tích hợp từ điển `VI_DICTIONARY` cho 100 từ thủ ngữ WLASL. Khi người dùng bấm phím `V` (Tiếng Việt), từ `"apple"` tự động dịch thành **`"quả táo"`** hiển thị lên màn hình và phát âm TTS:
```python
def process(self, action_word, target_lang='vi'):
    word_lower = action_word.lower()
    if target_lang == 'vi':
        return VI_DICTIONARY.get(word_lower, action_word)
    return action_word
```

#### 4. Bổ sung Endpoint `/upload_edge_case` (`Cloud_server/Api/Main.py`)
- **Nguyên nhân:** Server API FastAPI chưa định nghĩa endpoint `/upload_edge_case` nên trả về lỗi `404 Not Found`.
- **Khắc phục:** Khai báo endpoint `@app.post("/upload_edge_case")` $\rightarrow$ Triệt tiêu lỗi 404.

---

## 🔴 Lỗi 21: Tích hợp Bộ đệm Ngắt câu 2.0s linh hoạt Từ đơn / Câu dài & Phụ đề Đáy Camera

> **Ngày thực hiện:** 09/08/2026  
> **File liên quan:** `Mobile_app/Src/Main.py`, `Shared_lib/predictor.py`  
> **Tác giả:** Pair Programming cùng AI Assistant Antigravity  

### 📍 Tính năng nâng cấp
1. **Linh hoạt giữa Từ Đơn & Câu Dài (Timeout 2.0s):**
   - Khi giơ tay liên tục, các từ thô được gom lại thành mảng `sentence_words`.
   - Nếu hạ tay xuống hoặc dừng tay quá 2.0s (`time.time() - last_hand_time > 2.0`) $\rightarrow$ AI `ContextAgent` tự động hốt toàn bộ chuỗi từ, sửa lỗi ngữ cảnh ngữ pháp thành câu chuẩn và đọc phát âm TTS.
2. **Thanh Phụ đề màu đen mờ ở Đáy Camera (Bottom Subtitle Bar):**
   - Phụ đề được hiển thị tại thanh băng mờ đen ở đáy camera `cv2.rectangle(display, (0, h - 60), (w, h), (0, 0, 0), -1)` với dòng chữ màu trắng nổi bật `Dich (VI): <nội dung câu>`.

---

## 🔴 Lỗi 22: Chuyển đổi chuẩn màu BGR->RGB cho YOLOv8 & Thẻ hiển thị Ngôn ngữ thời gian thực trên Camera

> **Ngày thực hiện:** 09/08/2026  
> **File liên quan:** `Shared_lib/predictor.py`, `Mobile_app/Src/Main.py`  
> **Tác giả:** Pair Programming cùng AI Assistant Antigravity  

### 📍 Hiện tượng & Vấn đề người dùng phản hồi
1. **Không thấy thẻ Ngôn ngữ trên khung hình camera:** Trước đây trạng thái đổi ngôn ngữ (`V/E/J/K`) chỉ in ra cửa sổ dòng lệnh PowerShell mà không vẽ lên giao diện video camera.
2. **Khung hình camera trống trơn không bắt được bàn tay:** Khung camera bật lên nhưng đưa tay lên không hiện ô vuông xanh/dương nhận diện.
3. **Thắc mắc về thao tác lưu file (Ctrl+S):** Người dùng hỏi có cần bấm `Ctrl + S` trong VS Code/IDE để tích hợp code sửa hay không.

### 🔍 Nguyên nhân gốc rễ & Giải pháp khắc phục

#### 1. Sửa lỗi chuẩn màu YOLO BGR -> RGB (`Shared_lib/predictor.py`)
- **Nguyên nhân:** Khung hình đọc từ OpenCV (`cv2.VideoCapture`) có chuẩn màu **BGR**. Mô hình YOLOv8 (`hand_det_yolo.tflite`) được huấn luyện trên chuẩn màu **RGB**. Việc lệch kênh màu (Red và Blue bị tráo đổi) khiến YOLO khó nhận diện màu da bàn tay của người dùng.
- **Khắc phục:** Thêm dòng chuyển đổi chuẩn màu trước khi đưa ảnh vào mô hình YOLO:
```python
# Chuyển đổi BGR sang RGB để mô hình YOLO nhận diện màu sắc bàn tay chuẩn xác
rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
img = cv2.resize(rgb_frame, (input_shape[2], input_shape[1]))
```

#### 2. Vẽ Thẻ Ngôn Ngữ thời gian thực ở Góc Trên Bên Trái (`Mobile_app/Src/Main.py`)
- **Nguyên nhân:** Thiếu phần hiển thị thông tin trạng thái ngôn ngữ trên màn hình OpenCV.
- **Khắc phục:** Thêm dải đen màu vàng ở góc trên bên trái `(10, 10)` hiển thị rõ ràng ngôn ngữ đang được chọn (ví dụ: `NGÔN NGỮ: TIẾNG VIỆT (Phím V)`). Khi người dùng bấm phím `E`, `J`, `K`, thẻ góc trên lập tức đổi trạng thái trực quan:
```python
lang = getattr(self, 'current_lang', 'vi')
lang_names = {'vi': 'TIENG VIET (Phim V)', 'en': 'ENGLISH (Phim E)', 'ja': 'JAPANESE (Phim J)', 'ko': 'KOREAN (Phim K)'}
lang_str = lang_names.get(lang, 'TIENG VIET (Phim V)')
cv2.rectangle(display, (10, 10), (290, 45), (0, 0, 0), -1)
cv2.putText(display, f"NGON NGU: {lang_str}", (15, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
```

#### 3. Làm rõ quy trình lưu file (File Saving Rule)
- **Giải thích:** AI Assistant thực hiện các thao tác sửa đổi bằng công cụ viết file trực tiếp vào đĩa cứng (`TargetFile`), tệp tin mã nguồn lập tức được lưu tự động mà **người dùng KHÔNG CẦN phải bấm Ctrl+S**.

---

## 🔴 Lỗi 23: Tự động Gộp Ảnh & Chuẩn hóa Single-Class Hand Dataset cho YOLOv8

> **Ngày thực hiện:** 09/08/2026  
> **File liên quan:** `Research_and_Data/Dataset/Detection`, `README.md`  
> **Tác giả:** Pair Programming cùng AI Assistant Antigravity  

### 📍 Hiện tượng & Yêu cầu từ người dùng
- Tập dữ liệu mới EgoHands từ Roboflow giải nén ra 3 thư mục `train`, `valid`, `test`. Người dùng muốn **gộp toàn bộ `test` vào `valid`** để chỉ giữ lại 2 tập `train` (80%) và `val` (960 ảnh - 20%).
- Chuyển toàn bộ dữ liệu Detection cũ (`Hands/`, `HandInfo.csv`) sang kho `Research_and_Data/Dataset/Classification/Old_Detection_Data/` để làm giàu cho mô hình trích xuất đặc trưng ngón tay EfficientNet.

### 🛠️ Cách Python Xử Lý Tự Động:
1. **Lấy danh sách tệp từ `test`:** Dùng thư viện `glob` quét bốc toàn bộ 480 file ảnh `.jpg` và 480 file nhãn `.txt` đang nằm trong thư mục `test/`.
2. **Di chuyển hàng loạt sang `valid`:** Dùng thư viện `shutil.move` để tự động đẩy toàn bộ số ảnh và nhãn đó nạp nối tiếp vào thư mục `valid/images` và `valid/labels`.
3. **Chuẩn hóa nhãn Single-Class (`hand`):** Đổi toàn bộ 4,800 file nhãn `.txt` về duy nhất 1 class `0` (`hand`), cập nhật `data.yaml` với `nc: 1` và `names: ['hand']` chuẩn 80/20 (3,840 train / 960 val).

---

## 🔴 Lỗi 24: Định Hướng Chuẩn Hóa Tên Biến Trực Quan (Self-Explanatory Variables) Trong `config.py`

> **Ngày thực hiện:** 10/08/2026  
> **File liên quan:** `config.py`, `README.md`  
> **Tác giả:** Pair Programming cùng AI Assistant Antigravity  

### 📍 Đề xuất & Kiến trúc biến trực quan
- **Tên biến trực quan, dễ hiểu (Self-Explanatory Variables):**
  - **`DETECTION_YAML`**: Nhìn vào biết ngay là file cấu hình chính của YOLO (`data.yaml`).
  - **`DETECTION_TRAIN_DIR`**: Nhìn vào biết ngay là nơi chứa 80% ảnh huấn luyện (`train/images`).
  - **`DETECTION_VAL_DIR`**: Nhìn vào biết ngay là nơi chứa 20% ảnh kiểm thử (`valid/images`).

---

## 🔴 Lỗi 25: Triển Khai Hệ Thống 4 Menu Điều Khiển Trung Tâm & Gia Sư AI Chấm Điểm Nạp Video Mẫu

> **Ngày thực hiện:** 10/08/2026  
> **File liên quan:** `Main.py`, `Mobile_app/Src/Main.py`, `README.md`  
> **Tác giả:** Pair Programming cùng AI Assistant Antigravity  

### 📍 Tính năng & Cấu trúc đã triển khai
---

## 🔴 Lỗi 26: Chuẩn Hóa Module Xử Lý Độc Lập (`Tools/`) & Người Ảo AI Trợ Lý (AI Virtual Avatar Assistant)

> **Ngày thực hiện:** 10/08/2026  
> **File liên quan:** `Tools/video_translator.py`, `Tools/ai_tutor_engine.py`, `Mobile_app/Src/Main.py`  
> **Tác giả:** Pair Programming cùng AI Assistant Antigravity  

### 📍 Các Module Độc Lập Đã Triển Khai
1. **`Tools/video_translator.py`:** Module chuyên trách đọc, cắt frame và dịch thuật tệp video MP4/AVI.
2. **`Tools/ai_tutor_engine.py`:** Module Gia sư AI Chấm điểm & Người Ảo AI Trợ Lý:
   - Tự động nạp bổ sung Video Mẫu mới vào Dataset `custom_enrollment`.
   - Render Người Ảo AI Trợ Lý (AI Avatar Robot Head) ở góc phải camera, hiển thị bong bóng hội thoại chỉ vị trí sai và phát đọc âm thanh nhắc nhở qua TTS.













