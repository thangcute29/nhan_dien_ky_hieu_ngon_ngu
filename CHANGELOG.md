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
