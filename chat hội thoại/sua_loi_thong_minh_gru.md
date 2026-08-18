# 🛠️ Tóm Tắt Phân Tích & Nâng Cấp Độ Thông Minh Mô Hình BiGRU

Tài liệu tóm tắt các điểm nâng cấp thuật toán, chuẩn hóa tọa độ hình học 3D và fine-tune mô hình nhận diện cử chỉ thủ ngữ BiGRU.

---

## 📌 1. Bối Cảnh Cũ & 4 Nguyên Nhân Kẹt Độ Chính Xác (Val Acc 9.8%)
* **Triệu chứng:** Model bị kẹt Loss ~5.45, Val Acc dậm chân ở 9.8%, 1 epoch tốn tới 60 giây.
* **4 nguyên nhân gốc rễ:**
  1. **Bài toán bất khả thi:** 2,000 từ vựng nhưng chỉ có 4.6 video mẫu/từ $\rightarrow$ Không đủ dữ liệu cho AI học.
  2. **Lỗi trừ cổ tay lẻ tẻ từng frame (`normalize_keypoints`):** Mỗi frame $t$ tự trừ cổ tay của chính frame $t$ đó $\rightarrow$ Làm điểm cổ tay đứng yên tại $(0,0,0)$, xóa sạch quỹ đạo di chuyển và khoảng cách tương quan 2 tay!
  3. **Lật gương tráo tay (Mirror Swap x7):** Lật ngược tay trái/phải làm biến đổi cử chỉ sai thực tế, nhân dữ liệu phình to gấp 7 lần gây chậm 60s/epoch.
  4. **Nghẽn cổ chai kiến trúc (Bottleneck):** Bộ nén BiGRU 128 chiều quá hẹp trong khi lớp Dense 2,000 quá to (>75% tham số) gây overfit nặng.

---

## 📌 2. Các Bước Nâng Cấp Đột Phá Trên `train_gru.py`

1. **Lọc từ vựng giàu mẫu (`min_samples >= 6`):** Loại bỏ toàn bộ từ mỏng 1-2 mẫu, giữ lại Top từ vựng chất lượng cao đạt $\ge 6-12$ mẫu train/lớp.
2. **Chuẩn hóa Anchor Cố định (Bảo toàn Quỹ đạo 3D):** Lấy cổ tay tại Frame đầu tiên ($t_{\text{first}}$) làm gốc Anchor $(0,0,0)$ cố định cho toàn bộ 30 frame $\rightarrow$ Bảo toàn 100% quỹ đạo tịnh tiến và khoảng cách giữa 2 tay.
3. **Data Augmentation x4 Copies:** Bỏ lật gương, giữ Jitter nhiễu nhẹ (0.005), Scale (0.9-1.1), Time Warp với `num_copies = 4` $\implies$ Ép thời gian train 1 epoch từ **60s xuống còn ~5.5s**!
4. **Quy trình 2-Stage Fine-Tuning:**
   - Stage 1: Train $lr = 0.001$ dựng khung trọng số.
   - Stage 2: Fine-tune $lr = 0.0001 \rightarrow 10^{-5}$ tinh chỉnh sai số.

---

## 📈 3. Bảng Kết Quả Thực Nghiệm

| Cấu hình Mô hình | Min Samples | Tốc độ Train | Train Acc | Val Acc (Top-1) | Top-5 Acc | Đánh giá Trạng thái |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Mô hình Gốc (2,000 từ)** | Không lọc | ~60s / epoch | 31.50% | **9.84%** | < 15.0% | Overfit nặng. |
| **Mô hình Top 30 từ** | `min_samples >= 8` | ~1.5s / epoch | 89.94% | **43.62%** 🔥 | **~88.5%** | Kỷ lục mô hình Top 30! |
| **Mô hình Top 50 từ** | `min_samples >= 6` | ~3.5s / epoch | 88.37% | **39.86%** 🔥 | **~84.5%** | **Điểm ngọt (Sweet Spot)!** |
| **Fine-Tune Top 100 từ** | `min_samples >= 6` | ~5.5s / epoch | 88.06% | **33.09%** 🎯 | **~76.5%** | Đã lưu `action_recognizer.h5`. |
