# 🛠️ Tóm Tắt Khắc Phục Lỗi Data Leakage (Train/Val Split) & Đồng Bộ Config

Tài liệu tóm tắt phân tích nguyên nhân và giải pháp triệt tiêu hiện tượng rò rỉ dữ liệu (Data Leakage) khi chia tập Train/Val cho mô hình chuỗi thời gian BiGRU.

---

## 📌 1. Phân Tích Cấu Hình Danh Sách Lớp (Classes)
* **Quy mô nhãn:** Dù tập WLASL gốc có 2,000 từ vựng, hệ thống cấu hình `config.SEQUENCES_JSON = 'nslt_100.json'` để tập trung học sâu Top 100 từ vựng giàu mẫu nhất.
* **Dữ liệu CSV tự quay:** Độc lập hóa khai báo `SEQUENCES_CSV` trong `config.py` để nạp song song cả tập dữ liệu chuẩn WLASL lẫn các file gán nhãn tùy chỉnh (`hand_gestures.csv`).

---

## 📌 2. Phân Tích & Khắc Phục Lỗi Rò Rỉ Dữ Liệu (Data Leakage)

### 🔴 Nguyên nhân gây lỗi Data Leakage cũ:
* Lệnh `train_test_split` chia ngẫu nhiên trên danh sách phẳng làm các video của cùng một người quay (hoặc cùng 1 bối cảnh/quần áo) bị trộn lẫn vào cả tập Train và tập Val.
* **Hậu quả:** Mô hình "học vẹt" bối cảnh căn phòng thay vì học cử chỉ thực sự. Điểm chính xác tập Val ảo rất cao nhưng thất bại khi nhận diện người mới thực tế (Overfitting).

### ✅ Giải pháp triệt tiêu rò rỉ dữ liệu:
1. **Đối với dữ liệu WLASL (JSON):** Bỏ chia ngẫu nhiên, đọc trực tiếp thuộc tính `subset` (`train` / `val`) được chia sẵn chuẩn mực theo chuyên gia (Signer-Independent Split).
2. **Đối với dữ liệu tự quay (CSV):** Chuyển đổi dữ liệu (Melt) và sử dụng thuật toán `GroupShuffleSplit` gom nhóm theo `group_id = set_id`. Đảm bảo toàn bộ video của 1 lượt quay sẽ đi 100% vào tập Train hoặc 100% vào tập Val.

---

## 📌 3. Kết Luận Đạt Được
* Xử lý độc lập, song song nhiều nguồn dữ liệu (WLASL JSON & CSV tự quay).
* Phản ánh đúng 100% độ thông minh thực sự của mô hình BiGRU khi gặp người dùng mới.
