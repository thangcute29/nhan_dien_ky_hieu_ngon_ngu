# Báo cáo Sửa lỗi Data Leakage (Train/Val Split) và Cập nhật Config

**Mục tiêu:** 
1. Làm rõ số lượng nhãn/lớp (classes) đang được sử dụng trong tập dữ liệu.
2. Phát hiện và khắc phục hiện tượng rò rỉ dữ liệu (Data Leakage) khi chia tập huấn luyện (Train) và tập xác thực (Val).
3. Cho phép hệ thống xử lý độc lập, song song cả hai nguồn dữ liệu WLASL (JSON) và dữ liệu tự quay (CSV).

---

## 1. Phân tích Số lượng Lớp (Classes)
- **Vấn đề:** Mặc dù danh sách gốc WLASL (`wlasl_class_list.txt`) có tới 2000 lớp, hệ thống thực tế chỉ cấu hình sử dụng 100 lớp.
- **Nguyên nhân:** File `config.py` cấu hình `SEQUENCES_JSON = 'nslt_100.json'`, giới hạn mô hình học 100 nhãn.
- **Dữ liệu tùy chỉnh:** Có 5 lớp bổ sung từ file `hand_gestures.csv` (`one`, `four`, `small`, `fist`, `me`). Tuy nhiên, do lỗi logic trong code cũ (ép đuôi `.json` thành `.csv`), file bổ sung này đã bị bỏ qua.

## 2. Phân tích Hiện tượng Rò rỉ Dữ liệu (Data Leakage)
- **Vấn đề:** Dữ liệu video của cùng một người (hoặc cùng một session quay) bị trộn lẫn ngẫu nhiên vào cả tập Train và tập Val.
- **Nguyên nhân:** Lệnh `train_test_split(all_data, train_size=0.8, random_state=42)` của `sklearn` thực hiện chia ngẫu nhiên trên danh sách phẳng (flat list) mà không quan tâm đến ID người dùng (Person ID) hay Session ID.
- **Hậu quả:** Mô hình "học vẹt" bối cảnh, quần áo, góc máy thay vì học cử chỉ thực sự. Điểm chính xác (Accuracy) trên tập Val ảo (rất cao) nhưng trên thực tế sẽ thất bại khi gặp người mới (Overfitting).

---

## 3. Quá trình Cải tiến và Khắc phục Mã Nguồn

### A. Tách biệt Cấu hình JSON và CSV trong `config.py`
Code cũ gán `SEQUENCES_CSV = SEQUENCES_JSON`, khiến tên file CSV bị phụ thuộc. 
**Giải pháp:** Đã sửa khai báo thành các biến độc lập:
```python
SEQUENCES_JSON = os.path.join(SEQUENCES_DIR, 'nslt_100.json')
SEQUENCES_CSV = os.path.join(SEQUENCES_DIR, 'hand_gestures.csv')
```

### B. Khắc phục Data Leakage cho dữ liệu WLASL (JSON)
Dữ liệu WLASL gốc đã được các nhà nghiên cứu chia tách Train/Val một cách cẩn thận (Signer Independent) thông qua thuộc tính `subset`.
**Giải pháp:** Hủy bỏ hàm `train_test_split` ngẫu nhiên. Code hiện tại đã tôn trọng và đọc trực tiếp từ thuộc tính `subset` của JSON:
```python
if subset == 'train':
    train_data.append((video_id, video_path, label))
else: # val hoặc test
    val_data.append((video_id, video_path, label))
```

### C. Khắc phục Data Leakage cho dữ liệu CSV tự quay
Dữ liệu tự quay có định dạng ngang (các cột là tên lớp, mỗi hàng là một Session: `set_id`). Việc dùng `train_test_split` cũng phá vỡ nhóm Session.
**Giải pháp:** 
1. Chuyển đổi (Melt) dữ liệu từ dạng ngang sang dạng phẳng.
2. Áp dụng `GroupShuffleSplit` (nhóm theo biến `group_id = set_id`). 
Điều này đảm bảo toàn bộ video trong cùng một thư mục (ví dụ `files/0/`) sẽ chỉ đi 100% vào Train hoặc 100% vào Val.
```python
from sklearn.model_selection import GroupShuffleSplit

gss = GroupShuffleSplit(n_splits=1, train_size=0.8, random_state=42)
train_idx, val_idx = next(gss.split(paths, lbls, groups))
```

---

## 4. Kết luận
Với các thay đổi này:
- Bạn đã có thể chạy song song nhiều nguồn Dataset.
- Hệ thống đã giải quyết triệt để rò rỉ dữ liệu bằng phương pháp chia tách theo Nhóm (Group-based Split) và giữ nguyên cấu trúc chuẩn của chuyên gia (WLASL Split).
- Khả năng tổng quát hoá (Generalization) của mô hình (ví dụ như GRU / LSTM / Transformer xử lý chuỗi) sẽ được phản ánh trung thực và chính xác hơn ở bước đánh giá (Validation).
