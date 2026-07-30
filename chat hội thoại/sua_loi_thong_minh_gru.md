# 🧠 Sửa Lỗi Thông Minh GRU & Nâng Cấp Nhận Diện 2 Tay

Tài liệu này lưu lại quá trình thảo luận và các giải pháp nâng cấp mô hình GRU (từ mức accuracy ~41.87% lên mục tiêu ≥ 90%).

## 🎯 5 Giải Pháp Nâng Cấp GRU

### 1. Chuẩn Hoá Keypoints (Wrist-relative + Scale)
- **Vấn đề cũ:** Toạ độ lưu trữ là toạ độ tuyệt đối trong khung hình camera. Cùng 1 ký hiệu nhưng ở vị trí khác nhau trong camera sẽ sinh ra các vector hoàn toàn khác nhau khiến model lúng túng (model học vị trí thay vì hình dáng).
- **Cách xử lý:** 
  - Lấy toạ độ cổ tay (landmark 0) làm gốc (0,0,0).
  - Trừ tất cả các điểm khác cho cổ tay (chuyển về toạ độ tương đối).
  - Chia cho khoảng cách xa nhất để co về khoảng `[-1, 1]` (giúp bất biến với kích thước tay to/nhỏ).
- **Kết quả:** Model học được "hình dạng" thuần tuý của bàn tay.

### 2. Data Augmentation (Tăng Cường Dữ Liệu)
- **Vấn đề cũ:** Số lượng mẫu quá ít (trung bình 8 mẫu/lớp cho 100 lớp), dẫn đến overfitting.
- **Cách xử lý:** Từ 1 chuỗi frame gốc, sinh thêm 5 biến thể:
  - **Jitter:** Thêm nhiễu ngẫu nhiên (mô phỏng tay rung).
  - **Scale:** Co giãn toạ độ (mô phỏng khoảng cách tới camera).
  - **Time Warp:** Co giãn thời gian (mô phỏng tốc độ ra ký hiệu nhanh/chậm).
  - **Mirror:** Đảo chiều trục X (mô phỏng đổi tay trái/phải).
- **Kết quả:** Số lượng mẫu tăng gấp 6-8 lần, giúp model khái quát hoá tốt hơn.

**💡 Chú ý quan trọng: Tại sao lại thực hiện Augmentation lúc Train (train_gru.py) thay vì lúc Prepare (Prepare_sequences.py)?**
Có 3 lý do cốt lõi:
1. **Tránh Data Leakage (Rò rỉ dữ liệu):** Nếu sinh dữ liệu ảo trước khi chia Train/Val, các bản sao của cùng một video có thể bị chia vào cả 2 rổ. Kết quả là model học thuộc lòng (memorize) video ở tập Train và đánh lừa bài kiểm tra ở tập Val, làm điểm Val cao giả tạo. Augment lúc Train đảm bảo tập Validation hoàn toàn "nguyên bản".
2. **Augment chuẩn xác hơn:** Các kỹ thuật như Scale hay Mirror thực hiện dễ dàng và chuẩn xác nhất *sau khi* tọa độ đã được chuẩn hoá (đưa cổ tay về gốc 0). Do bước chuẩn hoá nằm ở `train_gru.py`, bước Augment cũng phải nằm ở đó.
3. **Tiết kiệm ổ cứng (Disk Space):** Tạo 7 bản sao cho 810 video sẽ đẻ ra ~5,670 file `.npy`. Nếu lưu hết xuống ổ cứng sẽ rất nặng và làm chậm tốc độ đọc. Sinh ra "ảo" trên RAM lúc Train sẽ giải phóng bộ nhớ ngay sau khi dùng xong.

### 3. Thu Gọn Mô Hình (Architecture)
- **Vấn đề cũ:** Model cũ có ~1.5 triệu tham số, quá lớn so với lượng dữ liệu hiện có (~810 mẫu gốc), chắc chắn gây ra overfitting.
- **Cách xử lý:** Bỏ lớp Dense trung gian, thu nhỏ số node của BiGRU (từ 256/128 xuống 128/64), tăng cường Dropout lên 0.4.
- **Kết quả:** Model nhỏ gọn hơn (~350k params), học các đặc trưng thiết thực hơn, giảm tỷ lệ overfitting.

### 4. Label Smoothing & Tăng Regularization (L2)
- **Vấn đề cũ:** Hàm `loss='categorical_crossentropy'` bản chất là hàm Loss dùng Nhãn Cứng (Hard One-Hot Labels). Nó cực kỳ phù hợp với những bài toán thỏa mãn 3 điều kiện sau:
  1. **Ranh giới lớp rõ ràng tuyệt đối (Deterministic):** Ví dụ: Nhận diện số viết tay MNIST ($0, 1, 2, \dots, 9$). Số 3 là số 3, nó không thể "lai" $10\%$ số 8 hay $5\%$ số 5 được. Phân loại linh kiện lỗi/không lỗi trên dây chuyền công nghiệp.
  2. **Dữ liệu cực kỳ sạch và KHÔNG CÓ sai số diễn giải:** Ảnh chụp rõ nét, không bị mờ nhòe, không có góc khuất.
  3. **Tập dữ liệu khổng lồ (Hàng trăm nghìn mẫu):** Khi dữ liệu đủ dầy, dù bị ép gán cứng $1.0$, mô hình vẫn không sợ bị Overfitting vì nó có quá nhiều ví dụ để học.
  
  Do bài này dataset cho GRU quá ít nên không thể sử dụng hàm này nguyên bản mà phải "làm mềm" nó ra. L2 regularization cũ cũng quá nhỏ (0.0001).
- **Cách xử lý:** 
  - Thêm `label_smoothing=0.1`. Nếu làm mềm thì 99 nhãn còn lại KHÔNG BỊ COI LÀ RÁC CỦA CON SỐ 0 TUYỆT ĐỐI NỮA, mà mỗi nhãn sẽ nhận được một lượng xác suất nhỏ xíu là 0.001 ($0.1\%$). Việc này gửi một thông điệp toán học đến mạng GRU: *"Từ vựng chính xác là Hello ($90.1\%$), nhưng cử chỉ tay này vẫn có $0.1\%$ nét tương đồng với các từ khác, hãy giữ trọng số ở mức vừa phải, đừng học vẹt $100\%$!"*
  - Tăng L2 lên `0.001` để phạt các trọng số quá lớn.

### 5. Điều Chỉnh Training Config
- **Cách xử lý:** Tăng số epochs (lên 200) và patience (lên 30) để model có thời gian hội tụ với dữ liệu đã được Augment. Giảm batch_size xuống 16 để gradient cập nhật ổn định hơn. Tăng Learning Rate khởi điểm lên `0.001`.

---

## 🤚🤚 Cập Nhật Đặc Biệt: Hỗ Trợ 2 Bàn Tay (Phân Biệt Trái/Phải)

**Lý do:** Ngôn ngữ ký hiệu bắt buộc cần cả 2 tay. Code cũ cấu hình `max_num_hands=1`, làm mất đi thông tin quan trọng.

### Những thay đổi đã thực hiện:

1. **File `Prepare_sequences.py`:**
   - Sửa `max_num_hands=2`.
   - Tạo vector `126 features` (63 số cho tay trái + 63 số cho tay phải) cho mỗi frame.
   - **Xử lý trái/phải:** Sử dụng `results.multi_handedness` của MediaPipe để luôn cố định vị trí:
     - 63 số đầu `[0:63]` luôn chứa toạ độ TAY TRÁI.
     - 63 số sau `[63:126]` luôn chứa toạ độ TAY PHẢI.
   - Nếu chỉ có 1 tay, tay còn lại tự động điền `0` (padding).

2. **File `train_gru.py` (Hàm `normalize_keypoints`):**
   - Tự động nhận diện số lượng tay thông qua chiều của input data (`input_dim`).
   - Xử lý **Normalize riêng rẽ từng tay** (tay trái tính theo gốc cổ tay trái, tay phải tính theo gốc cổ tay phải).
   - Nếu tay nào là padding (toàn số 0), thì giữ nguyên và bỏ qua.

### Tác động của thay đổi:
Việc phân chia rõ rệt vùng dữ liệu của Tay Trái và Tay Phải trong một vector giúp GRU không bị "nhầm lẫn" khi MediaPipe thỉnh thoảng đổi thứ tự detect tay trong từng frame. Model sẽ học được sự phối hợp đồng bộ giữa 2 tay để định danh chính xác các từ vựng phức tạp.

*(Lưu ý: Luôn nhớ chạy lại `Prepare_sequences.py` để generate lại tập `.npy` (shape mới: `(30, 126)`) trước khi tiến hành train lại bằng `train_gru.py`.)*

---

## 🔬 Thảo Luận Nâng Cao: Lọc Ngưỡng & Chiến Lược Bơm Dữ Liệu

Quá trình nâng cấp đã xem xét 2 chiến thuật quan trọng về xử lý dữ liệu mà bạn đã đề xuất:

### 1. Về việc "Lọc ngưỡng" (Loại bỏ class có quá ít mẫu)
- **Tình trạng hiện tại:** Chúng ta **CHƯA** thực hiện bước này trong mã nguồn. Hàm `load_data()` hiện vẫn đang nạp 100% các từ vựng, kể cả những từ chỉ có 3-4 video mẫu.
- **Phân tích:** Việc cố chấp giữ lại các từ vựng "hẻo" data này có thể làm giảm độ chính xác tổng thể vì AI không đủ số lượng ví dụ đa dạng để khái quát hóa. Nếu sau này điểm số vẫn không như kỳ vọng, việc chủ động "hy sinh" (loại bỏ) các từ vựng có quá ít mẫu (< 12 hoặc < 15 mẫu) là một bước Data Cleaning (làm sạch dữ liệu) cực kỳ đúng đắn và chuyên nghiệp.

### 2. Về việc "Bơm dữ liệu" (Augmentation): Offline vs Online
- **Chiến thuật đề xuất ban đầu (Offline):** Tạo thêm các phiên bản biến thể `.npy` (thêm nhiễu, tịnh tiến) và lưu vĩnh viễn vào ổ cứng.
- **Chiến thuật đã thực thi (Online):** Chúng ta ĐÃ ÁP DỤNG Augmentation, nhưng bằng phương pháp **Online** (bơm trực tiếp vào thanh RAM lúc đang huấn luyện thông qua hàm `augment_keypoints`).
- **Tại sao Online lại ưu việt hơn?** 
  - **Tiết kiệm tài nguyên:** Không làm đầy rác ổ cứng (hoặc Google Drive trên Colab) bởi hàng chục ngàn file `.npy` biến thể. Máy tính sinh dữ liệu ảo trên RAM để học, học xong là giải phóng RAM ngay lập tức.
  - **An toàn tuyệt đối:** Tránh hoàn toàn thảm họa **Data Leakage** (rò rỉ dữ liệu). Nếu làm Offline rồi mới chia Train/Val, biến thể của một video có thể lọt vào cả 2 tập. Làm Online đảm bảo tập Validation luôn trong sạch và là dữ liệu nguyên bản 100%.
