# Nhật ký sửa lỗi và hướng dẫn luồng chạy dự án

## 1. Tối ưu mô hình trích xuất đặc trưng (MultiHeadModel)
- **Tình trạng cũ**: File `train_classification.py` sử dụng `if-else` để chọn giữa `efficientnet`, `mobilenet`, và `resnet50`.
- **Giải pháp**: Đã loại bỏ hoàn toàn khối `if-else`.
- **Lý do**: Vì mục tiêu là chạy trên điện thoại di động, nên chỉ giữ lại duy nhất mạng **EfficientNet-B0** vì nó rất nhẹ và tối ưu. Việc bỏ qua các mô hình khác (như ResNet50, MobileNet) giúp code gọn gàng, tránh lỗi gõ sai tham số và tránh việc chọn nhầm mô hình nặng nề.

## 2. Quy trình chạy dự án sau khi hoàn tất Huấn luyện (Train)
Sau khi đã chạy xong toàn bộ 4 file huấn luyện:
1. `train_classification.py`
2. `train_feature_extractor.py`
3. `train_gru.py`
4. `train_yolo.py`

Trình tự các file cần thực thi tiếp theo để hệ thống hoạt động là:

### Bước 1: Chuyển đổi định dạng mô hình (TFLite)
- **File cần chạy**: `Tools/convert_to_mobile.py`
- **Mục đích**: Script này sẽ lấy các mô hình PyTorch/Keras (`.pt`, `.h5`) vừa train xong để chuyển đổi sang định dạng `.tflite` nhẹ nhàng. Các file sau khi chuyển đổi sẽ được đẩy tự động vào thư mục `Mobile_app/assets` hoặc `mobile_app/assets`.

### Bước 2: Khởi động hệ thống Máy chủ (API Server)
- **File cần chạy**: `Cloud_server/Api/Main.py`
- **Mục đích**: Bật Backend Server. File này sẽ tải các logic xử lý ngữ cảnh (LLM Corrector, Context Agent, v.v...) lên, sẵn sàng nhận dữ liệu video/hình ảnh được truyền về và trả ra kết quả biên dịch ngôn ngữ ký hiệu.

### Bước 3: Chạy giao diện (Frontend / Mobile App)
- **Demo trên máy tính**: Khởi chạy script giao diện chính trong thư mục `Demo_ui/` (nếu có).
- **Ứng dụng trên Điện thoại**: Mở thư mục `Mobile_app/` trong môi trường lập trình (như Android Studio). Kiểm tra lại trong thư mục `assets` đảm bảo đã có đầy đủ các file `.tflite`. Cuối cùng, thực hiện Build và Run ứng dụng lên thiết bị ảo hoặc thiết bị thật để sử dụng hoàn chỉnh.

## 3. Cập nhật Logic lưu trữ Mô hình (Models) và Quy trình chạy (Cập nhật mới)
- **Logic lưu trữ gốc**: Các file huấn luyện (`train_yolo.py`, `train_feature_extractor.py`, `train_gru.py`) đã được viết logic để sau khi huấn luyện xong sẽ lưu trực tiếp các file model thành phẩm (`.pt`, `.h5`) vào chung một kho là `Shared_lib/assets/` thay vì tạo thư mục `models/` rời rạc. Điều này giúp cả Mobile App, Web App và API Server đều có thể dùng chung.
- **Trạng thái hiện tại**: Lần trước dự án mới chỉ dừng ở bước "Viết code logic". Các file huấn luyện **CHƯA ĐƯỢC CHẠY (RUN)**, do đó các model thành phẩm chưa tồn tại trên ổ cứng.
- **Cần khắc phục sắp tới**:
  1. Cần sửa file `Tools/convert_to_mobile.py` để trỏ đường dẫn đọc model từ `Shared_lib/assets/` (đồng bộ với file `config.py`) thay vì đọc từ `models/` (gây lỗi `FileNotFoundError`).
  2. Bắt buộc phải bấm Run các file `train_...py` để máy tính thực sự học và sinh ra file model vào `Shared_lib/assets/`, sau đó mới chạy được file `convert_to_mobile.py`.

## 4. Cập nhật tính năng Checkpoint (Resume & Save Period) cho file `train_classification.py`
- **Vấn đề**: Khi huấn luyện, nếu tiến trình bị dừng đột ngột, mô hình sẽ phải học lại từ đầu gây lãng phí thời gian.
- **Giải pháp**:
  - **Lưu định kỳ (Save Period)**: Khởi tạo biến `save_period = 5` để lưu mô hình dự phòng sau mỗi 5 epoch (ví dụ `classification_epoch_5.pth`).
  - **Lưu tiến trình liên tục**: Cập nhật liên tục tiến trình ở mỗi epoch vào file `classification_last.pth`.
  - **Tự động khôi phục (Resume)**: Trước khi bắt đầu vòng lặp Epoch, kiểm tra xem có file `classification_last.pth` không. Nếu có, tự động nạp lại trọng số mô hình, optimizer, các tham số để tiếp tục huấn luyện chính xác từ epoch bị gián đoạn.
  - **Tự động dọn dẹp**: Xóa file `classification_last.pth` khi phiên huấn luyện hoàn thành hoàn toàn, nhằm đảm bảo đợt huấn luyện sau này sẽ khởi chạy sạch sẽ.

## 5. Đồng bộ hóa logic Save Period, Resume và Dọn rác tự động cho toàn bộ dự án
Nhận thấy việc sinh ra các file lưu định kỳ (`save_period`) có thể gây đầy ổ cứng nếu không dọn dẹp, toàn bộ 4 file huấn luyện đã được nâng cấp đồng loạt:
- **Các file được nâng cấp**: `train_classification.py`, `train_feature_extractor.py`, `train_gru.py`, `train_yolo.py`.
- **Cơ chế hoạt động**:
  1. Thêm tính năng **Save Period = 5** cho tất cả (lưu dự phòng định kỳ mỗi 5 epoch).
  2. Bổ sung tính năng **Resume** (tự động load lại file checkpoint gần nhất `_last` nếu quá trình train bị gián đoạn giữa chừng).
  3. Bổ sung vòng lặp **quét dọn rác tự động bằng `glob`**: Khi huấn luyện kết thúc (thành công hoặc thất bại), hệ thống sẽ quét các thư mục chứa trọng số và tự động xóa bỏ mọi file dự phòng dạng `_epoch_*.pth`/`.h5` và `_last.pth`/`.h5`. Điều này đảm bảo ổ cứng luôn sạch sẽ, tránh đầy bộ nhớ vô ích.

> ***** LƯU Ý QUAN TRỌNG DÀNH CHO BẠN *****
> Tuy nhiên, có một lưu ý nhỏ: Tiến trình YOLO mà bạn đang chạy hiện tại là do đoạn code "cũ" khởi động. Nên khi nó chạy xong đợt này, nó sẽ chưa biết cách tự dọn rác đâu. Lần này bạn sẽ chịu khó vào thư mục `runs/yolo_hands/weights/` xóa bằng tay các file `epoch*.pt` giúp tôi nhé.
