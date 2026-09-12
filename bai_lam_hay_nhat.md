# Lộ trình nâng cấp lên bản PRO: MediaPipe Holistic (Đọc hiểu Toàn thân)

Đây là tài liệu ghi chú lại quy trình "Đại phẫu thuật" để nâng cấp AI ngôn ngữ ký hiệu từ chỉ nhận diện bàn tay (Hand-only) lên nhận diện toàn thân (Holistic: Mặt + Vai + Tay). Việc này giúp độ chính xác trên tập 1000 từ vựng có thể tăng vọt lên mức 70% - 80% nhờ AI có khả năng đọc được biểu cảm.

## 1. Thay "Mắt thần" trong file Trích xuất (`Shared_lib/sequence_utils.py`)
- Chuyển thư viện `mp.solutions.hands` thành `mp.solutions.holistic`.
- **Khối lượng dữ liệu tăng vọt:** Cần thu thập 42 điểm (hai tay) + 468 điểm (khuôn mặt) + 33 điểm (cơ thể). Cập nhật mảng kết xuất thành vector 1D dài **1.662 con số** thay vì 126 số như hiện tại.

## 2. Sửa "Thước đo" trong file Huấn luyện (`Cloud_server/Trainer/train_scripts/train_gru.py`)
- **Sửa điểm neo (Anchor point):** Trong hàm `normalize_keypoints`, phải chuyển việc neo gốc tọa độ (0,0) từ cổ tay về **chóp mũi** hoặc **giữa ngực**.
- **Sửa cổng nạp dữ liệu:** Đổi thông số `input_dim` từ `126` lên `1662` (hoặc con số khớp với mảng output của Holistic).

## 3. Vô hiệu hóa "Ống nhòm Sniper" trong file Nạp từ mới (`Cloud_server/Trainer/Retrain.py`)
- **Vô hiệu hóa thuật toán crop bằng YOLO:** Holistic yêu cầu phải nhìn thấy TOÀN BỘ khuôn mặt và bả vai. Việc dùng YOLO để crop và phóng to riêng phần bàn tay sẽ khiến Holistic bị "mù" khuôn mặt và không chạy được.

## 4. Nâng cấp Camera Thực tế ảo (`Demo_ui/App.py` & `Tools/voice_to_sign_engine.py`)
- Cập nhật các hàm vẽ (`mp.solutions.drawing_utils`) để hiển thị mặt nạ lưới (Face Mesh) và khung xương vai trên luồng Camera thực.
- Đồng bộ lại hàm trích xuất dữ liệu của camera trực tiếp: đảm bảo nó đẩy đủ vector 1662 điểm vào hàm predict.

## 5. ⚠️ CẢNH BÁO QUAN TRỌNG VỀ PHẦN CỨNG (SỤP RAM)
- Việc mảng dữ liệu phình to gấp hơn 10 lần đồng nghĩa với việc đẩy 48.000 files `.npy` lên RAM lúc huấn luyện có thể **ngốn tàn bạo lên tới 32GB RAM**.
- Nếu máy tính của bạn chỉ có 8GB hoặc 16GB RAM, tiến trình `train_gru.py` sẽ lập tức văng lỗi **Out of Memory (OOM)** và chết ngang.
- **Giải pháp bắt buộc:** Không được dùng hàm `load_data()` đẩy 100% data vào RAM một lượt nữa. Phải sửa code thành cơ chế `DataGenerator` (Dùng `tf.keras.utils.Sequence` hoặc `tf.data.Dataset`) để máy tự động múc từng gáo nhỏ dữ liệu từ ổ cứng nạp vào RAM trong lúc Train.
