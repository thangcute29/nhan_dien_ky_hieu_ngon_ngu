# mobile_app/src/inference_engine/predictor.py
import cv2
import numpy as np
import tensorflow as tf
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..')) # vì nằm ở file gốc nên chỉ cần lên 1 cấp thôi 
from Shared_lib.Constants import ALPHABET_CLASSES, ASL_29_CLASSES, ACTION_CLASSES
from collections import Counter


class PredictionBufferFilter:
    """
    Bộ lọc 3 bước lọc nhiễu dự đoán (3-Step Prediction Pipeline):
    Bước 1: Bầu chọn số đông (Majority Voting) trong window_size khung hình gần nhất.
    Bước 2: Lọc theo ngưỡng tự tin tối thiểu (min_conf >= 0.35).
    Bước 3: Lọc trùng lặp trạng thái (Debounce check: chỉ kích hoạt khi nhãn đổi mới).
    """
    def __init__(self, window_size=5, min_conf=0.35):
        self.window_size = window_size
        self.min_conf = min_conf
        self.history = []
        self.last_emitted = None

    def update(self, raw_label, conf):
        # Bước 2: Kiểm tra ngưỡng tự tin tối thiểu
        if conf < self.min_conf or raw_label is None or raw_label in ['nothing', 'del']:
            return None

        # Tích lũy vào đệm (Ring Buffer)
        self.history.append(raw_label)
        if len(self.history) > self.window_size:
            self.history.pop(0)

        if len(self.history) < 3:
            return None

        # Bước 1: Bầu chọn số đông (Majority Voting)
        counts = Counter(self.history)
        winner_label, freq = counts.most_common(1)[0]

        # Đạt tối thiểu 50% + 1 số phiếu đồng thuận
        required_votes = (len(self.history) // 2 + 1)
        if freq < required_votes:
            return None

        # Bước 3: Lọc trùng lặp (Debounce Check)
        if winner_label != self.last_emitted:
            self.last_emitted = winner_label
            return winner_label  # Kích hoạt nhãn mới!

        return None


class SignLanguagePredictor:
    def __init__(self, yolo_path, feature_path, gru_path, seq_len=30):
        self.yolo_interpreter = tf.lite.Interpreter(model_path=yolo_path)
        self.yolo_interpreter.allocate_tensors()
        self.yolo_input_details = self.yolo_interpreter.get_input_details()
        self.yolo_output_details = self.yolo_interpreter.get_output_details()

        self.feat_interpreter = tf.lite.Interpreter(model_path=feature_path)
        self.feat_interpreter.allocate_tensors()
        self.feat_input_details = self.feat_interpreter.get_input_details()
        self.feat_output_details = self.feat_interpreter.get_output_details()

        self.gru_interpreter = tf.lite.Interpreter(model_path=gru_path)
        self.gru_interpreter.allocate_tensors()
        self.gru_input_details = self.gru_interpreter.get_input_details()
        self.gru_output_details = self.gru_interpreter.get_output_details()

        self.seq_len = seq_len
        self.feat_dim = self.gru_input_details[0]['shape'][2]
        self.buffer = []

        # Khởi tạo bộ lọc 3 bước cho chữ cái tĩnh và từ vựng động
        self.static_filter = PredictionBufferFilter(window_size=5, min_conf=0.35)
        self.action_filter = PredictionBufferFilter(window_size=5, min_conf=0.35)

        # ===== DEBUG: in ra thông tin model YOLO khi khởi tạo =====
        print(f"[DEBUG] YOLO input shape : {self.yolo_input_details[0]['shape']}, dtype: {self.yolo_input_details[0]['dtype']}", flush=True)
        print(f"[DEBUG] YOLO output shape: {self.yolo_output_details[0]['shape']}, dtype: {self.yolo_output_details[0]['dtype']}", flush=True)
        self._debug_frame_count = 0
        # ===========================================================

    def detect_hands(self, frame):
        input_shape = self.yolo_input_details[0]['shape']
        img_h, img_w = frame.shape[:2]
        # Chuyển đổi BGR sang RGB để mô hình YOLO nhận diện màu sắc bàn tay chuẩn xác
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = cv2.resize(rgb_frame, (input_shape[2], input_shape[1]))
        img = img.astype(np.float32) / 255.0
        img = np.expand_dims(img, axis=0)

        self.yolo_interpreter.set_tensor(self.yolo_input_details[0]['index'], img)
        self.yolo_interpreter.invoke()
        output = self.yolo_interpreter.get_tensor(self.yolo_output_details[0]['index'])

        predictions = output[0].T  # shape [2100, 5]

        # ===== DEBUG: in thống kê confidence thô mỗi 10 frame =====
        self._debug_frame_count += 1
        if self._debug_frame_count % 10 == 0:
            max_conf = predictions[:, 4].max()
            mean_conf = predictions[:, 4].mean()
            n_above_035 = int((predictions[:, 4] >= 0.35).sum())
            n_above_010 = int((predictions[:, 4] >= 0.10).sum())
            print(f"[DEBUG] frame#{self._debug_frame_count} | max_conf={max_conf:.4f} | mean_conf={mean_conf:.4f} "
                  f"| n>=0.35: {n_above_035} | n>=0.10: {n_above_010}", flush=True)
        # ============================================================

        # SỰ CỐ 1 FIX: Nâng ngưỡng tự tin lên 0.45 để loại bỏ 100% Bounding box nhấp nháy rác
        CONF_THRESHOLD_HAND = 0.45
        valid_indices = np.where(predictions[:, 4] >= CONF_THRESHOLD_HAND)[0]
        if len(valid_indices) == 0:
            return []

        boxes = []
        scores = []
        scale_x = img_w / input_shape[2]
        scale_y = img_h / input_shape[1]

        for idx in valid_indices:
            cx, cy, w, h = predictions[idx, :4]
            x1 = max(0, int((cx - w / 2) * scale_x))
            y1 = max(0, int((cy - h / 2) * scale_y))
            x2 = min(img_w, int((cx + w / 2) * scale_x))
            y2 = min(img_h, int((cy + h / 2) * scale_y))
            
            boxes.append([x1, y1, x2 - x1, y2 - y1])
            scores.append(float(predictions[idx, 4]))

        if len(boxes) == 0:
            return []

        indices = cv2.dnn.NMSBoxes(boxes, scores, score_threshold=CONF_THRESHOLD_HAND, nms_threshold=0.45)

        # ===== DEBUG: xem NMS giữ lại bao nhiêu box =====
        if self._debug_frame_count % 10 == 0:
            print(f"[DEBUG] frame#{self._debug_frame_count} | boxes sau NMS={len(indices) if indices is not None else 0}", flush=True)
        # ==================================================

        detected_hands = []
        
        if len(indices) > 0:
            for i in indices.flatten()[:2]:  
                x, y, w, h = boxes[i]
                center_x = x + w / 2
                detected_hands.append({
                    "box": (x, y, x + w, y + h),
                    "center_x": center_x
                })

            # Sắp xếp từ trái sang phải theo góc nhìn màn hình
            detected_hands = sorted(detected_hands, key=lambda item: item["center_x"])
            
            bboxes = []
            num_hands = len(detected_hands)
            for idx, hand in enumerate(detected_hands):
                if num_hands == 2:
                    hand_label = "Left Hand" if idx == 0 else "Right Hand"
                else:
                    # Nếu chỉ có 1 tay: Dựa vào vị trí nửa trái hay nửa phải màn hình
                    hand_label = "Left Hand" if hand["center_x"] < (img_w / 2) else "Right Hand"

                x1, y1, x2, y2 = hand["box"]
                bboxes.append((x1, y1, x2, y2, hand_label))

        return bboxes

    def process_hand_crop(self, hand_roi):
        """
        SỰ CỐ 2 FIX: Gộp cả tác vụ Dự đoán Chữ cái tĩnh A-Z và Trích xuất Feature Vector 
        vào 1 LẦN GỌI invoke() DUY NHẤT cho mỗi bàn tay crop -> Tiết kiệm 50% CPU/RAM!
        """
        if hand_roi is None or hand_roi.size == 0:
            return None, 0.0, np.zeros(self.feat_dim, dtype=np.float32)

        # Chuyển BGR sang RGB và giữ dải pixel [0, 255] phù hợp với EfficientNetB0 đã train
        rgb_hand = cv2.cvtColor(hand_roi, cv2.COLOR_BGR2RGB)
        img = cv2.resize(rgb_hand, (224, 224)).astype(np.float32)
        img = np.expand_dims(img, axis=0)

        # CHỈ GỌI INVOKE 1 LẦN DUY NHẤT PER HAND CROP
        self.feat_interpreter.set_tensor(self.feat_input_details[0]['index'], img)
        self.feat_interpreter.invoke()
        probs = self.feat_interpreter.get_tensor(self.feat_output_details[0]['index'])[0]

        # 1. Trích xuất chữ cái tĩnh A-Z
        idx = np.argmax(probs)
        conf = float(probs[idx])
        static_char = None

        # ===== THÊM DEBUG LOG (Top-3) =====
        if self._debug_frame_count % 10 == 0:
            top_indices = np.argsort(probs)[::-1][:3]
            top_probs = np.sort(probs)[::-1][:3]
            print(f"[DEBUG FEAT] Top indexes: {top_indices} | Top confs: {[round(float(p), 4) for p in top_probs]}", flush=True)
            if idx < len(ASL_29_CLASSES):
                print(f"[DEBUG FEAT] Predicted Class: '{ASL_29_CLASSES[idx]}' | Conf: {conf:.4f}", flush=True)
        # ==================================

        if idx < len(ASL_29_CLASSES):
            predicted_class = ASL_29_CLASSES[idx]
            if predicted_class not in ['nothing', 'del'] and conf >= 0.35:
                static_char = predicted_class

        # 2. Lấy luôn feature vector (đầu ra flatten của model)
        feat_vec = probs.flatten()
        return static_char, conf, feat_vec

    def predict_static_alphabet(self, hand_roi):
        char, conf, _ = self.process_hand_crop(hand_roi)
        return char, conf

    def extract_feature(self, hand_roi):
        _, _, feat = self.process_hand_crop(hand_roi)
        return feat

    def predict_action(self):
        if len(self.buffer) < self.seq_len:
            return None, 0.0
        seq = np.array(self.buffer[-self.seq_len:], dtype=np.float32)
        seq = np.expand_dims(seq, axis=0)
        self.gru_interpreter.set_tensor(self.gru_input_details[0]['index'], seq)
        self.gru_interpreter.invoke()
        pred = self.gru_interpreter.get_tensor(self.gru_output_details[0]['index'])[0]
        
        idx = np.argmax(pred)
        conf = float(pred[idx])

        # Ngưỡng tin cậy cho GRU nâng lên 0.55 để tránh kích hoạt rác
        GRU_CONF_THRESHOLD = 0.55

        # ===== DEBUG: In kết quả dự đoán GRU mỗi 10 frame =====
        if self._debug_frame_count % 10 == 0 and len(ACTION_CLASSES) > idx:
            print(f"[DEBUG] GRU predict -> Action: '{ACTION_CLASSES[idx]}' | Conf: {conf:.4f} (Threshold: {GRU_CONF_THRESHOLD})", flush=True)
        # =======================================================

        if conf >= GRU_CONF_THRESHOLD and len(ACTION_CLASSES) > idx:
            return ACTION_CLASSES[idx], conf
        else:
            return None, conf

    def process_frame(self, frame):
        bboxes = self.detect_hands(frame)
        static_char = None
        static_conf = 0.0
        feature_vec = np.zeros(self.feat_dim, dtype=np.float32)

        if not hasattr(self, '_missing_hand_count'):
            self._missing_hand_count = 0

        if len(bboxes) > 0:
            self._missing_hand_count = 0
            feats = []
            static_candidates = []

            for item in bboxes:
                x1, y1, x2, y2 = item[:4]
                hand = frame[y1:y2, x1:x2]
                if hand.size > 0:
                    # GỌI 1 LẦN DUY NHẤT LẤY CẢ CHỮ CÁI LẪN FEATURE VECTOR
                    char, s_conf, feat = self.process_hand_crop(hand)
                    if char:
                        static_candidates.append((char, s_conf))
                    feats.append(feat)

            if len(static_candidates) > 0:
                static_candidates.sort(key=lambda x: x[1], reverse=True)
                static_char, static_conf = static_candidates[0]

            if len(feats) > 0:
                combined_feat = np.mean(feats, axis=0)
                n = min(len(combined_feat), self.feat_dim)
                feature_vec[:n] = combined_feat[:n]
            self.buffer.append(feature_vec)
        else:
            self._missing_hand_count += 1
            # Chỉ xóa bộ đệm nếu hạ tay hoàn toàn quá 10 frame liên tiếp
            if self._missing_hand_count > 10:
                self.buffer.clear()

        if len(self.buffer) > self.seq_len * 2:
            self.buffer = self.buffer[-self.seq_len:]

        action, action_conf = self.predict_action()

        # Áp dụng Bộ Lọc 3 Lớp (Ring Buffer + Majority Vote + Debounce)
        filtered_static = self.static_filter.update(static_char, static_conf)
        filtered_action = self.action_filter.update(action, action_conf)

        final_static = filtered_static if filtered_static else static_char
        final_action = filtered_action if filtered_action else action
        max_conf = max(static_conf, action_conf)

        # Trả về 4 tham số rõ ràng:
        # 1. Chữ cái tĩnh (hoặc None)
        # 2. Hành động từ vựng (hoặc None)
        # 3. Độ tin cậy cao nhất
        # 4. Các Bounding Box bàn tay
        return final_static, final_action, max_conf, bboxes
