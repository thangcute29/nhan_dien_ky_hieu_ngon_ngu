# mobile_app/src/inference_engine/predictor.py
import cv2
import numpy as np
import tensorflow as tf
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..')) # vì nằm ở file gốc nên chỉ cần lên 1 cấp thôi 
from Shared_lib.Constants import ALPHABET_CLASSES, ACTION_CLASSES


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

        # Ngưỡng tự tin 0.15 giúp bắt nhạy 2 bàn tay trên webcam
        valid_indices = np.where(predictions[:, 4] >= 0.15)[0]
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

        indices = cv2.dnn.NMSBoxes(boxes, scores, score_threshold=0.15, nms_threshold=0.45)

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

    def extract_feature(self, hand_roi):
        img = cv2.resize(hand_roi, (224, 224))
        img = img.astype(np.float32) / 255.0
        img = np.expand_dims(img, axis=0)
        self.feat_interpreter.set_tensor(self.feat_input_details[0]['index'], img)
        self.feat_interpreter.invoke()
        feature = self.feat_interpreter.get_tensor(self.feat_output_details[0]['index'])
        return feature.flatten()

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

        # ===== DEBUG: In kết quả dự đoán GRU mỗi 10 frame =====
        if self._debug_frame_count % 10 == 0 and len(ACTION_CLASSES) > idx:
            print(f"[DEBUG] GRU predict -> Action: '{ACTION_CLASSES[idx]}' | Conf: {conf:.4f} (Threshold: 0.35)", flush=True)
        # =======================================================

        # Hạ ngưỡng tự tin dự đoán từ từ 0.55 xuống 0.35 để bắt dịch thuật từ nhạy hơn
        if conf >= 0.35 and len(ACTION_CLASSES) > idx:
            return ACTION_CLASSES[idx], conf
        else:
            return None, conf

    def process_frame(self, frame):
        bboxes = self.detect_hands(frame)
        static_char = None
        feature_vec = np.zeros(self.feat_dim, dtype=np.float32)

        if not hasattr(self, '_missing_hand_count'):
            self._missing_hand_count = 0

        if len(bboxes) > 0:
            self._missing_hand_count = 0
            feats = []
            for item in bboxes:
                x1, y1, x2, y2 = item[:4]
                hand = frame[y1:y2, x1:x2]
                if hand.size > 0:
                    feats.append(self.extract_feature(hand))
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

        action, conf = self.predict_action()
        return static_char, action, conf, bboxes