import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
from collections import deque
import time
import sys
import os
import pickle

import config
from Shared_lib.sequence_utils import landmarks_to_vector, prepare_sequence

ASL_29_CLASSES = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"]
ACTION_CLASSES = []
pkl_path = os.path.join(config.SHARED_ASSETS_DIR, "action_recognizer_info.pkl")
if os.path.exists(pkl_path):
    try:
        with open(pkl_path, 'rb') as f:
            info = pickle.load(f)
            ACTION_CLASSES = info.get('classes', [])
    except Exception as e:
        print(f"Warning: Could not load ACTION_CLASSES from {pkl_path}")

class PredictionBufferFilter:
    def __init__(self, window_size=7, min_conf=0.65, debounce_time=0.3):
        self.window = deque(maxlen=window_size)
        self.min_conf = min_conf
        self.debounce_time = debounce_time
        self.last_pred_time = 0
        self.last_pred = None

    def update(self, pred, conf):
        if pred is None or conf < self.min_conf:
            self.window.append(None)
            return None
        else:
            self.window.append(pred)

        current_time = time.time()
        if current_time - self.last_pred_time < self.debounce_time:
            return self.last_pred

        valid_preds = [p for p in self.window if p is not None]
        if len(valid_preds) >= (self.window.maxlen // 2) + 1:
            from collections import Counter
            most_common = Counter(valid_preds).most_common(1)[0][0]
            self.last_pred = most_common
            self.last_pred_time = current_time
            self.window.clear()
            return most_common
        return None

class SignLanguagePredictor:
    def __init__(self, feature_path=None, gru_path=None, seq_len=30):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.6
        )
        # MediaPipe chuyên dụng cho Video File Processing (detect mỗi frame, không tracking)
        self.hands_video = self.mp_hands.Hands(
            static_image_mode=True,
            max_num_hands=2,
            min_detection_confidence=0.3,
            min_tracking_confidence=0.3
        )
        self.mp_draw = mp.solutions.drawing_utils
        self.is_video_mode = False


        
        # Mô hình Tĩnh (2D CNN - Dùng ảnh đen trắng)
        self.feat_model = None
        self.feat_interpreter = None
        if feature_path:
            if feature_path.endswith('.h5') or feature_path.endswith('.keras'):
                self.feat_model = tf.keras.models.load_model(feature_path, compile=False)
            else:
                self.feat_interpreter = tf.lite.Interpreter(model_path=feature_path)
                self.feat_interpreter.allocate_tensors()
                self.feat_input_details = self.feat_interpreter.get_input_details()
                self.feat_output_details = self.feat_interpreter.get_output_details()

        # --- KIẾN TRÚC LAI (HYBRID PIPELINE) ---
        # Tải mô hình YOLO (Ống nhòm Sniper) nếu có
        self.yolo_model = None
        yolo_path = os.path.join(config.SHARED_ASSETS_DIR, 'hand_det_yolo.pt')
        if getattr(config, 'USE_YOLO_HAND_CROPS', False) and os.path.exists(yolo_path):
            from ultralytics import YOLO
            self.yolo_model = YOLO(yolo_path)
            # Khởi tạo thêm 1 bản MediaPipe ở chế độ Static Image (dành riêng cho Crop YOLO)
            self.hands_static = self.mp_hands.Hands(
                static_image_mode=True, max_num_hands=1, min_detection_confidence=0.3
            )
            print("👁️ [Hybrid] Đã kích hoạt ống nhòm YOLOv8 thành công!")
            
        # Mô hình Động (GRU - Dùng mảng 126 số 3D)
        self.gru_model = None
        self.gru_interpreter = None
        if gru_path:
            if gru_path.endswith('.h5') or gru_path.endswith('.keras'):
                self.gru_model = tf.keras.models.load_model(gru_path, compile=False)
            else:
                self.gru_interpreter = tf.lite.Interpreter(model_path=gru_path)
                self.gru_interpreter.allocate_tensors()
                self.gru_input_details = self.gru_interpreter.get_input_details()
                self.gru_output_details = self.gru_interpreter.get_output_details()

        self.seq_len = seq_len
        self.buffer = []
        
        self.static_filter = PredictionBufferFilter(window_size=7, min_conf=0.65, debounce_time=0.3)
        self.action_filter = PredictionBufferFilter(window_size=1, min_conf=0.60, debounce_time=1.0)
        
        self.state = 'IDLE'
        self.last_wrist_pos = None
        self.velocity_history = []
        self.motion_streak = 0
        self.idle_streak = 0
        self.missing_hand_count = 0

    def enable_video_mode(self):
        if not self.is_video_mode:
            self.hands.close()
            self.hands = self.mp_hands.Hands(
                static_image_mode=True,
                max_num_hands=2,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
            self.is_video_mode = True
            print("🎥 [Predictor] Chuyển sang chế độ Video (static_image_mode=True)")

    def disable_video_mode(self):
        """Return to tracked live-camera processing after translating a file."""
        if self.is_video_mode:
            self.hands.close()
            self.hands = self.mp_hands.Hands(
                static_image_mode=False,
                max_num_hands=2,
                min_detection_confidence=0.6,
                min_tracking_confidence=0.6,
            )
            self.is_video_mode = False

    def _predict_action(self, raw_sequence):
        sequence = prepare_sequence(raw_sequence, self.seq_len)[None, ...]
        if self.gru_model:
            prediction = self.gru_model.predict(sequence, verbose=0)[0]
        else:
            self.gru_interpreter.set_tensor(self.gru_input_details[0]['index'], sequence)
            self.gru_interpreter.invoke()
            prediction = self.gru_interpreter.get_tensor(self.gru_output_details[0]['index'])[0]
        index = int(np.argmax(prediction))
        confidence = float(prediction[index])
        word = ACTION_CLASSES[index] if confidence > 0.65 and index < len(ACTION_CLASSES) else None
        return word, confidence

    def calculate_adaptive_motion(self, current_wrist_pos):
        if self.last_wrist_pos is None:
            self.last_wrist_pos = current_wrist_pos
            return 0.0, 3.0
        dx = current_wrist_pos[0] - self.last_wrist_pos[0]
        dy = current_wrist_pos[1] - self.last_wrist_pos[1]
        velocity = np.sqrt(dx**2 + dy**2)
        self.last_wrist_pos = current_wrist_pos
        
        self.velocity_history.append(velocity)
        if len(self.velocity_history) > 15:
            self.velocity_history.pop(0)
            
        avg_velocity = np.mean(self.velocity_history)
        adaptive_threshold = max(10.0, avg_velocity * 0.6)
        return velocity, adaptive_threshold

    def create_skeleton_image(self, hand_landmarks, img_w, img_h):
        # KHÔI PHỤC: Dùng để vẽ ảnh 2D phục vụ mạng CNN lúc IDLE
        # Training images use a green skeleton on a white background.
        canvas = np.full((128, 128, 3), 255, dtype=np.uint8)
        
        x_coords = [lm.x for lm in hand_landmarks.landmark]
        y_coords = [lm.y for lm in hand_landmarks.landmark]
        x_min, x_max = min(x_coords), max(x_coords)
        y_min, y_max = min(y_coords), max(y_coords)
        
        box_w = max(x_max - x_min, 0.001)
        box_h = max(y_max - y_min, 0.001)
        
        points = []
        for lm in hand_landmarks.landmark:
            px = int(((lm.x - x_min) / box_w) * 100 + 14)
            py = int(((lm.y - y_min) / box_h) * 100 + 14)
            points.append((px, py))
            cv2.circle(canvas, (px, py), 2, (0, 255, 0), -1)
            
        for connection in self.mp_hands.HAND_CONNECTIONS:
            pt1 = points[connection[0]]
            pt2 = points[connection[1]]
            cv2.line(canvas, pt1, pt2, (0, 255, 0), 2)
            
        return canvas

    def process_frame(self, frame, video_mode=False):
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_h, img_w, _ = frame.shape
        
        # --- HYBRID PIPELINE LOGIC ---
        if self.yolo_model:
            # 1. Ống nhòm YOLO phát hiện vị trí bàn tay
            yolo_results = self.yolo_model(img_rgb, verbose=False)
            boxes = yolo_results[0].boxes.xyxy.cpu().numpy()
            
            multi_hand_landmarks = []
            multi_handedness = []
            
            for box in boxes:
                x1, y1, x2, y2 = map(int, box[:4])
                # Mở rộng (padding) 25% để MediaPipe thấy cả cổ tay
                pad_x = int((x2 - x1) * 0.25)
                pad_y = int((y2 - y1) * 0.25)
                x1 = max(0, x1 - pad_x)
                y1 = max(0, y1 - pad_y)
                x2 = min(img_w, x2 + pad_x)
                y2 = min(img_h, y2 + pad_y)
                
                crop_w, crop_h = x2 - x1, y2 - y1
                if crop_w < 10 or crop_h < 10: continue
                
                crop = img_rgb[y1:y2, x1:x2]
                
                # 2. Phóng to ảnh (Zoom) và nhờ MediaPipe giải phẫu tĩnh
                # Ép kích thước lên ít nhất 256 để MediaPipe nhìn cực rõ
                if crop_w < 256 or crop_h < 256:
                    crop = cv2.resize(crop, (256, 256), interpolation=cv2.INTER_LINEAR)
                    
                mp_res = self.hands_static.process(crop)
                
                if mp_res.multi_hand_landmarks:
                    hand_lm = mp_res.multi_hand_landmarks[0]
                    # 3. Toán học nội suy: Ánh xạ 21 điểm về khung ảnh lớn
                    for lm in hand_lm.landmark:
                        lm.x = (lm.x * crop_w + x1) / img_w
                        lm.y = (lm.y * crop_h + y1) / img_h
                        # Z giữ nguyên tỷ lệ tương đối
                        
                    multi_hand_landmarks.append(hand_lm)
                    if mp_res.multi_handedness:
                        multi_handedness.append(mp_res.multi_handedness[0])
            
            class FakeResults:
                pass
            results = FakeResults()
            results.multi_hand_landmarks = multi_hand_landmarks if multi_hand_landmarks else None
            results.multi_handedness = multi_handedness if multi_handedness else None
            # Fallback cho Video mode: nếu YOLO không detect được tay, thử MediaPipe Static toàn khung hình
            if not multi_hand_landmarks and video_mode:
                fallback_res = self.hands_video.process(img_rgb)
                if fallback_res.multi_hand_landmarks:
                    results.multi_hand_landmarks = fallback_res.multi_hand_landmarks
                    results.multi_handedness = fallback_res.multi_handedness
        else:
            # Ống kính thường (Original MediaPipe)
            if video_mode:
                results = self.hands_video.process(img_rgb)
            else:
                results = self.hands.process(img_rgb)
        
        static_char = None
        static_conf = 0.0
        action_word = None
        action_conf = 0.0
        bboxes = []
        
        if results.multi_hand_landmarks:
            self.missing_hand_count = 0
            
            # --- LUÔN TRÍCH XUẤT 126 SỐ 3D CHO GRU (CHUẨN HÓA MỚI) ---
            current_126_features = landmarks_to_vector(
                results.multi_hand_landmarks, results.multi_handedness
            )
            
            for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                hand_label = results.multi_handedness[idx].classification[0].label if results.multi_handedness else "Hand"
                self.mp_draw.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                
                # Bounding Box (Thay YOLO)
                x_min, y_min = img_w, img_h
                x_max, y_max = 0, 0
                for lm in hand_landmarks.landmark:
                    x, y = int(lm.x * img_w), int(lm.y * img_h)
                    x_min, y_min = min(x_min, x), min(y_min, y)
                    x_max, y_max = max(x_max, x), max(y_max, y)
                pad = 20
                bboxes.append((max(0, x_min-pad), max(0, y_min-pad), min(img_w, x_max+pad), min(img_h, y_max+pad), hand_label))
                
            # Tính chuyển động
            wrist_pos = (results.multi_hand_landmarks[0].landmark[0].x * img_w, results.multi_hand_landmarks[0].landmark[0].y * img_h)
            velocity, adaptive_thresh = self.calculate_adaptive_motion(wrist_pos)
            
            num_hands = len(results.multi_hand_landmarks)
            is_moving = (velocity > adaptive_thresh)
            
            if is_moving:
                self.motion_streak += 1
                self.idle_streak = 0
                if self.motion_streak >= 3 and self.state == 'IDLE':
                    self.state = 'GESTURE'
                    self.buffer.clear()
            else:
                self.idle_streak += 1
                self.motion_streak = 0
                if self.idle_streak >= 5 and self.state == 'GESTURE':
                    self.state = 'IDLE' 
                    
                    # GRU - SỬ DỤNG 126 SỐ 3D HOÀN HẢO
                    if (self.gru_model or self.gru_interpreter) and len(self.buffer) >= 5:
                        action_word, action_conf = self._predict_action(self.buffer)
                    self.buffer.clear()
                    
            if self.state == 'IDLE' and (num_hands == 1 or video_mode):
                # CNN - VẼ LẠI ẢNH BỘ XƯƠNG ĐỂ TRUYỀN VÀO CNN CŨ
                skeleton_img = self.create_skeleton_image(results.multi_hand_landmarks[0], img_w, img_h)
                # The saved CNN contains MobileNetV2 preprocessing already.
                input_arr = skeleton_img.astype(np.float32)
                input_arr = np.expand_dims(input_arr, axis=0)
                
                cnn_preds = None
                if self.feat_model:
                    cnn_preds = self.feat_model.predict(input_arr, verbose=0)[0]
                elif self.feat_interpreter:
                    self.feat_interpreter.set_tensor(self.feat_input_details[0]['index'], input_arr)
                    self.feat_interpreter.invoke()
                    cnn_preds = self.feat_interpreter.get_tensor(self.feat_output_details[0]['index'])[0]
                
                if cnn_preds is not None:
                    max_idx = np.argmax(cnn_preds)
                    static_conf = float(cnn_preds[max_idx])
                    if static_conf > 0.7 and max_idx < len(ASL_29_CLASSES):
                        static_char = ASL_29_CLASSES[max_idx]
                    
            elif self.state == 'GESTURE':
                # Đang múa thì lưu mảng 126 số 3D vào buffer
                self.buffer.append(current_126_features.copy())
                if len(self.buffer) > self.seq_len * 2:
                    self.buffer = self.buffer[-self.seq_len:]
        else:
            self.missing_hand_count += 1
            self.last_wrist_pos = None
            self.velocity_history.clear()
            
            # Kích hoạt dự đoán GRU nếu người dùng múa xong và hạ tay xuống (mất dấu tay)
            if self.missing_hand_count == 5 and self.state == 'GESTURE':
                self.state = 'IDLE'
                if (self.gru_model or self.gru_interpreter) and len(self.buffer) >= 5:
                    action_word, action_conf = self._predict_action(self.buffer)
                
                self.buffer.clear()
                
            if self.missing_hand_count > 15:
                self.buffer.clear()
                self.state = 'IDLE'

        if video_mode:
            # Video mode: bỏ qua debounce timer (xử lý nhanh hơn real-time), chỉ lọc theo ngưỡng confidence
            final_static = static_char if static_conf >= 0.65 else None
            final_action = action_word if action_conf >= 0.60 else None
        else:
            final_static = self.static_filter.update(static_char, static_conf)
            final_action = self.action_filter.update(action_word, action_conf)
        selected_conf = action_conf if final_action else static_conf
        
        state_text = "MUA CUM TU" if self.state == 'GESTURE' else "DANH VAN"
        cv2.putText(frame, f"Mode: {state_text}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0) if self.state == 'IDLE' else (0, 165, 255), 2)

        return final_static, final_action, selected_conf, bboxes
