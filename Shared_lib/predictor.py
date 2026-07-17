# mobile_app/src/inference_engine/predictor.py
import cv2
import numpy as np
import tensorflow as tf
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..')) # vì nằm ở file gốc nên chỉ cần lên 1 cấp thôi 
from shared_lib.constants import ALPHABET_CLASSES, ACTION_CLASSES


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
        self.buffer = []
        
        
    def detect_hand(self, frame):
        # Tiền xử lý ảnh cho YOLO (cần resize về đúng kích thước input, thường là 320x320)
        input_shape = self.yolo_input_details[0]['shape']
        img = cv2.resize(frame, (input_shape[2], input_shape[1]))
        img = img.astype(np.float32) / 255.0
        img = np.expand_dims(img, axis=0)

        self.yolo_interpreter.set_tensor(self.yolo_input_details[0]['index'], img)
        self.yolo_interpreter.invoke()
        output = self.yolo_interpreter.get_tensor(self.yolo_output_details[0]['index'])

        # Xử lý output YOLO (định dạng [batch, num_boxes, 6] với xyxy, conf, class)
        # Đơn giản lấy box có confidence cao nhất
        boxes = output[0]  # shape [num_boxes, 6]
        if len(boxes) == 0:
            return None
        best_box = max(boxes, key=lambda x: x[4])  # confidence ở index 4
        if best_box[4] < 0.5:
            return None
        x1, y1, x2, y2 = map(int, best_box[:4])
        return (x1, y1, x2, y2)

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
        return ACTION_CLASSES[idx], pred[idx]

    def process_frame(self, frame):
        """
        Nhận frame → trả về text thô + bbox
        Không sửa lỗi, không dịch — việc đó của Cloud
        """
        bbox = self.detect_hand(frame)
        static_char = None
        feature_vec = np.zeros(256, dtype=np.float32)  # Kích thước feature vector

        if bbox is not None:
            x1, y1, x2, y2 = bbox
            hand = frame[y1:y2, x1:x2]
            if hand.size > 0:
                feature_vec = self.extract_feature(hand)
                # Dự đoán tĩnh từ feature vector (có thể dùng thêm classifier head)
                # Tạm thời không dùng vì feature_extractor là EfficientNet không có softmax cho A-Z
                # static_char = ALPHABET_CLASSES[np.argmax(class_pred)]

        self.buffer.append(feature_vec)
        if len(self.buffer) > self.seq_len * 2:
            self.buffer = self.buffer[-self.seq_len:]  # giữ tối đa 2 lần seq_len

        action, conf = self.predict_action()
        return static_char, action, conf, bbox #Trả về thô, Cloud xử lý tiếp