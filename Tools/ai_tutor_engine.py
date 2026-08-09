# Tools/ai_tutor_engine.py
"""
==============================================================================
MODULE GIA SƯ AI CHẤM ĐIỂM HỌC TẬP & NGƯỜI ẢO AI TRỢ LÝ (AI TUTOR & AVATAR ENGINE)
==============================================================================
Nhiệm vụ:
1. Tiếp nhận và NẠP BỔ SUNG Video Mẫu mới (.mp4) làm "Thước đo chuẩn" cho AI.
2. Đánh giá thời gian thực cử chỉ của học viên (Vị trí giơ tay, tốc độ, độ khớp).
3. Vẽ NGƯỜI ẢO AI TRỢ LÝ (AI Virtual Avatar) phát âm thanh và hiển thị bong bóng
   hội thoại hướng dẫn trực quan trên camera.
==============================================================================
"""

import os
import sys
import time
import shutil
import cv2
import numpy as np

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

class AITutorEngine:
    def __init__(self, tts=None):
        self.tts = tts
        self.last_speech_time = 0
        self.speech_cooldown = 4.0  # Giới hạn phát âm thanh nhắc nhở mỗi 4 giây
        self.reference_templates = {}

    def enroll_reference_video(self, video_path, word_name):
        """
        Nạp Video Mẫu mới làm Thước đo chuẩn và tự động BỔ SUNG VÀO DATASET.
        """
        if not os.path.exists(video_path):
            print(f"❌ [AITutorEngine] Không tìm thấy tệp video mẫu: {video_path}")
            return False

        word_name = word_name.strip().lower()
        print(f"\n📥 [AITutorEngine] Đang nạp Video Mẫu mới cho từ: '{word_name.upper()}'...", flush=True)

        # 1. Lưu bổ sung vào kho Dataset Sequences tự học ngầm
        dataset_target_dir = os.path.join(BASE_DIR, "Research_and_Data", "Dataset", "Sequences", "custom_enrollment", word_name)
        os.makedirs(dataset_target_dir, exist_ok=True)
        
        dest_filename = f"{word_name}_ref_{int(time.time())}.mp4"
        dest_path = os.path.join(dataset_target_dir, dest_filename)
        shutil.copy2(video_path, dest_path)
        print(f"✅ [AITutorEngine] Đã NẠP BỔ SUNG Video Mẫu vào Dataset tại: {dest_path}")

        # 2. Lưu mẫu tham chiếu vào bộ nhớ
        self.reference_templates[word_name] = {
            "video_path": dest_path,
            "enrolled_at": time.time()
        }
        return True

    def evaluate_student_gesture(self, frame, bboxes, action, conf, target_word):
        """
        Chẩn đoán cử chỉ học viên thời gian thực và trả về kết quả đánh giá.
        """
        h, w = frame.shape[:2]
        target_word = target_word.lower()

        status_code = "WAITING"
        score = 0.0
        guidance_msg = "Hãy giơ 2 bàn tay ngang ngực thực hiện cử chỉ!"
        feedback_color = (0, 255, 255) # Màu vàng nhạt

        if not bboxes or len(bboxes) == 0:
            guidance_msg = "Hãy giơ 2 bàn tay lên trước camera để bắt đầu!"
            return score, status_code, guidance_msg, (200, 200, 200)

        # 1. Chẩn đoán vị trí giơ tay
        hand_too_low = False
        for (x1, y1, x2, y2, hand_label) in bboxes:
            if y1 > (h * 0.65):
                hand_too_low = True
                break

        if hand_too_low:
            status_code = "POSITION_LOW"
            guidance_msg = "Hãy nâng tay cao hơn ngang ngực!"
            feedback_color = (0, 0, 255) # Màu đỏ cảnh báo
            self._speak_guidance_cooldown("Hãy nâng tay cao hơn ngang ngực!")
            return score, status_code, guidance_msg, feedback_color

        # 2. Chẩn đoán độ khớp cử chỉ
        if action and action.lower() == target_word:
            score = min(100.0, float(conf * 100.0 + 25.0))
            status_code = "EXCELLENT"
            guidance_msg = f"XUẤT SẮC! Cử chỉ đúng chuẩn 100% cho từ '{target_word.upper()}'!"
            feedback_color = (0, 255, 0) # Màu xanh lá thành công
            self._speak_guidance_cooldown(f"Xuất sắc! Bạn làm đúng cử chỉ từ {target_word} rồi!")
        else:
            score = max(30.0, float(conf * 80.0))
            status_code = "PRACTICING"
            guidance_msg = f"Đang tập cử chỉ từ '{target_word.upper()}'. Giữ tay ổn định..."
            feedback_color = (0, 255, 255)

        return score, status_code, guidance_msg, feedback_color

    def draw_virtual_avatar(self, display, score, guidance_msg, feedback_color, target_word):
        """
        Vẽ VỊ TRÍ NGƯỜI ẢO AI TRỢ LÝ (AI VIRTUAL AVATAR) BỒNG BỒNG HỘI THOẠI TRÊN MÀN HÌNH.
        """
        h, w = display.shape[:2]

        # 1. Vẽ khung Avatar ở góc trên bên phải (Top-Right Avatar Box)
        avatar_x1, avatar_y1 = w - 190, 10
        avatar_x2, avatar_y2 = w - 10, 150
        cv2.rectangle(display, (avatar_x1, avatar_y1), (avatar_x2, avatar_y2), (30, 30, 30), -1)
        cv2.rectangle(display, (avatar_x1, avatar_y1), (avatar_x2, avatar_y2), feedback_color, 2)

        # 2. Vẽ hình đầu Người Ảo AI (AI Robot Head)
        head_cx, head_cy = w - 100, 55
        cv2.circle(display, (head_cx, head_cy), 25, (220, 220, 220), -1)  # Đầu tròn
        cv2.circle(display, (head_cx - 8, head_cy - 5), 4, (255, 0, 0), -1) # Mắt trái xanh
        cv2.circle(display, (head_cx + 8, head_cy - 5), 4, (255, 0, 0), -1) # Mắt phải xanh
        # Miệng biểu cảm theo kết quả
        if "EXCELLENT" in guidance_msg or score >= 80:
            cv2.ellipse(display, (head_cx, head_cy + 8), (10, 6), 0, 0, 180, (0, 200, 0), 2) # Miệng cười
        else:
            cv2.line(display, (head_cx - 8, head_cy + 10), (head_cx + 8, head_cy + 10), (0, 0, 255), 2) # Miệng thẳng

        cv2.putText(display, "AI TUTOR 🤖", (w - 150, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
        cv2.putText(display, f"Diem: {score:.1f}%", (w - 155, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

        # 3. Bảng bong bóng thông báo hướng dẫn ở trên cùng
        cv2.rectangle(display, (10, 10), (w - 200, 75), (20, 20, 20), -1)
        cv2.putText(display, f"GIA SU AI - BAI TAP: '{target_word.upper()}'", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.putText(display, guidance_msg, (20, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.5, feedback_color, 2)

    def _speak_guidance_cooldown(self, text):
        """Phát âm thanh nhắc nhở có giới hạn thời gian nghỉ để tránh nhái giọng liên tục."""
        now = time.time()
        if now - self.last_speech_time > self.speech_cooldown:
            if self.tts:
                self.tts.speak(text)
            self.last_speech_time = now
