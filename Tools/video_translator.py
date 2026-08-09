# Tools/video_translator.py
"""
==============================================================================
MODULE DỊCH THUẬT TỆP VIDEO (BATCH VIDEO TRANSLATION ENGINE)
==============================================================================
Nhiệm vụ:
1. Nhận tệp video MP4/AVI từ đĩa cứng.
2. Đọc từng khung hình, trích xuất bounding box bàn tay và chạy mô hình AI.
3. Gom từ ngữ, dùng ContextAgent tổng hợp thành câu có nghĩa.
4. Hiển thị xem trước (Preview) kèm thanh phụ đề và phát đọc âm thanh TTS.
==============================================================================
"""

import os
import sys
import time
import cv2

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

class VideoTranslator:
    def __init__(self, predictor, context_agent=None, tts=None):
        self.predictor = predictor
        self.context_agent = context_agent
        self.tts = tts

    def translate_video(self, video_path, target_lang='vi', show_preview=True):
        """Xử lý dịch thuật tệp video và trả về kết quả."""
        if not os.path.exists(video_path):
            print(f"❌ [VideoTranslator] Không tìm thấy file: {video_path}")
            return None

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"❌ [VideoTranslator] Không thể mở file video: {video_path}")
            return None

        print(f"\n🎬 [VideoTranslator] Đang tiến hành phân tích video: '{os.path.basename(video_path)}'...")
        
        sentence_words = []
        last_action = None
        final_sentence = ""

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            static_char, action, conf, bboxes = self.predictor.process_frame(frame)
            display = frame.copy()
            h, w = display.shape[:2]

            # 1. Vẽ ô vuông bàn tay
            if bboxes:
                for (x1, y1, x2, y2, hand_label) in bboxes:
                    color = (0, 255, 0) if "Left" in hand_label else (255, 0, 0)
                    cv2.rectangle(display, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(display, hand_label, (x1, max(15, y1 - 5)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            # 2. Tích lũy từ vựng
            if action and action != last_action:
                sentence_words.append(action)
                last_action = action

            # 3. Dịch ngữ cảnh
            if len(sentence_words) > 0:
                raw_sentence = " ".join(sentence_words)
                if self.context_agent:
                    final_sentence = self.context_agent.process(action_word=raw_sentence, target_lang=target_lang)
                else:
                    final_sentence = raw_sentence

            # 4. Hiển thị thanh phụ đề
            if final_sentence and show_preview:
                cv2.rectangle(display, (0, h - 65), (w, h), (0, 0, 0), -1)
                cv2.putText(display, f"Ket qua dich Video: {final_sentence}", (20, h - 22),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 2)

            if show_preview:
                cv2.imshow("Video Translation Engine", display)
                if cv2.waitKey(30) & 0xFF in (ord('q'), ord('Q')):
                    break

        cap.release()
        if show_preview:
            cv2.destroyAllWindows()

        print(f"✅ [VideoTranslator] Dịch hoàn tất: '{final_sentence}'")
        if final_sentence and self.tts:
            self.tts.speak(final_sentence)

        return {
            "video_path": video_path,
            "raw_words": sentence_words,
            "translated_sentence": final_sentence
        }
