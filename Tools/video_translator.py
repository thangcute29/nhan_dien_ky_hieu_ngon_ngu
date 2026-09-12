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

if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

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
        if not video_path:
            print("❌ [VideoTranslator] Đường dẫn file video rỗng!")
            return None

        # Tự động loại bỏ dấu nháy kép / nháy đơn thừa do copy-paste đường dẫn trên Windows
        video_path = video_path.strip('\'" \t\r\n')

        if not os.path.exists(video_path):
            print(f"❌ [VideoTranslator] Không tìm thấy file: {video_path}")
            return None

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"❌ [VideoTranslator] Không thể mở file video: {video_path}")
            return None

        print(f"\n🎬 [VideoTranslator] Đang tiến hành phân tích video: '{os.path.basename(video_path)}'...")
        
        if hasattr(self.predictor, 'enable_video_mode'):
            self.predictor.enable_video_mode()
        
        sentence_words = []
        last_action = None
        final_sentence = ""
        last_word_ts = 0.0
        last_translated_count = 0

        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0 or fps != fps: # Check nan or <= 0
            fps = 30.0
        frame_skip = max(1, int(fps / 10)) # ~10 FPS sampling

        # Xử lý với tốc độ khung hình tối đa (fast-forward) thay vì chờ bằng thời gian thực
        wait_time = 1
        frame_count = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            frame_count += 1
            current_ts = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0

            if frame_count % frame_skip != 0:
                continue

            static_char, action, conf, bboxes = self.predictor.process_frame(
                frame, video_mode=True
            )
            display = frame.copy()
            h, w = display.shape[:2]

            active_label = action if action else static_char

            # 1. Vẽ ô vuông bàn tay
            if bboxes:
                for (x1, y1, x2, y2, hand_label) in bboxes:
                    color = (0, 255, 0) if "Left" in hand_label else (255, 0, 0)
                    cv2.rectangle(display, (x1, y1), (x2, y2), color, 2)
                    label_text = hand_label
                    if active_label:
                        label_text += f": [{active_label}]"
                    cv2.putText(display, label_text, (x1, max(15, y1 - 5)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            # 2. Tích lũy từ vựng / ký tự
            if active_label and active_label != last_action:
                sentence_words.append(active_label)
                last_action = active_label
                last_word_ts = current_ts
                print(f"✨ [Video AI] Nhận diện được: [{active_label}] (Conf: {conf:.2f} tại {current_ts:.1f}s)", flush=True)
            elif not active_label and current_ts - last_word_ts >= 0.7:
                # A real pause permits the same sign to occur twice (e.g. L-L).
                last_action = None
            elif not bboxes and frame_count % int(fps * 2) == 0:
                # In cảnh báo mù (không thấy tay) mỗi ~2 giây
                print("⚠️ [Video AI] LOST HANDS - Không tìm thấy bàn tay trong video (Vui lòng chọn video rõ nét hơn)!", flush=True)

            # 3. Dịch ngữ cảnh (Gap Detection)
            if len(sentence_words) > last_translated_count and (current_ts - last_word_ts >= 2.0):
                raw_sentence = " ".join(sentence_words)
                if self.context_agent:
                    print(f"🔄 [VideoTranslator] Đang dịch chuỗi: '{raw_sentence}'...", flush=True)
                    final_sentence = self.context_agent.process(action_word=raw_sentence, target_lang=target_lang)
                else:
                    final_sentence = raw_sentence
                last_translated_count = len(sentence_words)

            # 4. Hiển thị thanh phụ đề (Xóa Debug trên màn hình)
            if show_preview:
                # Thu nhỏ kích thước để đảm bảo nhìn thấy toàn bộ cửa sổ (tránh bị che khuất viền dưới)
                display_small = cv2.resize(display, (800, int(800 * h / w)))
                h_s, w_s = display_small.shape[:2]

                # Vẽ thanh phụ đề màu đen ở dưới cùng
                cv2.rectangle(display_small, (0, h_s - 60), (w_s, h_s), (0, 0, 0), -1)
                sub_text = f"Dich ({target_lang.upper()}): {final_sentence}" if final_sentence else (f"Tu: {' '.join(sentence_words)}" if sentence_words else "Dang phan tich video...")
                cv2.putText(display_small, sub_text, (20, h_s - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

                cv2.imshow("Video Translation Engine", display_small)
                if cv2.waitKey(wait_time) & 0xFF in (ord('q'), ord('Q')):
                    break

        cap.release()
        if hasattr(self.predictor, 'disable_video_mode'):
            self.predictor.disable_video_mode()
        
        # Dịch nốt câu cuối cùng nếu video kết thúc mà chưa đủ thời gian timeout
        if len(sentence_words) > last_translated_count:
            raw_sentence = " ".join(sentence_words)
            if self.context_agent:
                print(f"🔄 [VideoTranslator] Đang dịch chuỗi cuối: '{raw_sentence}'...", flush=True)
                final_sentence = self.context_agent.process(action_word=raw_sentence, target_lang=target_lang)
            else:
                final_sentence = raw_sentence
                
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
