# Demo_ui/App.py
import cv2
import time
import os
import sys
import threading

# Thêm đường dẫn dự án
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

from Shared_lib.predictor import SignLanguagePredictor
from Cloud_server.Api.context_agent import ContextAgent
from Cloud_server.Api.llm_corrector import LLMCorrector
from Demo_ui.Utils import SubtitleRenderer, TextToSpeech, VirtualCamera, draw_hand_badge

class SignLanguageDemoUI:
    """
    Ứng dụng Demo nhận diện Ngôn ngữ ký hiệu giao diện cao cấp (Netflix/YouTube Subtitle UI)
    Phục vụ cho quay Video giới thiệu và Thuyết trình báo cáo thực tập / Đồ án.
    """
    def __init__(self):
        print("🚀 Khởi tạo Mô hình AI & Giao diện Demo...", flush=True)
        self.predictor = SignLanguagePredictor(
            yolo_path=os.path.join(BASE_DIR, "mobile_app", "assets", "hand_det_yolo.tflite"),
            feature_path=os.path.join(BASE_DIR, "mobile_app", "assets", "feature_extractor.tflite"),
            gru_path=os.path.join(BASE_DIR, "mobile_app", "assets", "action_recognizer.tflite")
        )
        
        self.subtitle_renderer = SubtitleRenderer(font_size=24)
        self.tts = TextToSpeech()
        self.virtual_cam = VirtualCamera()

        self.llm_corrector = LLMCorrector() if LLMCorrector else None
        self.context_agent = ContextAgent(llm_corrector=self.llm_corrector) if ContextAgent else None

        self.sentence_words = []
        self.final_sentence = ""
        self.last_action = None
        self.last_hand_time = time.time()
        self.PAUSE_TIMEOUT = 2.0

        self.last_process_time = 0
        self.process_interval = 0.08  # 12.5 FPS cho AI, camera mượt 30 FPS
        self._cached_static_char = None
        self._cached_action = None
        self._cached_conf = 0.0
        self._cached_bboxes = []

    def _async_translate_and_speak(self, raw_sentence, lang):
        """Xử lý dịch thuật LLM và đọc phát âm ngầm (Async Background Thread)"""
        try:
            if self.context_agent:
                translated = self.context_agent.process(action_word=raw_sentence, target_lang=lang)
            else:
                translated = raw_sentence
            
            self.final_sentence = translated
            if translated:
                print(f"🔊 [Netflix Subtitle Async] Đang đọc phát âm câu dịch: '{translated}'", flush=True)
                self.tts.speak(translated)
        except Exception as e:
            print(f"⚠️ Lỗi luồng ngầm dịch thuật: {e}", flush=True)

    def run(self):
        print("\n=======================================================", flush=True)
        print("  🎬 HỆ THỐNG DỊCH NGÔN NGỮ KÝ HIỆU - DEMO UI NETFLIX", flush=True)
        print("=======================================================\n", flush=True)

        print("--- [BẢNG CHỌN NGÔN NGỮ DỊCH THUẬT BAN ĐẦU] ---", flush=True)
        print(" [V] Tiếng Việt (Vietnamese - Mặc định)")
        print(" [E] Tiếng Anh (English)")
        print(" [J] Tiếng Nhật (Japanese)")
        print(" [K] Tiếng Hàn (Korean)")
        l_choice = input("👉 Nhập Lựa Chọn Ngôn Ngữ Mong Muốn (V/E/J/K, ấn Enter để chọn Việt): ").strip().lower()
        lang_map = {'v': 'vi', 'e': 'en', 'j': 'ja', 'k': 'ko'}
        current_lang = lang_map.get(l_choice, 'vi')
        lang_names = {'vi': 'TIẾNG VIỆT', 'en': 'TIẾNG ANH', 'ja': 'TIẾNG NHẬT', 'ko': 'TIẾNG HÀN'}

        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        print(f"\n🚀 Đã khởi tạo ngôn ngữ: {lang_names.get(current_lang, 'TIẾNG VIỆT')}", flush=True)
        print("Phím tắt điều khiển: [Q] Thoát | [V] Việt | [E] Anh | [J] Nhật | [K] Hàn | [C] Xóa câu\n", flush=True)

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # 1. AI FPS Throttling (Chạy AI ở 12.5 FPS, camera 30 FPS)
            now = time.time()
            if now - self.last_process_time >= self.process_interval:
                self.last_process_time = now
                static_char, action, conf, bboxes = self.predictor.process_frame(frame)
                self._cached_static_char = static_char
                self._cached_action = action
                self._cached_conf = conf
                self._cached_bboxes = bboxes
            else:
                static_char = self._cached_static_char
                action = self._cached_action
                conf = self._cached_conf
                bboxes = self._cached_bboxes

            display = frame.copy()
            active_label = static_char or action

            # 2. Vẽ Bounding box bàn tay & Thẻ nhãn chữ cái bo góc
            if bboxes and len(bboxes) > 0:
                self.last_hand_time = time.time()
                display = draw_hand_badge(display, bboxes, active_label)

            # 3. Gom từ mới khi AI nhận diện được nhãn
            if active_label and active_label != self.last_action:
                self.sentence_words.append(active_label)
                self.last_action = active_label
                self.last_hand_time = time.time()
                print(f"✨ AI Nhận diện được: [{active_label}] (Conf: {conf:.2f})", flush=True)

            # 4. Khi dừng tay > 2.0s -> Dịch câu và đọc phát âm ngầm
            if len(self.sentence_words) > 0 and (time.time() - self.last_hand_time > self.PAUSE_TIMEOUT):
                raw_sentence = " ".join(self.sentence_words)
                self.sentence_words = []
                threading.Thread(target=self._async_translate_and_speak, args=(raw_sentence, current_lang), daemon=True).start()

            # 5. VẼ PHỤ ĐỀ NETFLIX STYLE TIẾNG VIỆT CÓ DẤU NÉT CĂNG
            display_text = self.final_sentence if self.final_sentence else " ".join(self.sentence_words)
            if display_text:
                display = self.subtitle_renderer.draw_subtitle(display, display_text, lang_names.get(current_lang, 'TIẾNG VIỆT'))

            # 6. Truyền sang Webcam Ảo (nếu có) và Hiển thị cửa sổ OpenCV duy nhất
            self.virtual_cam.send_frame(display)
            cv2.imshow("Sign Language AI - Demo UI (Netflix Subtitle)", display)

            # 7. Xử lý phím tắt
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('v'):
                current_lang = 'vi'
                print("🔄 Đã chuyển sang Tiếng Việt", flush=True)
            elif key == ord('e'):
                current_lang = 'en'
                print("🔄 Đã chuyển sang Tiếng Anh", flush=True)
            elif key == ord('j'):
                current_lang = 'ja'
                print("🔄 Đã chuyển sang Tiếng Nhật", flush=True)
            elif key == ord('k'):
                current_lang = 'ko'
                print("🔄 Đã chuyển sang Tiếng Hàn", flush=True)
            elif key == ord('c'):
                self.sentence_words.clear()
                self.final_sentence = ""
                self.last_action = None
                print("🧹 Đã xóa toàn bộ câu tạm thời", flush=True)

        cap.release()
        self.virtual_cam.close()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    app = SignLanguageDemoUI()
    app.run()
