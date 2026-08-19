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
        self.tts_enabled = True  # Cờ Bật / Tắt âm thanh loa đọc phát âm

        self.llm_corrector = LLMCorrector() if LLMCorrector else None
        self.context_agent = ContextAgent(llm_corrector=self.llm_corrector) if ContextAgent else None

        self.spelled_chars = []
        self.action_words = []
        self.final_sentence = ""
        self.last_action = None
        self.last_hand_time = time.time()
        self.PAUSE_TIMEOUT = 2.0
        self.tts_notification = ""
        self.tts_notification_time = 0

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
                if self.tts_enabled:
                    print(f"🔊 [Netflix Subtitle Async] Đang đọc phát âm câu dịch: '{translated}'", flush=True)
                    self.tts.speak(translated)
                else:
                    print(f"🔇 [Loa TẮT] Đã dịch: '{translated}' (Không phát âm thanh)", flush=True)
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
        print("Phím tắt điều khiển: [Q] Thoát | [M] Bật/Tắt Loa | [V] Việt | [E] Anh | [J] Nhật | [K] Hàn | [C] Xóa câu\n", flush=True)

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
            h, w = display.shape[:2]
            active_label = static_char or action

            # Hiển thị thanh trạng thái góc trên (Ngôn ngữ + Loa phát âm)
            cv2.rectangle(display, (10, 10), (220, 42), (20, 20, 20), -1)
            cv2.putText(display, f"LANG: {lang_names.get(current_lang, 'VI')}", (15, 31),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 255, 255), 2)

            speaker_text = "LOA: BAT [M]" if self.tts_enabled else "LOA: TAT [M]"
            speaker_color = (0, 220, 0) if self.tts_enabled else (0, 0, 220)
            cv2.rectangle(display, (230, 10), (380, 42), (20, 20, 20), -1)
            cv2.putText(display, speaker_text, (235, 31),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.50, speaker_color, 2)

            # Pop-up thông báo nhanh 1.5s khi bấm phím M
            if self.tts_notification and (time.time() - self.tts_notification_time < 1.5):
                cv2.rectangle(display, (w // 2 - 120, 60), (w // 2 + 120, 110), (20, 20, 20), -1)
                pop_color = (0, 255, 0) if self.tts_enabled else (0, 0, 255)
                cv2.putText(display, self.tts_notification, (w // 2 - 95, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.8, pop_color, 2)

            # 2. Vẽ Bounding box bàn tay & Thẻ nhãn chữ cái bo góc
            if bboxes and len(bboxes) > 0:
                self.last_hand_time = time.time()
                display = draw_hand_badge(display, bboxes, active_label)

            # 3. Gom riêng Chữ cái A-Z và Từ vựng GRU (Tách 2 bộ đệm)
            if active_label and active_label != self.last_action:
                if len(active_label) == 1 and active_label.isalpha():
                    self.spelled_chars.append(active_label)
                else:
                    self.action_words.append(active_label)
                self.last_action = active_label
                self.last_hand_time = time.time()
                print(f"✨ AI Nhận diện được: [{active_label}] (Conf: {conf:.2f})", flush=True)

            # 4. Khi dừng tay > 2.0s -> Dịch câu và đọc phát âm ngầm (Ưu tiên từ vựng trước, đánh vần sau)
            if (time.time() - self.last_hand_time > self.PAUSE_TIMEOUT):
                if len(self.action_words) > 0:
                    raw_sentence = " ".join(self.action_words)
                    self.action_words = []
                    threading.Thread(target=self._async_translate_and_speak, args=(raw_sentence, current_lang), daemon=True).start()
                elif len(self.spelled_chars) > 0:
                    raw_sentence = " ".join(self.spelled_chars)
                    self.spelled_chars = []
                    threading.Thread(target=self._async_translate_and_speak, args=(raw_sentence, current_lang), daemon=True).start()

            # 5. VẼ PHỤ ĐỀ NETFLIX STYLE TIẾNG VIỆT CÓ DẤU NÉT CĂNG
            display_text = ""
            if self.final_sentence:
                display_text = self.final_sentence
            elif self.action_words:
                display_text = f"Từ vựng: {' '.join(self.action_words)}"
            elif self.spelled_chars:
                display_text = f"Đánh vần: {' '.join(self.spelled_chars)}"

            if display_text:
                display = self.subtitle_renderer.draw_subtitle(display, display_text, lang_names.get(current_lang, 'TIẾNG VIỆT'))

            # 6. Truyền sang Webcam Ảo (nếu có) và Hiển thị cửa sổ OpenCV duy nhất
            self.virtual_cam.send_frame(display)
            cv2.imshow("Sign Language AI - Demo UI (Netflix Subtitle)", display)

            # 7. Xử lý phím tắt
            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), ord('Q')):
                break
            elif key in (ord('m'), ord('M')):
                self.tts_enabled = not self.tts_enabled
                status_str = "BẬT 🔊" if self.tts_enabled else "TẮT 🔇"
                self.tts_notification = f"LOA: {status_str}"
                self.tts_notification_time = time.time()
                print(f"🔄 Đã chuyển trạng thái Loa: {status_str}", flush=True)
            elif key in (ord('v'), ord('V')):
                current_lang = 'vi'
                print("🔄 Đã chuyển sang Tiếng Việt", flush=True)
            elif key in (ord('e'), ord('E')):
                current_lang = 'en'
                print("🔄 Đã chuyển sang Tiếng Anh", flush=True)
            elif key in (ord('j'), ord('J')):
                current_lang = 'ja'
                print("🔄 Đã chuyển sang Tiếng Nhật", flush=True)
            elif key in (ord('k'), ord('K')):
                current_lang = 'ko'
                print("🔄 Đã chuyển sang Tiếng Hàn", flush=True)
            elif key in (ord('c'), ord('C')):
                self.spelled_chars.clear()
                self.action_words.clear()
                self.final_sentence = ""
                self.last_action = None
                print("🧹 Đã xóa toàn bộ câu tạm thời", flush=True)

        cap.release()
        self.virtual_cam.close()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    app = SignLanguageDemoUI()
    app.run()
