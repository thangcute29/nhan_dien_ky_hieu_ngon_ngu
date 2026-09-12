# mobile_app/src/main.py
"""
==============================================================================
HỆ THỐNG GIAO TIẾP & DỊCH THUẬT NGÔN NGỮ KÝ HIỆU AI (COMMERCIAL PRO SUITE)
==============================================================================
Bao gồm 5 Chức Năng Thương Mại Hoàn Chỉnh:
[1] 🗣️ Giao Tiếp 2 Chiều Thông Minh (Voice <-> Sign Two-Way Communication)
[2] 🎥 Cuộc Gọi Video Trực Tuyến Cho Zoom / Google Meet / Teams (Virtual Cam HD)
[3] ✍️ Bảng Soạn Thảo Văn Bản & Dịch Thuật Ký Hiệu (Sign Text Editor & Live)
[4] 🎓 Gia Sư AI Pro Chấm Điểm 3D Màn Hình Đôi & Nạp Video Mẫu (AI Tutor Duo-Screen)
[5] 📁 Dịch Thuật Từ Tệp Video MP4 / AVI (Video File Translation Engine)
==============================================================================
"""

import cv2
import requests
import threading
import time
import os
import sys

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

try:
    from Shared_lib.predictor import SignLanguagePredictor
except ImportError:
    from inference_engine.predictor import SignLanguagePredictor

try:
    from Cloud_server.Api.llm_corrector import LLMCorrector
except ImportError:
    LLMCorrector = None

try:
    from Cloud_server.Api.context_agent import ContextAgent
except ImportError:
    ContextAgent = None

try:
    from Cloud_server.Api.llm_translator import LLMTranslator
except ImportError:
    LLMTranslator = None

try:
    from Shared_lib.ui_helpers import TextToSpeech, VirtualCamera
except ImportError:
    from Demo_ui.Utils import TextToSpeech, VirtualCamera

try:
    from Tools.video_translator import VideoTranslator
    from Tools.ai_tutor_engine import AITutorEngine
    from Tools.voice_to_sign_engine import VoiceToSignEngine
except ImportError:
    VideoTranslator = None
    AITutorEngine = None
    VoiceToSignEngine = None

CONFIDENCE_THRESHOLD = 0.6
SERVER_URL = "http://localhost:8000"

class SignLanguageApp:
    def __init__(self):
        self.predictor = SignLanguagePredictor(
            feature_path=os.path.join(BASE_DIR, "Mobile_app", "assets", "feature_extractor.tflite"),
            gru_path=os.path.join(BASE_DIR, "Mobile_app", "assets", "action_recognizer.tflite")
        )
        self.tts = TextToSpeech()
        self.virtual_cam = VirtualCamera()
        self.cap = None
        self.tts_enabled = True  # Cờ Bật / Tắt âm thanh loa đọc phát âm

        # Khởi tạo LLMCorrector, ContextAgent & Translator
        self.llm_corrector = LLMCorrector() if LLMCorrector else None
        self.context_agent = ContextAgent(llm_corrector=self.llm_corrector) if ContextAgent else None
        self.translator = LLMTranslator(llm_corrector=self.llm_corrector) if LLMTranslator else None

        # Khởi tạo các Module Chức Năng Thương Mại
        self.video_translator = VideoTranslator(predictor=self.predictor, context_agent=self.context_agent, tts=self.tts) if VideoTranslator else None
        self.ai_tutor_engine = AITutorEngine(tts=self.tts) if AITutorEngine else None
        self.voice_to_sign_engine = VoiceToSignEngine(tts=self.tts) if VoiceToSignEngine else None

        self.last_action = None
        self._inactive_since = None
        self._translation_generation = 0
        self.sentence_words = []
        self.last_hand_time = time.time()
        self.PAUSE_TIMEOUT = 2.0
        self.final_sentence = ""

        # Giới hạn tốc độ xử lý AI ở 12.5 FPS (80ms), camera hiển thị 30 FPS mượt mà
        self.last_process_time = 0
        self.process_interval = 0.08
        self._cached_static_char = None
        self._cached_action = None
        self._cached_conf = 0.0
        self._cached_bboxes = []

    def _async_translate_and_speak(self, raw_sentence, lang, generation):
        """Xử lý dịch thuật LLM và phát âm thanh TTS ở luồng ngầm"""
        try:
            if self.context_agent:
                translated = self.context_agent.process(action_word=raw_sentence, target_lang=lang)
            else:
                translated = raw_sentence
            
            if generation != self._translation_generation:
                return
            self.final_sentence = translated
            if translated:
                if self.tts_enabled:
                    print(f"🔊 [Async] Đang đọc phát âm câu dịch: '{translated}'", flush=True)
                    self.tts.speak(translated)
                else:
                    print(f"🔇 [Loa TẮT] Đã dịch: '{translated}' (Không phát âm thanh)", flush=True)
                
                import threading
                threading.Timer(5.0, self.clear_sentence).start()
        except Exception as e:
            print(f"⚠️ Lỗi luồng ngầm dịch thuật: {e}", flush=True)

    def clear_sentence(self):
        self.final_sentence = ""

    def run_two_way_communication(self):
        """Chức năng [1]: Giao Tiếp 2 Chiều Thông Minh (Voice <-> Sign)"""
        if self.voice_to_sign_engine:
            self.voice_to_sign_engine.run_two_way_interface(
                predictor=self.predictor,
                context_agent=self.context_agent,
                tts=self.tts,
                cap=self.cap
            )
        else:
            print("⚠️ Module VoiceToSignEngine chưa sẵn sàng.")

    def run_virtual_cam_meeting(self):
        """Chức năng [2]: Cuộc Gọi Video Trực Tuyến Cho Zoom / Google Meet / Teams"""
        print("\n=== [2] CUỘC GỌI TRỰC TUYẾN ZOOM / MEET (VIRTUAL CAMERA HD) ===", flush=True)
        print("\n--- [BẢNG CHỌN NGÔN NGỮ DỊCH THUẬT BAN ĐẦU] ---", flush=True)
        print(" [V] Tiếng Việt (Vietnamese - Mặc định)")
        print(" [E] Tiếng Anh (English)")
        print(" [J] Tiếng Nhật (Japanese)")
        print(" [K] Tiếng Hàn (Korean)")
        l_choice = input("👉 Nhập Lựa Chọn Ngôn Ngữ (V/E/J/K, ấn Enter để chọn Việt): ").strip().lower()
        lang_map = {'v': 'vi', 'e': 'en', 'j': 'ja', 'k': 'ko'}
        self.current_lang = lang_map.get(l_choice, 'vi')
        
        print(f"\n🚀 Đã khởi tạo ngôn ngữ: {self.current_lang.upper()}", flush=True)
        print("Phím tắt điều khiển: [Q] Thoát | [M] Bật/Tắt Loa | [V] Việt | [E] Anh | [J] Nhật | [K] Hàn\n", flush=True)

        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        self.spelled_chars = []
        self.action_words = []
        self.final_sentence = ""
        self.last_action = None
        self.tts_notification = ""
        self.tts_notification_time = 0

        while True:
            ret, frame = self.cap.read()
            if not ret:
                break

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

            lang = getattr(self, 'current_lang', 'vi')
            lang_names = {'vi': 'TIENG VIET (Phim V)', 'en': 'ENGLISH (Phim E)', 'ja': 'JAPANESE (Phim J)', 'ko': 'KOREAN (Phim K)'}
            lang_str = lang_names.get(lang, 'TIENG VIET (Phim V)')
            cv2.rectangle(display, (10, 10), (320, 45), (0, 0, 0), -1)
            cv2.putText(display, f"NGON NGU: {lang_str}", (15, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)

            speaker_status = "LOA: BAT [Phim M]" if self.tts_enabled else "LOA: TAT [Phim M]"
            speaker_color = (0, 200, 0) if self.tts_enabled else (0, 0, 200)
            cv2.rectangle(display, (330, 10), (510, 45), (0, 0, 0), -1)
            cv2.putText(display, speaker_status, (335, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.50, speaker_color, 2)

            if self.tts_notification and (time.time() - self.tts_notification_time < 1.5):
                cv2.rectangle(display, (w // 2 - 120, 60), (w // 2 + 120, 110), (20, 20, 20), -1)
                pop_color = (0, 255, 0) if self.tts_enabled else (0, 0, 255)
                cv2.putText(display, self.tts_notification, (w // 2 - 95, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.8, pop_color, 2)

            # A completed dynamic sign has priority over an incidental static pose.
            active_label = action or static_char
            if bboxes and len(bboxes) > 0:
                self.last_hand_time = time.time()
                for (x1, y1, x2, y2, hand_label) in bboxes:
                    color = (0, 255, 0) if "Left" in hand_label else (255, 0, 0)
                    cv2.rectangle(display, (x1, y1), (x2, y2), color, 2)
                    display_text = hand_label
                    if active_label:
                        display_text += f": [{active_label}]"
                    cv2.rectangle(display, (x1, max(0, y1 - 25)), (x1 + len(display_text) * 11, y1), color, -1)
                    cv2.putText(display, display_text, (x1 + 3, max(15, y1 - 7)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

            if active_label and active_label != self.last_action:
                if len(active_label) == 1 and active_label.isalpha():
                    self.spelled_chars.append(active_label)
                else:
                    self.action_words.append(active_label)
                self.last_action = active_label
                self.last_hand_time = time.time()
                print(f"✨ AI Nhận diện được: [{active_label}] (Conf: {conf:.2f})", flush=True)
            if active_label:
                self._inactive_since = None
            elif self._inactive_since is None:
                self._inactive_since = time.time()
            elif time.time() - self._inactive_since >= 0.35:
                self.last_action = None

            if (time.time() - self.last_hand_time > self.PAUSE_TIMEOUT):
                if len(self.action_words) > 0:
                    raw_sentence = " ".join(self.action_words)
                    self.action_words = []
                    self.last_action = None
                    self._translation_generation += 1
                    threading.Thread(target=self._async_translate_and_speak, args=(raw_sentence, lang, self._translation_generation), daemon=True).start()
                elif len(self.spelled_chars) > 0:
                    raw_sentence = " ".join(self.spelled_chars)
                    self.spelled_chars = []
                    self.last_action = None
                    self._translation_generation += 1
                    threading.Thread(target=self._async_translate_and_speak, args=(raw_sentence, lang, self._translation_generation), daemon=True).start()

            cv2.rectangle(display, (0, h - 80), (w, h), (0, 0, 0), -1)
            tu_vung_text = f"Tu vung: {' '.join(self.action_words)}" if self.action_words else "Tu vung: (Dang cho ky hieu...)"
            danh_van_text = f"Danh van: {' '.join(self.spelled_chars)}"

            if self.final_sentence:
                tu_vung_text = f"Dich ({lang.upper()}): {self.final_sentence}"

            # Giới hạn độ dài chữ để không bị tràn màn hình
            text_size = cv2.getTextSize(tu_vung_text, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)[0]
            if text_size[0] > w - 40:
                while text_size[0] > w - 60 and len(tu_vung_text) > 5:
                    tu_vung_text = tu_vung_text[:-1]
                    text_size = cv2.getTextSize(tu_vung_text + "...", cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)[0]
                tu_vung_text += "..."

            cv2.putText(display, tu_vung_text, (20, h - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)
            cv2.putText(display, danh_van_text, (20, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)

            # Truyền khung hình đã vẽ phụ đề sang Webcam Ảo (Zoom, Google Meet)
            self.virtual_cam.send_frame(display)
            cv2.imshow("Sign Language AI - Zoom & Google Meet Virtual Camera", display)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), ord('Q')):
                break
            elif key in (ord('m'), ord('M')):
                self.tts_enabled = not self.tts_enabled
                status_str = "BẬT 🔊" if self.tts_enabled else "TẮT 🔇"
                self.tts_notification = f"LOA: {status_str}"
                self.tts_notification_time = time.time()
                print(f"🔄 Đã chuyển trạng thái Loa: {status_str}", flush=True)
            elif key in (ord('v'), ord('V')): self.current_lang = 'vi'
            elif key in (ord('e'), ord('E')): self.current_lang = 'en'
            elif key in (ord('j'), ord('J')): self.current_lang = 'ja'
            elif key in (ord('k'), ord('K')): self.current_lang = 'ko'

            time.sleep(0.03)

        self.cap.release()
        cv2.destroyAllWindows()

    def run_sign_text_editor(self):
        """Chức năng [3]: Bảng Soạn thảo Văn bản bằng Ngôn ngữ Ký hiệu (Sign Text Editor)"""
        print("\n=== [3] BẢNG SOẠN THẢO VĂN BẢN KÝ HIỆU (SIGN TEXT EDITOR) ===", flush=True)
        print("Phím thao tác: [Space] Nhập từ | [D] Xóa từ cuối | [C] Xóa hết | [S] Lưu file | [T] Đọc TTS | [Q] Thoát", flush=True)

        self.cap = cv2.VideoCapture(0)
        composed_words = []
        current_word = ""

        while True:
            ret, frame = self.cap.read()
            if not ret:
                break

            static_char, action, conf, bboxes = self.predictor.process_frame(frame)
            display = frame.copy()
            h, w = display.shape[:2]

            if action:
                current_word = action

            if bboxes:
                for (x1, y1, x2, y2, hand_label) in bboxes:
                    color = (0, 255, 0) if "Left" in hand_label else (255, 0, 0)
                    cv2.rectangle(display, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(display, hand_label, (x1, max(15, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            cv2.rectangle(display, (0, 0), (w, 80), (30, 30, 30), -1)
            full_text = " ".join(composed_words) + (f" [{current_word}]" if current_word else "")
            cv2.putText(display, f"VAN BAN SOAN THAO: {full_text}", (15, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(display, "Phim: [Space] Them | [D] Xoa | [C] Xoa het | [S] Luu | [T] Doc | [Q] Ra", (15, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

            cv2.imshow("Sign Text Editor", display)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), ord('Q')):
                break
            elif key == ord(' '):
                if current_word:
                    composed_words.append(current_word)
                    print(f"➕ Đã thêm từ: '{current_word}'")
            elif key in (ord('d'), ord('D')):
                if composed_words:
                    popped = composed_words.pop()
                    print(f"➖ Đã xóa từ: '{popped}'")
            elif key in (ord('c'), ord('C')):
                composed_words.clear()
                print("🧹 Đã xóa sạch bảng văn bản!")
            elif key in (ord('s'), ord('S')):
                out_txt = os.path.join(BASE_DIR, "composed_sign_text.txt")
                with open(out_txt, 'w', encoding='utf-8') as f:
                    f.write(" ".join(composed_words))
                print(f"💾 Đã lưu văn bản vào: {out_txt}")
            elif key in (ord('t'), ord('T')):
                txt = " ".join(composed_words)
                if txt:
                    print(f"🔊 Đang phát âm thanh văn bản: '{txt}'")
                    self.tts.speak(txt)

            time.sleep(0.03)

        self.cap.release()
        cv2.destroyAllWindows()

    def run_ai_tutor_pro(self):
        """Chức năng [4]: Gia Sư AI Pro Chấm Điểm 3D Màn Hình Đôi & Nạp Video Mẫu (AI Tutor Pro)"""
        print("\n=== [4] GIA SƯ AI PRO CHẤM ĐIỂM 3D MÀN HÌNH ĐÔI & NẠP VIDEO MẪU ===", flush=True)
        print(" [A] 🎓 Luyện tập thực hành từ vựng với Màn hình đôi (Video Mẫu + Camera AI)")
        print(" [B] 📁 Nạp Video Ngắn (1 Từ vựng đơn lẻ 1-3s) làm Thước đo chuẩn")
        print(" [C] 🎞️ Nạp Video Dài Đa Từ Vựng (Tự động quét, Cắt đoạn & Tạo Dataset mới)")
        print(" [D] 🗑️ Quản lý / Xóa / Đổi Tên Video Mẫu Đã Nạp Nhầm Trong Dataset")
        choice = input("👉 Nhập lựa chọn của bạn (A/B/C/D): ").strip().upper()

        if choice == "B":
            video_input = input("Nhập đường dẫn file Video Ngắn (.mp4): ").strip().strip('\'"')
            word_name = input("Nhập tên từ vựng đại diện cho video này: ").strip().lower()
            if not video_input or not os.path.exists(video_input):
                print("⚠️ File video không tồn tại.")
                return
            if not word_name:
                print("⚠️ Tên từ vựng không được để trống.")
                return

            confirm = input(f"❓ Xác nhận nạp video vào Dataset với nhãn '{word_name.upper()}'? (Y/N): ").strip().upper()
            if confirm in ('Y', 'YES', ''):
                if self.ai_tutor_engine:
                    self.ai_tutor_engine.enroll_reference_video(video_input, word_name)
            else:
                print("❌ Đã hủy thao tác nạp video.")
        elif choice == "C":
            video_input = input("Nhập đường dẫn file Video Dài (.mp4): ").strip().strip('\'"')
            if self.ai_tutor_engine and os.path.exists(video_input):
                self.ai_tutor_engine.auto_segment_and_enroll_long_video(video_input)
            else:
                print("⚠️ File video không tồn tại hoặc module chưa sẵn sàng.")
        elif choice == "D":
            if not self.ai_tutor_engine:
                print("⚠️ AITutorEngine chưa sẵn sàng.")
                return
            items = self.ai_tutor_engine.list_enrolled_words()
            print("\n📁 --- DANH SÁCH TỪ VỰNG TRONG KHO CUSTOM ENROLLMENT ---", flush=True)
            if not items:
                print("   (Chưa có từ vựng nào được nạp thêm)")
                return
            for idx, (w_name, count, p) in enumerate(items, 1):
                print(f"   [{idx}] Từ vựng: '{w_name.upper()}' ({count} video mẫu) -> {p}")

            print("\n [1] 🗑️ Xóa bỏ 1 từ vựng (Xóa file nhầm)")
            print(" [2] ✏️ Đổi tên nhãn từ vựng (Sửa tên nhầm)")
            print(" [0] 🔙 Quay lại")
            sub_c = input("👉 Lựa chọn thao tác (0-2): ").strip()
            if sub_c == "1":
                del_word = input("Nhập tên từ vựng muốn xóa: ").strip().lower()
                self.ai_tutor_engine.delete_enrolled_word(del_word)
            elif sub_c == "2":
                old_w = input("Nhập tên từ vựng hiện tại: ").strip().lower()
                new_w = input("Nhập tên mới muốn đổi: ").strip().lower()
                self.ai_tutor_engine.rename_enrolled_word(old_w, new_w)
        else:
            word_input = input("Nhập tên từ vựng bạn muốn tập thực hành (Mặc định 'apple', 'hello', 'book'...): ").strip().lower()
            target_word = word_input if word_input else "apple"
            if self.ai_tutor_engine:
                self.ai_tutor_engine.run_tutor_split_screen(target_word, self.predictor, self.cap)
            else:
                print("⚠️ AITutorEngine chưa sẵn sàng.")

    def run_video_file_translation(self):
        """Chức năng [5]: Dịch thuật từ Tệp Video MP4 / AVI"""
        print("\n=== [5] DỊCH THUẬT TỪ TỆP VIDEO (VIDEO FILE TRANSLATION ENGINE) ===", flush=True)
        video_path = input("Nhập đường dẫn file video (hoặc ấn Enter để dùng video mẫu temp_edge.mp4): ").strip().strip('\'"')
        if not video_path:
            video_path = os.path.join(BASE_DIR, "temp_edge.mp4")

        print("🌐 Chọn ngôn ngữ đích:")
        print(" [1] Tiếng Việt (vi)")
        print(" [2] Tiếng Anh (en)")
        print(" [3] Tiếng Nhật (ja)")
        print(" [4] Tiếng Hàn (ko)")
        lang_choice = input("👉 Nhập (1-4) [Mặc định: 1]: ").strip()
        
        lang_map = {'1': 'vi', '2': 'en', '3': 'ja', '4': 'ko'}
        target_lang = lang_map.get(lang_choice, 'vi')

        if self.video_translator:
            print("🚀 Đang khởi động tiến trình dịch thuật trực tiếp trên Video...")
            self.video_translator.translate_video(video_path, target_lang=target_lang, show_preview=True)
        else:
            print("⚠️ VideoTranslator module chưa sẵn sàng.")

    def run(self):
        """Menu Trung Tâm Điều Khiển 5 Chức Năng Thương Mại"""
        while True:
            print("\n======================================================================", flush=True)
            print("   🤟 HỆ THỐNG GIAO TIẾP & DỊCH THUẬT NGÔN NGỮ KÝ HIỆU AI (COMMERCIAL PRO)", flush=True)
            print("======================================================================", flush=True)
            print(" [1] 🗣️ Giao Tiếp 2 Chiều Thông Minh (Voice <-> Sign Two-Way Communication)", flush=True)
            print(" [2] 🎥 Cuộc Gọi Video Trực Tuyến Cho Zoom / Meet / Teams (Virtual Cam HD)", flush=True)
            print(" [3] ✍️ Bảng Soạn Thảo Văn Bản & Dịch Thuật Ký Hiệu (Sign Text Editor)", flush=True)
            print(" [4] 🎓 Gia Sư AI Pro Chấm Điểm 3D Màn Hình Đôi & Nạp Video Mẫu (AI Tutor)", flush=True)
            print(" [5] 📁 Dịch Thuật Từ Tệp Video MP4 / AVI (Video File Translation)", flush=True)
            print(" [0] 🚪 Thoát ứng dụng", flush=True)
            print("======================================================================", flush=True)

            choice = input("👉 Nhập lựa chọn của bạn (0 - 5): ").strip()
            if choice == "1":
                self.run_two_way_communication()
            elif choice == "2":
                self.run_virtual_cam_meeting()
            elif choice == "3":
                self.run_sign_text_editor()
            elif choice == "4":
                self.run_ai_tutor_pro()
            elif choice == "5":
                self.run_video_file_translation()
            elif choice == "0":
                print("👋 Cảm ơn bạn đã sử dụng Hệ thống Ngôn ngữ Ký hiệu AI!", flush=True)
                break
            else:
                print("⚠️ Lựa chọn không hợp lệ, vui lòng chọn từ 0 đến 5.", flush=True)

if __name__ == "__main__":
    app = SignLanguageApp()
    app.run()
