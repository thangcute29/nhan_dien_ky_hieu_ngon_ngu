# mobile_app/src/main.py
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
except ImportError:
    VideoTranslator = None
    AITutorEngine = None

CONFIDENCE_THRESHOLD = 0.6
SERVER_URL = "http://localhost:8000"

class SignLanguageApp:
    def __init__(self):
        self.predictor = SignLanguagePredictor(
            yolo_path=os.path.join(BASE_DIR, "mobile_app", "assets", "hand_det_yolo.tflite"),
            feature_path=os.path.join(BASE_DIR, "mobile_app", "assets", "feature_extractor.tflite"),
            gru_path=os.path.join(BASE_DIR, "mobile_app", "assets", "action_recognizer.tflite")
        )
        self.tts = TextToSpeech()
        self.virtual_cam = VirtualCamera()
        self.cap = cv2.VideoCapture(0)

#=======================khởi tạo LLMCorrector và ContextAgent========================
        self.llm_corrector = LLMCorrector() if LLMCorrector else None
        self.context_agent = ContextAgent(llm_corrector=self.llm_corrector) if ContextAgent else None
        self.translator = LLMTranslator(llm_corrector=self.llm_corrector) if LLMTranslator else None

#=======================Khởi tạo Module Dịch Video & Gia sư AI Người Ảo ===================
        self.video_translator = VideoTranslator(predictor=self.predictor, context_agent=self.context_agent, tts=self.tts) if VideoTranslator else None
        self.ai_tutor_engine = AITutorEngine(tts=self.tts) if AITutorEngine else None
    
    
        self.edge_case_buffer = []
        self.last_action = None
        self.sentence_words = []
        self.last_hand_time = time.time()
        self.PAUSE_TIMEOUT = 2.0
        self.final_sentence = ""

#=======================Giới hạn tốc độ xử lý AI =========================
        self.last_process_time = 0
        self.process_interval = 0.08  # NÂNG CẤP 1: Giới hạn xử lý AI ở 12.5 FPS (80ms), camera hiển thị 30 FPS mượt mà
        self._cached_static_char = None
        self._cached_action = None
        self._cached_conf = 0.0
        self._cached_bboxes = []

    def send_edge_case(self, video_frames, predicted_label):
        """
        NÂNG CẤP 2: Gửi video/ảnh bị nhận diện sai lên server bằng In-Memory Byte Stream (RAM).
        Loại bỏ 100% việc tạo file rác temp_edge.mp4 trên đĩa cứng SSD!
        """
        if not video_frames:
            return

        import io
        try:
            # Mã hóa khung hình trực tiếp trên bộ nhớ RAM thành mảng Byte JPEG
            success, buffer = cv2.imencode('.jpg', video_frames[0])
            if not success:
                return

            ram_stream = io.BytesIO(buffer.tobytes())
            files = {'image': ('edge_case.jpg', ram_stream, 'image/jpeg')}
            data = {'label': predicted_label}

            requests.post(f"{SERVER_URL}/upload_edge_case", files=files, data=data, timeout=3)
            print("✅ [In-Memory Stream] Đã gửi Edge Case trực tiếp qua RAM (không ghi file SSD).", flush=True)
        except Exception as e:
            print(f"⚠️ Lỗi khi gửi edge case qua RAM: {e}", flush=True)

    def _async_translate_and_speak(self, raw_sentence, lang):
        """SỰ CỐ 3 FIX: Xử lý dịch thuật LLM và phát âm thanh TTS ở luồng ngầm (Background Thread)"""
        try:
            if self.context_agent:
                translated = self.context_agent.process(action_word=raw_sentence, target_lang=lang)
            else:
                translated = raw_sentence
            
            self.final_sentence = translated
            if translated:
                print(f"🔊 [Async] Đang đọc phát âm câu dịch: '{translated}'", flush=True)
                self.tts.speak(translated)
        except Exception as e:
            print(f"⚠️ Lỗi luồng ngầm dịch thuật: {e}", flush=True)

    def run_live_webcam(self):
        """Option 1: Dịch thuật thời gian thực qua Webcam"""
        print("\n=== [1] DỊCH THUẬT TRỰC TIẾP QUA WEBCAM ===", flush=True)
        print("\n--- [BẢNG CHỌN NGÔN NGỮ DỊCH THUẬT BAN ĐẦU] ---", flush=True)
        print(" [V] Tiếng Việt (Vietnamese - Mặc định)")
        print(" [E] Tiếng Anh (English)")
        print(" [J] Tiếng Nhật (Japanese)")
        print(" [K] Tiếng Hàn (Korean)")
        l_choice = input("👉 Nhập Lựa Chọn Ngôn Ngữ Mong Muốn (V/E/J/K, ấn Enter để chọn Việt): ").strip().lower()
        lang_map = {'v': 'vi', 'e': 'en', 'j': 'ja', 'k': 'ko'}
        self.current_lang = lang_map.get(l_choice, 'vi')
        
        print(f"\n🚀 Đã khởi tạo ngôn ngữ: {self.current_lang.upper()}", flush=True)
        print("Phím tắt đổi ngôn ngữ lúc chạy: [Q] Thoát | [V] Việt | [E] Anh | [J] Nhật | [K] Hàn\n", flush=True)

        self.cap = cv2.VideoCapture(0)
        # NÂNG CẤP 1: Thiết lập độ phân giải Webcam chuẩn 640x480 tối ưu cho YOLO & EfficientNet
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        self.sentence_words = []
        self.final_sentence = ""
        self.last_action = None

        while True:
            ret, frame = self.cap.read()
            if not ret:
                break

            # NÂNG CẤP 1: Giới hạn tần suất chạy AI 12.5 FPS (0.08s) có cache hiển thị, camera vẫn mượt 30 FPS
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


            # Hiển thị nhãn ngôn ngữ đang chọn
            lang = getattr(self, 'current_lang', 'vi')
            lang_names = {'vi': 'TIENG VIET (Phim V)', 'en': 'ENGLISH (Phim E)', 'ja': 'JAPANESE (Phim J)', 'ko': 'KOREAN (Phim K)'}
            lang_str = lang_names.get(lang, 'TIENG VIET (Phim V)')
            cv2.rectangle(display, (10, 10), (320, 45), (0, 0, 0), -1)
            cv2.putText(display, f"NGON NGU: {lang_str}", (15, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)

            # 1. Vẽ bounding box tay trái (Xanh lá) và tay phải (Xanh dương) kèm Chữ cái/Từ vựng
            active_label = static_char or action
            if bboxes and len(bboxes) > 0:
                self.last_hand_time = time.time()
                for (x1, y1, x2, y2, hand_label) in bboxes:
                    color = (0, 255, 0) if "Left" in hand_label else (255, 0, 0)
                    cv2.rectangle(display, (x1, y1), (x2, y2), color, 2)
                    
                    # Thêm nhãn nhận diện chữ cái/từ vựng ngay trên Bounding Box
                    display_text = hand_label
                    if active_label:
                        display_text += f": [{active_label}]"
                    
                    # Background mờ cho chữ hiển thị rõ nét
                    cv2.rectangle(display, (x1, max(0, y1 - 25)), (x1 + len(display_text) * 11, y1), color, -1)
                    cv2.putText(display, display_text, (x1 + 3, max(15, y1 - 7)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

            # 2. Gom chữ cái / từ ngữ khi AI nhận diện được nhãn mới
            if active_label and active_label != self.last_action:
                self.sentence_words.append(active_label)
                self.last_action = active_label
                self.last_hand_time = time.time()
                print(f"✨ AI Nhận diện được: [{active_label}] (Conf: {conf:.2f})", flush=True)

            # 3. SỰ CỐ 3 FIX: Khi dừng tay > 2.0s -> Đưa tác vụ Dịch & Đọc Loa sang LUỒNG NGẦM (Threading)
            if len(self.sentence_words) > 0 and (time.time() - self.last_hand_time > self.PAUSE_TIMEOUT):
                raw_sentence = " ".join(self.sentence_words)
                self.sentence_words = []
                # Chạy ngầm không làm đứng camera
                threading.Thread(target=self._async_translate_and_speak, args=(raw_sentence, lang), daemon=True).start()


            # 4. Thanh phụ đề màu đen mờ ở đáy màn hình
            current_composed = " ".join(self.sentence_words)
            if self.final_sentence or current_composed:
                text_to_show = self.final_sentence if self.final_sentence else f"Dang nhap: {current_composed}"
                cv2.rectangle(display, (0, h - 60), (w, h), (0, 0, 0), -1)
                cv2.putText(display, f"Dich ({lang.upper()}): {text_to_show}", (20, h - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 2)

            self.virtual_cam.send_frame(display)
            cv2.imshow("Sign Language App - Live Stream", display)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), ord('Q')):
                break
            elif key in (ord('v'), ord('V')): self.current_lang = 'vi'
            elif key in (ord('e'), ord('E')): self.current_lang = 'en'
            elif key in (ord('j'), ord('J')): self.current_lang = 'ja'
            elif key in (ord('k'), ord('K')): self.current_lang = 'ko'

            time.sleep(0.03)

        self.cap.release()
        cv2.destroyAllWindows()


    def run_video_file_translation(self):
        """Option 2: Dịch thuật từ Tệp Video MP4 / AVI (Sử dụng Module VideoTranslator)"""
        print("\n=== [2] DỊCH THUẬT TỪ TỆP VIDEO (VIDEO FILE TRANSLATION ENGINE) ===", flush=True)
        video_path = input("Nhập đường dẫn file video (hoặc ấn Enter để dùng video mẫu temp_edge.mp4): ").strip()
        if not video_path:
            video_path = os.path.join(BASE_DIR, "temp_edge.mp4")

        target_lang = getattr(self, 'current_lang', 'vi')
        if self.video_translator:
            result = self.video_translator.translate_video(video_path, target_lang=target_lang, show_preview=True)
        else:
            print("⚠️ VideoTranslator module chưa sẵn sàng. Chạy quy trình fallback...")

    def run_sign_text_editor(self):
        """Option 3: Bảng Soạn thảo Văn bản bằng Ngôn ngữ Ký hiệu (Sign Text Editor)"""
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

            # Bảng soạn thảo văn bản ở đỉnh màn hình
            cv2.rectangle(display, (0, 0), (w, 80), (30, 30, 30), -1)
            full_text = " ".join(composed_words) + (f" [{current_word}]" if current_word else "")
            cv2.putText(display, f"VAN BAN SOAN THAO: {full_text}", (15, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(display, "Phim: [Space] Them | [D] Xoa | [C] Xoa het | [S] Luu | [T] Doc | [Q] Ra", (15, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

            cv2.imshow("Sign Text Editor", display)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), ord('Q')):
                break
            elif key == ord(' '):  # Thêm từ hiện tại
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

    def run_ai_tutor_and_enrollment(self):
        """Option 4: Gia sư AI Chấm điểm Thực hành & NGƯỜI ẢO AI TRỢ LÝ (AI Tutor & Avatar Engine)"""
        print("\n=== [4] GIA SƯ AI CHẤM ĐIỂM THỰC HÀNH & NẠP VIDEO MẪU (AI TUTOR) ===", flush=True)
        print(" [A] Thực hành từ sẵn trong Từ điển (100 từ vựng)")
        print(" [B] Nạp Video Mẫu mới (.mp4) để AI Học & Bổ sung Dataset")
        choice = input("Nhập lựa chọn của bạn (A/B): ").strip().upper()

        target_word = "apple"

        if choice == "B":
            video_input = input("Nhập đường dẫn file Video Mẫu (.mp4) mới: ").strip()
            word_name = input("Nhập tên từ vựng đại diện cho video này: ").strip().lower()
            if self.ai_tutor_engine and os.path.exists(video_input) and word_name:
                target_word = word_name
                self.ai_tutor_engine.enroll_reference_video(video_input, word_name)
            else:
                print("⚠️ File không tồn tại hoặc module chưa sẵn sàng. Chuyển sang từ mẫu mặc định 'apple'.")
        else:
            word_input = input("Nhập tên từ vựng bạn muốn tập thực hành (Mặc định 'apple'): ").strip().lower()
            if word_input:
                target_word = word_input

        print(f"\n🎓 BẮT ĐẦU BUỔI THỰC HÀNH VỚI NGƯỜI ẢO AI CHO TỪ: '{target_word.upper()}'", flush=True)
        print("Vui lòng giơ 2 bàn tay lên camera thực hiện cử chỉ. [Q] để thoát.", flush=True)

        self.cap = cv2.VideoCapture(0)

        while True:
            ret, frame = self.cap.read()
            if not ret:
                break

            static_char, action, conf, bboxes = self.predictor.process_frame(frame)
            display = frame.copy()

            if bboxes:
                for (x1, y1, x2, y2, hand_label) in bboxes:
                    color = (0, 255, 0) if "Left" in hand_label else (255, 0, 0)
                    cv2.rectangle(display, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(display, hand_label, (x1, max(15, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            # Đánh giá cử chỉ và Vẽ NGƯỜI ẢO AI TRỢ LÝ (AI Virtual Avatar)
            if self.ai_tutor_engine:
                score, status, msg, color = self.ai_tutor_engine.evaluate_student_gesture(display, bboxes, action, conf, target_word)
                self.ai_tutor_engine.draw_virtual_avatar(display, score, msg, color, target_word)

            cv2.imshow("AI Sign Language Tutor - Virtual Avatar Assistant", display)
            if cv2.waitKey(1) & 0xFF in (ord('q'), ord('Q')):
                break

            time.sleep(0.03)

        self.cap.release()
        cv2.destroyAllWindows()

    def run(self):
        """Menu Trung Tâm Điều Khiển 4 Chức Năng"""
        while True:
            print("\n======================================================================", flush=True)
            print("   🤟 HỆ THỐNG DỊCH THUẬT & GIA SƯ HỌC KÝ HIỆU THÔNG MINH (AI PLATFORM)", flush=True)
            print("======================================================================", flush=True)
            print(" [1] 🎥 Dịch thuật Trực tiếp qua Webcam (Live Camera Translation)", flush=True)
            print(" [2] 📁 Dịch thuật từ Tệp Video MP4/AVI (Video File Translation)", flush=True)
            print(" [3] ✍️ Bảng Soạn thảo Văn bản Ngôn ngữ Ký hiệu (Sign Text Editor)", flush=True)
            print(" [4] 🎓 Gia sư AI Chấm điểm & Nạp Video Mẫu Thực hành (AI Tutor)", flush=True)
            print(" [0] 🚪 Thoát ứng dụng", flush=True)
            print("======================================================================", flush=True)

            choice = input("Nhập lựa chọn của bạn (0 - 4): ").strip()
            if choice == "1":
                self.run_live_webcam()
            elif choice == "2":
                self.run_video_file_translation()
            elif choice == "3":
                self.run_sign_text_editor()
            elif choice == "4":
                self.run_ai_tutor_and_enrollment()
            elif choice == "0":
                print("👋 Cảm ơn bạn đã sử dụng Hệ thống Nhận diện Ngôn ngữ Ký hiệu AI!", flush=True)
                break
            else:
                print("⚠️ Lựa chọn không hợp lệ, vui lòng chọn từ 0 đến 4.", flush=True)

if __name__ == "__main__":
    app = SignLanguageApp()
    app.run()







# tham khảo thêm về độ fps và tối ưu hiệu suất:
    #Phim chiếu rạp          24fps   → mắt người thấy mượt 
    #Game bình thường        30fps   → mượt
    #Game cao cấp             60fps   → rất mượt, đặc biệt với game hành động nhanh
    #Video call (Zoom)        30fps   → mượt
    #Video call (Google Meet) 15-30fps → mượt nếu đủ băng thông
    #Để đạt được 30fps, mỗi khung hình phải được xử lý trong khoảng 33ms. Nếu xử lý mất nhiều thời gian hơn, fps sẽ giảm xuống.
    #Nhận diện ký hiệu tay  10fps   → ĐỦ DÙNG, vì ngôn ngữ ký hiệu không yêu cầu chuyển động quá nhanh như game hành động, và việc xử lý hình ảnh phức tạp có thể mất nhiều thời gian hơn. 10fps là mức tối thiểu để đảm bảo trải nghiệm người dùng không bị giật lag quá nhiều, đồng thời vẫn cho phép mô hình có đủ thời gian để phân tích và đưa ra dự đoán chính xác. Nếu có thể tối ưu để đạt 15-20fps thì sẽ càng tốt, nhưng 10fps là mức chấp nhận được cho ứng dụng này.
