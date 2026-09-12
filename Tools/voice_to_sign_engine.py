# Tools/voice_to_sign_engine.py
"""
==============================================================================
MODULE GIAO TIẾP 2 CHIỀU THÔNG MINH (TWO-WAY VOICE <-> SIGN COMMUNICATION ENGINE)
==============================================================================
Nhiệm vụ:
1. Chiều 1: Người khiếm thính làm ký hiệu -> AI nhận diện -> Dịch thành chữ & Loa phát âm.
2. Chiều 2: Người bình thường nói vào Mic/nhập câu -> AI tra cứu kho 2.000 video thủ ngữ
   chuẩn và phát clip múa lại trực tiếp trên màn hình cho người khiếm thính xem.
==============================================================================
"""

import os
import sys
import time
import cv2
import numpy as np
import threading
import json
import config

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

def draw_text_wrapped(img, text, position, font, scale, color, thickness, max_width):
    words = text.split(' ')
    lines = []
    current_line = words[0]
    for word in words[1:]:
        (w, h), _ = cv2.getTextSize(current_line + " " + word, font, scale, thickness)
        if w < max_width:
            current_line += " " + word
        else:
            lines.append(current_line)
            current_line = word
    lines.append(current_line)
    
    x, y = position
    for line in lines:
        cv2.putText(img, line, (x, y), font, scale, color, thickness)
        y += 25 # line height offset

class VoiceToSignEngine:
    def __init__(self, tts=None):
        self.tts = tts
        self.sl_dataset_dir = config.WLASL_VIDEOS_DIR
        self.vocab_map = self._build_vocab_map()
        self.current_sign_video = None
        self.is_playing_sign = False

    def _build_vocab_map(self):
        """Khởi tạo danh mục ánh xạ từ vựng Tiếng Việt / Tiếng Anh sang thư mục video WLASL"""
        mapping = {}
        try:
            with open(config.WLASL_CLASS_LIST_PATH, 'r', encoding='utf-8') as stream:
                class_names = {
                    int(class_id): gloss.strip()
                    for class_id, gloss in (line.rstrip().split('\t', 1) for line in stream if line.strip())
                }
            with open(config.WLASL_METADATA_PATH, 'r', encoding='utf-8') as stream:
                metadata = json.load(stream)
            for video_id, item in metadata.items():
                gloss = class_names.get(int(item['action'][0]))
                video_path = os.path.join(self.sl_dataset_dir, f"{video_id}.mp4")
                if gloss and gloss.lower() not in mapping and os.path.isfile(video_path):
                    mapping[gloss.lower()] = video_path
        except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
            print(f"⚠️ Không thể lập danh mục video WLASL: {exc}")

        # Bổ sung từ điển Tiếng Việt tương ứng
        vi_to_en = {
            'xin chào': 'hello', 'chào': 'hello', 'chào bạn': 'hello',
            'cảm ơn': 'thank', 'cảm ơn bạn': 'thank', 'tạm biệt': 'goodbye',
            'quả táo': 'apple', 'táo': 'apple', 'cuốn sách': 'book', 'sách': 'book',
            'máy tính': 'computer', 'uống': 'drink', 'nước': 'water',
            'cái giường': 'bed', 'giường': 'bed', 'trợ giúp': 'help', 'giúp': 'help',
            'thích': 'like', 'không': 'no', 'có': 'yes', 'vâng': 'yes',
            'bố': 'father', 'mẹ': 'mother', 'gia đình': 'family',
            'tai nạn': 'accident', 'bệnh viện': 'hospital', 'bác sĩ': 'doctor',
            'áo': 'shirt', 'áo sơ mi': 'shirt', 'ngầu': 'cool', 'lạnh': 'cold'
        }
        for vi_word, en_word in vi_to_en.items():
            if en_word in mapping:
                mapping[vi_word.lower()] = mapping[en_word]

        return mapping

    def find_sign_video(self, query_text):
        """Tìm đường dẫn tệp video ký hiệu tương ứng với từ/câu được nói"""
        videos = self.find_sign_videos(query_text)
        return videos[0][1] if videos else None

    def find_sign_videos(self, query_text):
        """Resolve a phrase to an ordered list of available isolated signs."""
        query = query_text.strip().lower()
        if query in self.vocab_map:
            return [(query, self.vocab_map[query])]

        return [(word, self.vocab_map[word]) for word in query.split() if word in self.vocab_map]

    def listen_speech_or_text(self):
        """Lắng nghe giọng nói từ Microphone (hoặc fallback sang nhập text nhanh)"""
        print("\n🎤 [Microphone] Đang lắng nghe giọng nói người bình thường...", flush=True)
        try:
            import speech_recognition as sr
            r = sr.Recognizer()
            with sr.Microphone() as source:
                r.adjust_for_ambient_noise(source, duration=0.8)
                print("👉 Xin mời nói vào Micro (Ví dụ: 'Xin chào', 'Apple', 'Help', 'Book')...", flush=True)
                audio = r.listen(source, timeout=5, phrase_time_limit=4)
                try:
                    text = r.recognize_google(audio, language="vi-VN")
                    print(f"✅ Nhận diện giọng nói (Tiếng Việt): '{text}'", flush=True)
                    return text
                except Exception:
                    text = r.recognize_google(audio, language="en-US")
                    print(f"✅ Nhận diện giọng nói (Tiếng Anh): '{text}'", flush=True)
                    return text
        except Exception:
            # Fallback nếu máy tính không có microphone hoặc chưa cài PyAudio
            user_msg = input("💬 Nhập câu nói của người bình thường (Ví dụ: 'hello', 'apple', 'cảm ơn'): ").strip()
            return user_msg

    def run_two_way_interface(self, predictor, context_agent=None, tts=None, cap=None):
        """
        Bật giao diện Giao Tiếp 2 Chiều Toàn Diện (Two-Way Communication Session):
        - Chiều 1: Người khiếm thính làm ký hiệu -> Dịch ra chữ & Loa đọc
        - Chiều 2: Bấm phím [T] hoặc [V] -> Người bình thường nói -> AI phát video ký hiệu múa lại
        """
        print("\n======================================================================", flush=True)
        print(" 🗣️ PHIÊN GIAO TIẾP 2 CHIỀU THÔNG MINH (TWO-WAY COMMUNICATION SESSION)", flush=True)
        print("======================================================================", flush=True)
        print("👉 Hướng dẫn: [T]/[V] Người bình thường nói | [C] Xóa câu | [Q] Thoát", flush=True)

        owns_cap = cap is None
        if owns_cap:
            cap = cv2.VideoCapture(0)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        sign_cap = None
        sign_queue = []
        sign_word_label = ""
        current_sentence = ""
        last_action = None
        spelled_chars = []
        action_words = []
        last_hand_time = time.time()

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            h, w = frame.shape[:2]

            # 1. Quét cử chỉ người khiếm thính (Chiều 1)
            static_char, action, conf, bboxes = predictor.process_frame(frame)
            active_label = action if action else static_char

            if bboxes and len(bboxes) > 0:
                last_hand_time = time.time()
                for (x1, y1, x2, y2, hand_label) in bboxes:
                    color = (0, 255, 0) if "Left" in hand_label else (255, 0, 0)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    lbl = hand_label + (f": [{active_label}]" if active_label else "")
                    cv2.putText(frame, lbl, (x1, max(15, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

            if active_label and active_label != last_action:
                if len(active_label) == 1 and active_label.isalpha():
                    spelled_chars.append(active_label)
                else:
                    action_words.append(active_label)
                last_action = active_label
                last_hand_time = time.time()

            # Dừng tay > 1.5s -> Chốt câu, Dịch và đọc phát âm
            idle_time = time.time() - last_hand_time
            if idle_time > 1.5:
                if action_words or spelled_chars:
                    if action_words:
                        raw = " ".join(action_words)
                        action_words.clear()
                    else:
                        raw = "".join(spelled_chars)
                        spelled_chars.clear()
                        
                    if context_agent:
                        current_sentence = context_agent.process(raw, target_lang='vi')
                    else:
                        current_sentence = raw
                        
                    if tts and current_sentence:
                        tts.speak(current_sentence)
                        
                # Nếu đã nhàn rỗi quá 5 giây -> Tự động xóa sạch bảng phụ đề cũ
                elif idle_time > 5.0 and current_sentence:
                    current_sentence = ""

            # 2. Xử lý video múa lại cho người khiếm thính (Chiều 2)
            if sign_cap and sign_cap.isOpened():
                ret_sign, frame_sign = sign_cap.read()
                if not ret_sign:
                    sign_cap.release()
                    if sign_queue:
                        sign_word_label, next_path = sign_queue.pop(0)
                        sign_cap = cv2.VideoCapture(next_path)
                        ret_sign, frame_sign = sign_cap.read()
                    else:
                        sign_cap = None

                if ret_sign:
                    sign_resized = cv2.resize(frame_sign, (280, 210))
                    # Ghép khung video ký hiệu vào góc trên bên phải
                    frame[10:220, w - 290:w - 10] = sign_resized
                    cv2.rectangle(frame, (w - 290, 10), (w - 10, 220), (0, 255, 255), 2)
                    cv2.putText(frame, f"KHIEM THINH XEM: '{sign_word_label.upper()}'", (w - 285, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)

            # 3. Vẽ bảng phụ đề 2 chiều ở đáy màn hình
            cv2.rectangle(frame, (0, h - 90), (w, h), (20, 20, 20), -1)
            
            # Khung Live Preview (In chữ ra liền)
            live_draft = ""
            if action_words:
                live_draft = " ".join(action_words)
            elif spelled_chars:
                live_draft = "".join(spelled_chars)
            
            if live_draft:
                draw_text_wrapped(frame, f"[Tuong tac...]: {live_draft}", (15, h - 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1, w - 30)

            # Câu hoàn chỉnh (Đã qua AI)
            final_text = f"Nguoi Khiem Thinh: {current_sentence}" if current_sentence else "Nguoi Khiem Thinh: (Dang cho...)"
            draw_text_wrapped(frame, final_text, (15, h - 45), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2, w - 30)
            
            cv2.putText(frame, "Phim: [T]/[V] Nguoi thuong noi | [C] Xoa cau | [Q] Thoat",
                        (15, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

            cv2.imshow("Two-Way Sign Language Communication Platform", frame)
            key = cv2.waitKey(1) & 0xFF

            if key in (ord('q'), ord('Q')):
                break
            elif key in (ord('t'), ord('T'), ord('v'), ord('V')):
                spoken_text = self.listen_speech_or_text()
                if spoken_text:
                    found_videos = self.find_sign_videos(spoken_text)
                    if found_videos:
                        if sign_cap:
                            sign_cap.release()
                        (sign_word_label, found_vid), *sign_queue = found_videos
                        sign_cap = cv2.VideoCapture(found_vid)
                        print(f"🎬 [Giao tiếp 2 chiều] Phát {len(found_videos)} video cử chỉ cho: '{spoken_text}'")
                    else:
                        print(f"⚠️ Chưa có video mẫu sẵn cho từ: '{spoken_text}'. Bạn có thể nạp ở mục [4].")
            elif key in (ord('c'), ord('C')):
                current_sentence = ""
                sign_cap = None
                print("🧹 Đã làm sạch bảng dịch 2 chiều.")

        if sign_cap:
            sign_cap.release()
        if owns_cap and cap:
            cap.release()
        cv2.destroyAllWindows()
