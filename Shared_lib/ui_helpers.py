# Shared_lib/ui_helpers.py
"""
Thư viện Giao diện & Trợ lý Trung tâm (Central UI Helpers Library)
Chứa các module dùng chung cho Mobile_app, Demo_ui, Cloud_server và Tools:
1. TextToSpeech (Loa phát âm ngầm Async không đơ camera)
2. VirtualCamera (Camera ảo Google Meet / Zoom safe fallback)
3. SubtitleRenderer (Vẽ phụ đề Tiếng Việt nét căng bo góc chuẩn Netflix)
4. draw_hand_badge (Vẽ thẻ Bounding Box bàn tay bo góc rực rỡ)
"""

import cv2
import numpy as np
import threading
import os
from PIL import Image, ImageDraw, ImageFont


# ==============================================================================
# 🔊 1. MODULE LOA ĐỌC PHÁT ÂM TTS (Async Background Threading)
# ==============================================================================
class TextToSpeech:
    """Module phát âm thanh đọc câu dịch Tiếng Việt/Anh chạy ngầm không làm đơ giao diện"""
    def __init__(self, rate=150, volume=1.0):
        self.rate = rate
        self.volume = volume
        self._engine = None
        try:
            import pyttsx3
            self._pyttsx3 = pyttsx3
        except ImportError:
            self._pyttsx3 = None

    def _speak_thread(self, text):
        if not self._pyttsx3:
            return
        try:
            engine = self._pyttsx3.init()
            engine.setProperty('rate', self.rate)
            engine.setProperty('volume', self.volume)
            engine.say(text)
            engine.runAndWait()
        except Exception as e:
            print(f"⚠️ Lỗi phát âm thanh TTS: {e}", flush=True)

    def speak(self, text):
        if text and text.strip():
            threading.Thread(target=self._speak_thread, args=(text,), daemon=True).start()


# ==============================================================================
# 🎥 2. MODULE WEBCAM ẢO GOOGLE MEET / ZOOM (Safe Fallback)
# ==============================================================================
class VirtualCamera:
    """Module truyền luồng camera đã vẽ phụ đề sang Google Meet, Zoom, MS Teams"""
    def __init__(self, width=640, height=480, fps=20):
        self.width = width
        self.height = height
        self.fps = fps
        self.use_virtual = False
        try:
            import pyvirtualcam
            self.cam = pyvirtualcam.Camera(width=width, height=height, fps=fps)
            self.use_virtual = True
            print("🚀 Đã khởi tạo Webcam Ảo (Virtual Camera) thành công!", flush=True)
        except Exception:
            self.use_virtual = False

    def send_frame(self, frame):
        if self.use_virtual and self.cam:
            try:
                frame_resized = cv2.resize(frame, (self.width, self.height))
                rgb_frame = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
                self.cam.send(rgb_frame)
            except Exception:
                pass

    def close(self):
        if self.use_virtual and self.cam:
            try:
                self.cam.close()
            except Exception:
                pass


# ==============================================================================
# 🎬 3. BỘ VẼ PHỤ ĐỀ CHUẨN NETFLIX / YOUTUBE (Hỗ trợ 100% Tiếng Việt có dấu)
# ==============================================================================
class SubtitleRenderer:
    """
    Bộ vẽ phụ đề chuẩn Netflix / YouTube Style:
    - Hỗ trợ Tiếng Việt UTF-8 có dấu nét căng (dùng PIL).
    - Khung nền đen mờ bo tròn góc (Translucent Rounded Box).
    - Đổ bóng viền chữ (Text Outline/Shadow) tương phản nổi bật trên mọi background.
    - Căn giữa đáy màn hình tự động ngắt dòng.
    """
    def __init__(self, font_size=24):
        self.font_size = font_size
        self.font = self._load_vietnamese_font(font_size)

    def _load_vietnamese_font(self, size):
        font_paths = [
            "C:/Windows/Fonts/arial.ttf",
            "C:/Windows/Fonts/calibri.ttf",
            "C:/Windows/Fonts/segoeui.ttf",
            "C:/Windows/Fonts/tahoma.ttf"
        ]
        for path in font_paths:
            if os.path.exists(path):
                try:
                    return ImageFont.truetype(path, size)
                except Exception:
                    pass
        return ImageFont.load_default()

    def draw_subtitle(self, cv2_img, text, lang_name="TIẾNG VIỆT"):
        if not text or not text.strip():
            return cv2_img

        h, w = cv2_img.shape[:2]
        full_text = f"Dịch ({lang_name}): {text}"

        pil_img = Image.fromarray(cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_img, "RGBA")

        bbox = self.font.getbbox(full_text)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]

        padding_x = 20
        padding_y = 10
        box_w = min(w - 40, text_w + padding_x * 2)
        box_h = text_h + padding_y * 2

        box_x1 = (w - box_w) // 2
        box_y1 = h - box_h - 25
        box_x2 = box_x1 + box_w
        box_y2 = box_y1 + box_h

        # 1. Nền đen mờ bo góc kiểu Netflix
        overlay = Image.new("RGBA", pil_img.size, (0, 0, 0, 0))
        overlay_draw = ImageDraw.Draw(overlay)
        overlay_draw.rounded_rectangle(
            [box_x1, box_y1, box_x2, box_y2],
            radius=12,
            fill=(15, 15, 20, 195),
            outline=(255, 215, 0, 220), # Viền vàng kim
            width=2
        )
        pil_img = Image.alpha_composite(pil_img.convert("RGBA"), overlay)
        draw = ImageDraw.Draw(pil_img)

        text_x = box_x1 + (box_w - text_w) // 2
        text_y = box_y1 + padding_y - 2

        # 2. Đổ bóng viền chữ đen (Text Shadow)
        shadow_color = (0, 0, 0, 255)
        for offset_x, offset_y in [(-1, -1), (1, -1), (-1, 1), (1, 1), (0, 2)]:
            draw.text((text_x + offset_x, text_y + offset_y), full_text, font=self.font, fill=shadow_color)

        # 3. Chữ màu trắng nổi bật
        draw.text((text_x, text_y), full_text, font=self.font, fill=(255, 255, 255, 255))

        return cv2.cvtColor(np.array(pil_img.convert("RGB")), cv2.COLOR_RGB2BGR)


# ==============================================================================
# 🏷️ 4. BỘ VẼ BÀN TAY & CHỮ CÁI TĨNH (Hand Badge Overlay)
# ==============================================================================
def draw_hand_badge(frame, bboxes, active_label=None):
    """Vẽ Bounding box tay trái (Xanh lá) và tay phải (Cam) kèm thẻ chữ cái bo góc"""
    if not bboxes:
        return frame

    for (x1, y1, x2, y2, hand_label) in bboxes:
        is_left = "Left" in hand_label
        color = (0, 230, 115) if is_left else (255, 128, 0)
        
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        display_text = hand_label
        if active_label:
            display_text += f": [{active_label}]"

        text_size = cv2.getTextSize(display_text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)[0]
        badge_w = text_size[0] + 12
        badge_h = 24
        badge_y1 = max(0, y1 - badge_h)

        cv2.rectangle(frame, (x1, badge_y1), (x1 + badge_w, y1), color, -1)
        cv2.putText(frame, display_text, (x1 + 6, max(16, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

    return frame
