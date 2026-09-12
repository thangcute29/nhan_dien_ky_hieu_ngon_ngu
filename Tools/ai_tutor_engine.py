# Tools/ai_tutor_engine.py
"""
==============================================================================
MODULE GIA SƯ AI CHẤM ĐIỂM HỌC TẬP & NGƯỜI ẢO AI TRỢ LÝ (AI TUTOR PRO ENGINE)
==============================================================================
Nhiệm vụ:
1. Màn hình đôi Split-Screen (1280x480):
   - Nửa Trái: Video Mẫu của chuyên gia múa chuẩn (từ kho 2.000 từ vựng).
   - Nửa Phải: Camera trực tiếp của học viên đang tập theo kèm AI chấm điểm 0-100%.
2. Tiếp nhận và NẠP BỔ SUNG Video Mẫu mới (.mp4) làm "Thước đo chuẩn" cho AI.
3. Chẩn đoán hình học bàn tay & Phát âm thanh hướng dẫn cụ thể theo thời gian thực.
==============================================================================
"""

import os
import sys
import time
import shutil
import json
import cv2
import numpy as np
import re

if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)
import config

SAFE_WORD = re.compile(r"[a-z0-9][a-z0-9_-]{0,63}")


def normalize_word_name(word_name):
    word = str(word_name).strip().lower().replace(" ", "_")
    return word if SAFE_WORD.fullmatch(word) else None

class AITutorEngine:
    def __init__(self, tts=None):
        self.tts = tts
        self.last_speech_time = 0
        self.speech_cooldown = 4.0  # Giới hạn phát âm thanh nhắc nhở mỗi 4 giây
        self.reference_templates = {}
        self.sl_dataset_dir = config.WLASL_VIDEOS_DIR
        self.wlasl_references = self._build_wlasl_references()

    def _build_wlasl_references(self):
        references = {}
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
                path = os.path.join(self.sl_dataset_dir, f"{video_id}.mp4")
                if gloss and gloss not in references and os.path.isfile(path):
                    references[gloss] = path
        except (OSError, ValueError, KeyError, json.JSONDecodeError):
            pass
        return references

    def get_reference_video(self, word_name):
        """Tìm video mẫu từ kho tham chiếu hoặc kho 2.000 từ vựng WLASL"""
        word_clean = normalize_word_name(word_name)
        if not word_clean:
            return None
        if word_clean in self.reference_templates:
            return self.reference_templates[word_clean]["video_path"]

        # Kiểm tra kho custom enrollment
        custom_dir = os.path.join(BASE_DIR, "Research_and_Data", "Dataset", "Sequences", "custom_enrollment", word_clean)
        if os.path.exists(custom_dir):
            files = [f for f in os.listdir(custom_dir) if f.endswith(('.mp4', '.avi'))]
            if files:
                return os.path.join(custom_dir, files[0])

        # Numeric filenames are resolved through WLASL metadata.
        if word_clean in self.wlasl_references:
            return self.wlasl_references[word_clean]

        return None

    def enroll_reference_video(self, video_path, word_name):
        """Nạp Video Mẫu mới làm Thước đo chuẩn và tự động BỔ SUNG VÀO DATASET."""
        if not video_path:
            print("❌ [AITutorEngine] Đường dẫn video rỗng!")
            return False

        video_path = video_path.strip('\'" \t\r\n')
        if (not os.path.isfile(video_path) or
                os.path.splitext(video_path)[1].lower() not in {'.mp4', '.avi', '.mov', '.mkv'}):
            print(f"❌ [AITutorEngine] Không tìm thấy tệp video mẫu: {video_path}")
            return False

        word_name = normalize_word_name(word_name)
        if not word_name:
            print("❌ Tên nhãn chỉ được chứa chữ a-z, số, dấu gạch ngang hoặc gạch dưới.")
            return False
        print(f"\n📥 [AITutorEngine] Đang nạp Video Mẫu mới cho từ: '{word_name.upper()}'...", flush=True)

        dataset_target_dir = os.path.join(BASE_DIR, "Research_and_Data", "Dataset", "Sequences", "custom_enrollment", word_name)
        os.makedirs(dataset_target_dir, exist_ok=True)
        
        dest_filename = f"{word_name}_ref_{int(time.time())}.mp4"
        dest_path = os.path.join(dataset_target_dir, dest_filename)
        shutil.copy2(video_path, dest_path)
        print(f"✅ [AITutorEngine] Đã NẠP BỔ SUNG Video Mẫu vào Dataset tại: {dest_path}")

        self.reference_templates[word_name] = {
            "video_path": dest_path,
            "enrolled_at": time.time()
        }
        return True

    def list_enrolled_words(self):
        """Liệt kê danh sách tất cả các từ vựng đã nạp trong custom_enrollment"""
        custom_base = os.path.join(BASE_DIR, "Research_and_Data", "Dataset", "Sequences", "custom_enrollment")
        if not os.path.exists(custom_base):
            return []
        items = []
        for name in os.listdir(custom_base):
            p = os.path.join(custom_base, name)
            if os.path.isdir(p) and name != "long_video_segments":
                files = [f for f in os.listdir(p) if f.endswith(('.mp4', '.avi'))]
                items.append((name, len(files), p))
        return items

    def delete_enrolled_word(self, word_name):
        """Xóa bỏ thư mục từ vựng nạp nhầm khỏi kho Dataset custom_enrollment"""
        word_clean = normalize_word_name(word_name)
        if not word_clean:
            return False
        target_dir = os.path.join(BASE_DIR, "Research_and_Data", "Dataset", "Sequences", "custom_enrollment", word_clean)
        if os.path.exists(target_dir):
            shutil.rmtree(target_dir)
            if word_clean in self.reference_templates:
                del self.reference_templates[word_clean]
            print(f"🗑️ [AITutorEngine] Đã XÓA THÀNH CÔNG từ vựng '{word_clean}' khỏi kho Dataset!", flush=True)
            return True
        else:
            print(f"⚠️ [AITutorEngine] Không tìm thấy từ vựng '{word_clean}' trong kho custom_enrollment.", flush=True)
            return False

    def rename_enrolled_word(self, old_name, new_name):
        """Đổi tên nhãn từ vựng khi người dùng gán nhầm tên"""
        old_clean = normalize_word_name(old_name)
        new_clean = normalize_word_name(new_name)
        if not old_clean or not new_clean:
            return False
        old_dir = os.path.join(BASE_DIR, "Research_and_Data", "Dataset", "Sequences", "custom_enrollment", old_clean)
        new_dir = os.path.join(BASE_DIR, "Research_and_Data", "Dataset", "Sequences", "custom_enrollment", new_clean)

        if not os.path.exists(old_dir):
            print(f"⚠️ [AITutorEngine] Không tìm thấy từ vựng '{old_clean}'.")
            return False

        if os.path.exists(new_dir):
            # Nếu thư mục mới đã có -> di chuyển các file sang
            for f in os.listdir(old_dir):
                shutil.move(os.path.join(old_dir, f), os.path.join(new_dir, f))
            shutil.rmtree(old_dir)
        else:
            os.rename(old_dir, new_dir)

        if old_clean in self.reference_templates:
            del self.reference_templates[old_clean]
        print(f"✏️ [AITutorEngine] Đã ĐỔI TÊN NHÃN từ '{old_clean}' sang '{new_clean}' thành công!", flush=True)
        return True

    def auto_segment_and_enroll_long_video(self, video_path, max_segments=30):
        """
        Quét video dài đa từ vựng (Continuous Multi-Word Video), tự động nhận diện
        ranh giới chuyển động và cắt thành các phân đoạn mẫu (Sub-clips) để tạo Dataset mới.
        """
        if not video_path:
            print("❌ [AITutorEngine] Đường dẫn video rỗng!")
            return []

        video_path = video_path.strip('\'" \t\r\n')
        if not os.path.exists(video_path):
            print(f"❌ [AITutorEngine] Không tìm thấy tệp video: {video_path}")
            return []

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"❌ [AITutorEngine] Không thể mở file video: {video_path}")
            return []

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration_sec = total_frames / fps

        video_name = os.path.splitext(os.path.basename(video_path))[0]
        print(f"\n🎞️ [AITutorEngine] Đang quét video dài: '{video_name}'", flush=True)
        print(f"   Thông số: {width}x{height} | {total_frames} frames | {fps:.2f} FPS | Thời lượng: {duration_sec:.1f}s ({duration_sec/60:.2f} phút)", flush=True)
        print(f"   Đang phân tích dòng thời gian chuyển động (Motion Scanner)...", flush=True)

        target_base_dir = os.path.join(BASE_DIR, "Research_and_Data", "Dataset", "Sequences", "custom_enrollment", "long_video_segments", video_name)
        os.makedirs(target_base_dir, exist_ok=True)

        # Giai đoạn 1: Quét cường độ chuyển động theo khung hình
        motion_scores = []
        prev_gray = None
        frame_idx = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame_idx += 1
            small_frame = cv2.resize(frame, (160, 90))
            gray = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)
            if prev_gray is None:
                motion_scores.append(0.0)
            else:
                diff = cv2.absdiff(gray, prev_gray)
                score = float(np.mean(diff))
                motion_scores.append(score)
            prev_gray = gray

        cap.release()

        # Giai đoạn 2: Phát hiện các đoạn cử chỉ hoạt động (Activity Windows)
        if not motion_scores:
            print("❌ [AITutorEngine] Không đọc được dữ liệu chuyển động từ video.")
            return []

        mean_motion = np.mean(motion_scores)
        motion_threshold = max(2.5, mean_motion * 0.8)

        active_flags = [s >= motion_threshold for s in motion_scores]
        raw_segments = []
        in_segment = False
        start_f = 0

        for i, is_act in enumerate(active_flags):
            if is_act and not in_segment:
                in_segment = True
                start_f = i
            elif not is_act and in_segment:
                if (i - start_f) >= int(fps * 0.8):  # Tối thiểu 0.8 giây cử chỉ
                    raw_segments.append((start_f, i))
                in_segment = False
        if in_segment and (len(active_flags) - start_f) >= int(fps * 0.8):
            raw_segments.append((start_f, len(active_flags) - 1))

        # Gộp các đoạn gần nhau (khoảng cách nghỉ < 0.6s)
        merged_segments = []
        max_gap_frames = int(fps * 0.6)
        max_seg_frames = int(fps * 4.5)  # Tối đa 4.5s cho 1 từ/cụm từ

        for seg in raw_segments:
            if not merged_segments:
                merged_segments.append(seg)
            else:
                last_start, last_end = merged_segments[-1]
                if (seg[0] - last_end) <= max_gap_frames and (seg[1] - last_start) <= max_seg_frames:
                    merged_segments[-1] = (last_start, seg[1])
                else:
                    merged_segments.append(seg)

        # Giới hạn số lượng segments nếu video quá dài
        selected_segments = merged_segments[:max_segments]
        print(f"✅ [AITutorEngine] Đã phát hiện {len(selected_segments)} phân đoạn cử chỉ hợp lệ!", flush=True)

        # Giai đoạn 3: Cắt và xuất các file video phân đoạn (.mp4)
        cap = cv2.VideoCapture(video_path)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        saved_clips = []

        for idx, (s_frame, e_frame) in enumerate(selected_segments, 1):
            s_time = s_frame / fps
            e_time = e_frame / fps
            seg_filename = f"seg_{idx:03d}_{s_time:.1f}s_to_{e_time:.1f}s.mp4"
            out_clip_path = os.path.join(target_base_dir, seg_filename)

            cap.set(cv2.CAP_PROP_POS_FRAMES, s_frame)
            writer = cv2.VideoWriter(out_clip_path, fourcc, fps, (width, height))

            for f_i in range(s_frame, e_frame + 1):
                r, f = cap.read()
                if not r:
                    break
                writer.write(f)
            writer.release()

            saved_clips.append({
                "segment_index": idx,
                "start_frame": s_frame,
                "end_frame": e_frame,
                "start_time_sec": round(s_time, 2),
                "end_time_sec": round(e_time, 2),
                "duration_sec": round(e_time - s_time, 2),
                "clip_path": out_clip_path
            })
            print(f"   [Đoạn {idx:02d}] ⏱️ {s_time:.1f}s -> {e_time:.1f}s ({e_time-s_time:.1f}s) -> Đã xuất: {seg_filename}", flush=True)

        cap.release()

        # Lưu bản ghi JSON thông tin các phân đoạn
        json_meta = os.path.join(target_base_dir, "segments_metadata.json")
        with open(json_meta, 'w', encoding='utf-8') as jf:
            json.dump({
                "source_video": video_path,
                "total_frames": total_frames,
                "fps": fps,
                "total_segments": len(saved_clips),
                "segments": saved_clips
            }, jf, ensure_ascii=False, indent=2)

        print(f"\n🎉 [AITutorEngine] Hoàn tất nạp & cắt Dataset! Tổng cộng {len(saved_clips)} mẫu video tại:")
        print(f"   📁 {target_base_dir}", flush=True)
        return saved_clips

    def evaluate_student_gesture(self, frame, bboxes, action, conf, target_word):
        """Chẩn đoán cử chỉ học viên thời gian thực và tính điểm 0 - 100%"""
        h, w = frame.shape[:2]
        target_word = target_word.lower()

        status_code = "WAITING"
        score = 0.0
        guidance_msg = "Hãy giơ 2 bàn tay ngang ngực thực hiện cử chỉ!"
        feedback_color = (0, 255, 255) # Vàng

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
            guidance_msg = "Nang tay cao ngang nguc!"
            feedback_color = (0, 0, 255) # Đỏ
            self._speak_guidance_cooldown("Hãy nâng tay cao hơn ngang ngực!")
            return score, status_code, guidance_msg, feedback_color

        # 2. Chẩn đoán độ khớp cử chỉ
        if action and action.lower() == target_word:
            score = min(100.0, float(conf * 100.0))
            status_code = "EXCELLENT"
            guidance_msg = f"DUNG NHAN! Do tin cay {score:.0f}% cho '{target_word.upper()}'"
            feedback_color = (0, 255, 0) # Xanh lá
            self._speak_guidance_cooldown(f"Xuất sắc! Bạn làm đúng cử chỉ từ {target_word} rồi!")
        else:
            score = 0.0
            status_code = "PRACTICING"
            guidance_msg = f"Dang tap tu '{target_word.upper()}'. Giu tay on dinh..."
            feedback_color = (0, 255, 255)

        return score, status_code, guidance_msg, feedback_color

    def run_tutor_split_screen(self, target_word, predictor, cap=None):
        """
        Bật giao diện Gia Sư AI Pro Màn Hình Đôi (Split-Screen Duo-View 1280x480):
        - Nửa Trái: Video Mẫu chuyên gia múa lặp lại liên tục.
        - Nửa Phải: Camera học viên trực tiếp kèm đo điểm & Avatar.
        """
        target_word = target_word.strip().lower()
        ref_video_path = self.get_reference_video(target_word)
        ref_cap = cv2.VideoCapture(ref_video_path) if ref_video_path else None

        owns_cap = cap is None
        if owns_cap:
            cap = cv2.VideoCapture(0)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        print(f"\n🎓 [AI Tutor Pro] Bắt đầu bài tập từ: '{target_word.upper()}'", flush=True)
        if ref_video_path:
            print(f"🎬 [Video Mẫu] Đã nạp video chuyên gia: '{os.path.basename(ref_video_path)}'")
        else:
            print(f"⚠️ Chưa có video mẫu trong kho cho từ '{target_word}'. Camera học viên vẫn chấm điểm AI bình thường.")

        while True:
            ret_student, frame_student = cap.read()
            if not ret_student:
                break

            # 1. Đọc và tạo khung video mẫu bên trái (640x480)
            if ref_cap and ref_cap.isOpened():
                ret_ref, frame_ref = ref_cap.read()
                if not ret_ref:
                    ref_cap.set(cv2.CAP_PROP_POS_FRAMES, 0) # Lặp lại video mẫu
                    ret_ref, frame_ref = ref_cap.read()

                if ret_ref:
                    left_panel = cv2.resize(frame_ref, (640, 480))
                else:
                    left_panel = np.zeros((480, 640, 3), dtype=np.uint8)
            else:
                left_panel = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(left_panel, f"Chua co Video Mau: '{target_word.upper()}'", (60, 240),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (150, 150, 150), 2)
                cv2.putText(left_panel, "Ban co the nap Video Mau o che do [B]", (60, 280),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1)

            # Đính nhãn tiêu đề cho Nửa Trái
            cv2.rectangle(left_panel, (0, 0), (640, 50), (20, 20, 20), -1)
            cv2.putText(left_panel, f"1. VIDEO MAU CHUAN: '{target_word.upper()}'", (20, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)

            # 2. Xử lý Camera học viên ở Nửa Phải (640x480)
            right_panel = cv2.resize(frame_student, (640, 480))
            static_char, action, conf, bboxes = predictor.process_frame(right_panel)

            if bboxes:
                for (x1, y1, x2, y2, hand_label) in bboxes:
                    color = (0, 255, 0) if "Left" in hand_label else (255, 0, 0)
                    cv2.rectangle(right_panel, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(right_panel, hand_label, (x1, max(15, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

            score, status, msg, color = self.evaluate_student_gesture(right_panel, bboxes, action, conf, target_word)

            # Đính nhãn tiêu đề cho Nửa Phải & Thanh điểm số
            cv2.rectangle(right_panel, (0, 0), (640, 50), (20, 20, 20), -1)
            cv2.putText(right_panel, "2. CAMERA HOC VIEN (LIVE)", (20, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)

            # Thanh đo điểm số (Progress Gauge Bar 0 - 100%)
            bar_w = int((score / 100.0) * 200)
            cv2.rectangle(right_panel, (420, 15), (620, 35), (50, 50, 50), -1)
            cv2.rectangle(right_panel, (420, 15), (420 + bar_w, 35), color, -1)
            cv2.putText(right_panel, f"{score:.0f}%", (505, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

            # Banner thông báo hướng dẫn ở đáy Nửa Phải
            cv2.rectangle(right_panel, (0, 430), (640, 480), (15, 15, 15), -1)
            cv2.putText(right_panel, msg, (15, 465), cv2.FONT_HERSHEY_SIMPLEX, 0.60, color, 2)

            # 3. Ghép 2 màn hình cạnh nhau tạo khung hình đôi 1280x480
            split_screen = np.hstack([left_panel, right_panel])

            # Viền ngăn cách giữa 2 màn hình
            cv2.line(split_screen, (640, 0), (640, 480), (0, 255, 255), 3)

            cv2.imshow("AI Sign Tutor Pro - Split Screen Duo Practice (1280x480)", split_screen)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), ord('Q')):
                break

        if ref_cap:
            ref_cap.release()
        if owns_cap and cap:
            cap.release()
        cv2.destroyAllWindows()

    def _speak_guidance_cooldown(self, text):
        """Phát âm thanh nhắc nhở có giới hạn thời gian nghỉ để tránh nhái giọng liên tục."""
        now = time.time()
        if now - self.last_speech_time > self.speech_cooldown:
            if self.tts:
                self.tts.speak(text)
            self.last_speech_time = now
