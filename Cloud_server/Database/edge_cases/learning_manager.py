# Cloud_server/Database/edge_cases/learning_manager.py
import os
import time
import json
import re
import uuid
from pathlib import Path
import config


SAFE_COMPONENT = re.compile(r"[^A-Za-z0-9_-]+")


def _safe_component(value, fallback):
    cleaned = SAFE_COMPONENT.sub("_", str(value)).strip("_-")
    return cleaned[:80] or fallback

class LearningManager:
    """
    Quản lý luồng dữ liệu tự học (Implicit Feedback Loop):
    - save_to_unverified: Lưu dữ liệu nghi vấn / chưa kiểm chứng
    - promote_to_labeled: Chuyển dữ liệu có phản hồi / đã gán nhãn chuẩn sang thư mục Labeled
    """
    def __init__(self):
        self.unverified_dir = getattr(config, 'UNVERIFIED_LEARNING_DIR', os.path.join(config.BASE_DIR, "Cloud_server", "Database", "edge_cases", "Unverified_Learning"))
        self.labeled_dir = getattr(config, 'LABELED_DIR', os.path.join(config.BASE_DIR, "Cloud_server", "Database", "edge_cases", "Labeled"))
        os.makedirs(self.unverified_dir, exist_ok=True)
        os.makedirs(self.labeled_dir, exist_ok=True)

    def save_to_unverified(self, user_id, raw_data, ai_predicted_text):
        """Lưu file json dữ liệu ca khó vào Unverified_Learning"""
        safe_user_id = _safe_component(user_id, "anonymous")
        filename = f"{int(time.time() * 1000)}_{uuid.uuid4().hex[:8]}_{safe_user_id}.json"
        filepath = os.path.join(self.unverified_dir, filename)
        data = {
            "user_id": user_id,
            "raw_data": raw_data,
            "ai_predicted_text": ai_predicted_text,
            "timestamp": time.time()
        }
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        return filename

    def promote_to_labeled(self, filename, correct_label):
        """Di chuyển file json từ Unverified_Learning sang Labeled kèm nhãn chuẩn do user/chuyên gia gán"""
        safe_filename = Path(filename).name
        if safe_filename != filename or not safe_filename.lower().endswith('.json'):
            return False
        label = _safe_component(correct_label.lower(), "")
        if not label or label != correct_label.strip().lower().replace(" ", "_"):
            # Retraining accepts one isolated gloss, not an arbitrary sentence.
            return False
        src_path = os.path.join(self.unverified_dir, safe_filename)
        if not os.path.isfile(src_path):
            return False
        try:
            with open(src_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            data["correct_label"] = correct_label
            label_dir = os.path.join(self.labeled_dir, label)
            os.makedirs(label_dir, exist_ok=True)
            dst_path = os.path.join(label_dir, safe_filename)
            with open(dst_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            os.remove(src_path)
            return True
        except Exception as e:
            print(f"Error promoting file to labeled: {e}")
            return False
