# Cloud_server/Database/edge_cases/learning_manager.py
import os
import time
import json
import config

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
        filename = f"{int(time.time() * 1000)}_{user_id}.json"
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
        src_path = os.path.join(self.unverified_dir, filename)
        if not os.path.exists(src_path):
            return False
        try:
            with open(src_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            data["correct_label"] = correct_label
            dst_path = os.path.join(self.labeled_dir, filename)
            with open(dst_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            os.remove(src_path)
            return True
        except Exception as e:
            print(f"Error promoting file to labeled: {e}")
            return False
