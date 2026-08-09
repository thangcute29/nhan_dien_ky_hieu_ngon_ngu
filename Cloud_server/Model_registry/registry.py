# Cloud_server/Model_registry/registry.py
import os
import json
import shutil
from datetime import datetime

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
MODEL_STORE = os.path.join(BASE_DIR, "Cloud_server", "Model_registry", "store")
VERSION_FILE = os.path.join(MODEL_STORE, "versions.json")

class ModelRegistry:
    def __init__(self):
        os.makedirs(MODEL_STORE, exist_ok=True)
        if not os.path.exists(VERSION_FILE):
            with open(VERSION_FILE, 'w', encoding='utf-8') as f:
                json.dump({"latest": "1.0.0", "history": []}, f)

    def get_latest_version(self):
        with open(VERSION_FILE, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data["latest"]

    def register_new_version(self, version, yolo_path, feature_path, gru_path):
        # Lưu model vào thư mục versioned
        version_dir = os.path.join(MODEL_STORE, version)
        os.makedirs(version_dir, exist_ok=True)
        # Copy model files
        if os.path.exists(yolo_path):
            shutil.copy(yolo_path, os.path.join(version_dir, "hand_det_yolo.pt"))
        if os.path.exists(feature_path):
            shutil.copy(feature_path, os.path.join(version_dir, "feature_extractor.h5"))
        if os.path.exists(gru_path):
            shutil.copy(gru_path, os.path.join(version_dir, "action_recognizer.h5"))

        # Cập nhật versions.json
        with open(VERSION_FILE, 'r', encoding='utf-8') as f:
            data = json.load(f)
        data["latest"] = version
        data["history"].append({
            "version": version,
            "timestamp": datetime.now().isoformat(),
            "models": {
                "yolo": os.path.join(version_dir, "hand_det_yolo.pt"),
                "feature": os.path.join(version_dir, "feature_extractor.h5"),
                "gru": os.path.join(version_dir, "action_recognizer.h5")
            }
        })
        with open(VERSION_FILE, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)

    def get_model_path(self, model_type, version):
        with open(VERSION_FILE, 'r', encoding='utf-8') as f:
            data = json.load(f)
        for entry in data["history"]:
            if entry["version"] == version:
                return entry["models"].get(model_type)
        return None