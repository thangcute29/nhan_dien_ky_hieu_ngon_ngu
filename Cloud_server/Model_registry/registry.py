# cloud_server/model_registry/registry.py
import os
import json
from datetime import datetime

MODEL_STORE = "cloud_server/model_registry/store"
VERSION_FILE = os.path.join(MODEL_STORE, "versions.json")

class ModelRegistry:
    def __init__(self):
        os.makedirs(MODEL_STORE, exist_ok=True)
        if not os.path.exists(VERSION_FILE):
            with open(VERSION_FILE, 'w') as f:
                json.dump({"latest": "1.0.0", "history": []}, f)

    def get_latest_version(self):
        with open(VERSION_FILE, 'r') as f:
            data = json.load(f)
        return data["latest"]

    def register_new_version(self, version, yolo_path, feature_path, gru_path):
        # Lưu model vào thư mục versioned
        version_dir = os.path.join(MODEL_STORE, version)
        os.makedirs(version_dir, exist_ok=True)
        # Copy model files
        import shutil
        shutil.copy(yolo_path, os.path.join(version_dir, "hand_det_yolo.pt"))
        shutil.copy(feature_path, os.path.join(version_dir, "feature_extractor.h5"))
        shutil.copy(gru_path, os.path.join(version_dir, "action_recognizer.h5"))

        # Cập nhật versions.json
        with open(VERSION_FILE, 'r') as f:
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
        with open(VERSION_FILE, 'w') as f:
            json.dump(data, f, indent=2)

    def get_model_path(self, model_type, version):
        with open(VERSION_FILE, 'r') as f:
            data = json.load(f)
        for entry in data["history"]:
            if entry["version"] == version:
                return entry["models"].get(model_type)
        return None