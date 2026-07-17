# SIGN_LANGUAGE_MARKET_READY/cloud_server/api/main.py
import os
import shutil
import uuid
from datetime import datetime
from fastapi import FastAPI, File, UploadFile, Form, BackgroundTasks
from fastapi.responses import JSONResponse, FileResponse
import uvicorn
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
import config
from cloud_server.model_registry.registry import ModelRegistry
from cloud_server.api.models import EdgeCaseUpload

app = FastAPI(title="Sign Language Cloud API")
registry = ModelRegistry()

# Đảm bảo thư mục edge cases tồn tại
os.makedirs(config.EDGE_CASES_PENDING, exist_ok=True)
os.makedirs(config.EDGE_CASES_LABELED, exist_ok=True)

@app.post("/upload_edge_case")
async def upload_edge_case(
    video: UploadFile = File(...),
    label: str = Form(None)
):
    """Nhận video edge case từ mobile app."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    unique_id = str(uuid.uuid4())[:8]
    filename = f"{timestamp}_{unique_id}_{video.filename}"
    filepath = os.path.join(config.EDGE_CASES_PENDING, filename)

    with open(filepath, "wb") as buffer:
        shutil.copyfileobj(video.file, buffer)

    if label:
        new_filename = f"{timestamp}_{unique_id}_pred_{label}_{video.filename}"
        new_filepath = os.path.join(config.EDGE_CASES_PENDING, new_filename)
        os.rename(filepath, new_filepath)
        filepath = new_filepath

    return JSONResponse({
        "status": "success",
        "message": "Edge case uploaded",
        "file_id": os.path.basename(filepath)
    })

@app.get("/model_version")
async def get_model_version():
    """Trả về phiên bản model hiện tại."""
    current_version = registry.get_latest_version()
    return {
        "version": current_version,
        "update_available": False,
        "models": {
            "yolo": f"/download_model/yolo/{current_version}",
            "feature_extractor": f"/download_model/feature/{current_version}",
            "gru": f"/download_model/gru/{current_version}"
        }
    }

@app.get("/download_model/{model_type}/{version}")
async def download_model(model_type: str, version: str):
    """Tải model cập nhật."""
    model_path = registry.get_model_path(model_type, version)
    if model_path and os.path.exists(model_path):
        return FileResponse(
            model_path, 
            media_type="application/octet-stream", 
            filename=os.path.basename(model_path)
        )
    return JSONResponse({"error": "Model not found"}, status_code=404)

@app.post("/trigger_retrain")
async def trigger_retrain(background_tasks: BackgroundTasks):
    """Kích hoạt huấn luyện lại (admin only)."""
    from cloud_server.trainer.retrain import run_retraining
    background_tasks.add_task(run_retraining)
    return {"status": "Retraining started in background"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)