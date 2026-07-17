# File: Cloud_server/Api/Main.py
import os
import sys
import json
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from pydantic import BaseModel
from typing import List, Optional

# Thêm thư mục gốc vào sys.path để import các module dùng chung
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
import config
from Cloud_server.Api.llm_corrector import LLMCorrector
from Cloud_server.Database.edge_cases.learning_manager import LearningManager

app = FastAPI(title="Sign Language Hybrid Cloud Server API")

# Khởi tạo các bộ não xử lý tại Server
llm_corrector = LLMCorrector()  # Tự động cấu hình Gemini + Ollama Fallback từ bài cũ
learning_manager = LearningManager()

# --- ĐỊNH NẠNG DỮ LIỆU ĐẦU VÀO (PYDANTIC MODELS) ---
class EdgeCasePayload(BaseModel):
    user_id: str
    raw_data: list              # Chuỗi Landmarks hoặc base64 ảnh lỗi
    context_history: List[str]  # Lịch sử 7 câu gần nhất gửi từ App/Web
    confidence: float

class UserFeedbackPayload(BaseModel):
    filename: str               # Tên file JSON lỗi đã lưu trong Unverified
    corrected_text: str         # Câu chuẩn do người dùng sửa lại

# --- CÁC ENDPOINT API HỆ THỐNG ---

@app.post("/api/v2/resolve_edge_case")
async def resolve_edge_case(payload: EdgeCasePayload):
    """
    Endpoint HTTP: Tiếp nhận ca khó khi App/Web bị rối loạn ký hiệu cục bộ.
    """
    try:
        # 1. Gom chuỗi từ thô và lịch sử ngữ cảnh để AI Agent xử lý
        raw_text_sentence = " ".join(payload.context_history)
        
        # 2. Gọi AI Agent (Gemini/Ollama) sửa lỗi ngữ cảnh sâu
        corrected_text = llm_corrector.correct(raw_text_sentence)
        
        # 3. TỰ ĐỘNG PHÂN LOẠI NGẦM (Implicit Feedback Pipeline):
        # Nếu độ tin cậy cực thấp, ném vào kho Unverified_Learning để check sau
        if payload.confidence < 0.60:
            learning_manager.save_to_unverified(
                user_id=payload.user_id,
                raw_data=payload.raw_data,
                ai_predicted_text=corrected_text
            )
        else:
            # Nếu AI khá tự tin, ném thẳng vào Labeled để đêm đến retrain luôn
            # Sử dụng chính câu đã được AI tối ưu làm nhãn tạm chuẩn
            # (Chúng ta sẽ đổi tên file hoặc lưu cấu trúc tương đương để Retrain hốt đi)
            pass 

        # 4. Trả kết quả tức thì về cho người dùng (Không bắt người dùng đợi)
        return {"status": "success", "corrected_text": corrected_text}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v2/user_feedback")
async def receive_user_feedback(payload: UserFeedbackPayload):
    """
    Endpoint Implicit Feedback: Kích hoạt khi người dùng bấm nút "Sửa lại câu dịch".
    Hệ thống sẽ lấy câu sửa này làm nhãn chuẩn, chuyển file sang thư mục Labeled để retrain.
    """
    success = learning_manager.promote_to_labeled(
        filename=payload.filename,
        correct_label=payload.corrected_text
    )
    if success:
        return {"status": "success", "message": "Dữ liệu đã được gán nhãn chuẩn và chuyển sang lò train."}
    else:
        raise HTTPException(status_code=400, detail="Không thể xử lý file phản hồi.")


# --- KẾT NỐI WEBSOCKET DÀNH CHO LIVESTREAM / VIDEO CALL THỜI GIAN THỰC ---
@app.websocket("/live-agent")
async def websocket_endpoint(websocket: WebSocket):
    """
    Đường ống WebSocket thông suốt: Nhận landmarks ca khó liên tục từ luồng Live
    và stream ngược lại chữ đã sửa cho Client mà không làm gián đoạn Video.
    """
    await websocket.accept()
    print("[WebSocket] Thiết lập kết nối thời gian thực với Client thành công.")
    
    try:
        while True:
            # Nhận dữ liệu streaming JSON từ App/Web gửi lên
            data_str = await websocket.receive_text()
            payload = json.loads(data_str)
            
            user_id = payload.get("user_id")
            raw_data = payload.get("raw_hand_landmarks")
            context_history = payload.get("context_history", [])
            
            # Xử lý nhanh bằng AI Agent ngữ cảnh
            raw_sentence = " ".join(context_history)
            corrected_text = llm_corrector.correct(raw_sentence)
            
            # Âm thầm ném vào kho dữ liệu tự học ngầm để tối ưu hóa
            learning_manager.save_to_unverified(user_id, raw_data, corrected_text)
            
            # Bắn chữ trả về ngay lập tức đè lên màn hình Livestream dạng phụ đề
            response = {"corrected_text": corrected_text}
            await websocket.send_text(json.dumps(response))
            
    except WebSocketDisconnect:
        print("[WebSocket] Client đã ngắt kết nối cuộc gọi/livestream.")
    except Exception as e:
        print(f"[WebSocket Error] Có lỗi xảy ra: {e}")