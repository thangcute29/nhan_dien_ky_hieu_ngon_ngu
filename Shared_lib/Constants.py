import os
import pickle

LLM_SYSTEM_PROMPT = """
Bạn là một trợ lý chuyên sửa lỗi và hoàn thiện các từ được đánh vần bằng thủ ngữ (ngôn ngữ ký hiệu).
Người dùng là người khiếm thính và đang sử dụng bảng chữ cái ngón tay (fingerspelling).
Khi nhận được một chuỗi các chữ cái thô, hãy sửa các lỗi chính tả rõ ràng, loại bỏ các chữ cái bị lặp dư thừa và đưa ra từ hoặc cụm từ có khả năng đúng nhất.
Nếu đầu vào đã là một từ có nghĩa, hãy giữ nguyên từ đó.
Nếu đầu vào có vẻ bị sai chính tả, hãy gợi ý từ đúng.
CHỈ trả về từ/cụm từ đã được sửa, KHÔNG giải thích gì thêm.
"""

ALPHABET_CLASSES = [chr(i) for i in range(ord('A'), ord('Z') + 1)]

_INFO_PKL_PATH = os.path.join(os.path.dirname(__file__), "Assets", "action_recognizer_info.pkl")
if os.path.exists(_INFO_PKL_PATH):
    try:
        with open(_INFO_PKL_PATH, 'rb') as _f:
            _info = pickle.load(_f)
            ACTION_CLASSES = _info.get('classes', [])
    except Exception:
        ACTION_CLASSES = []
else:
    ACTION_CLASSES = []