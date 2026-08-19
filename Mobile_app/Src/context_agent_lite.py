# Mobile_app/Src/context_agent_lite.py
"""
Phiên bản thu gọn (Lite) của ContextAgent dành riêng cho thiết bị Di động / Offline
Tự động tra cứu từ điển nhanh mà không cần phụ thuộc vào mạng Internet hay Cloud API Server.
"""

import sys
import os

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

try:
    from Cloud_server.Api.context_agent import ContextAgent, VI_DICTIONARY, JA_DICTIONARY, KO_DICTIONARY
    ContextAgentLite = ContextAgent
except ImportError:
    VI_DICTIONARY = {
        'apple': 'quả táo', 'mother': 'người mẹ', 'hello': 'xin chào', 'hi': 'chào bạn',
        'thanks': 'cảm ơn', 'bye': 'tạm biệt', 'help': 'trợ giúp', 'yes': 'vâng/có', 'no': 'không'
    }

    class ContextAgentLite:
        """Bộ dịch thuật ngoại tuyến siêu nhẹ cho Mobile App"""
        def __init__(self, llm_corrector=None):
            self.llm_corrector = llm_corrector

        def process(self, action_word, target_lang='vi'):
            if not action_word:
                return ""
            words = action_word.lower().split()
            translated = [VI_DICTIONARY.get(w, w) if target_lang == 'vi' else w for w in words]
            return " ".join(translated)
