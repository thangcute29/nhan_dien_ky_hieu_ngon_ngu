# module sửa lỗi văn bản bằng AI (LLM).
# mobile_app/src/inference_engine/llm_corrector.py
import os
import sys

sys.path.append(
    os.path.join(os.path.dirname(__file__), "..", "..", "..")
)  # Vì project AI thường có nhiều folder để tránh lỗi import và tiện chạy file từ nhiều nơi
import config

try:
    from Shared_lib.Constants import LLM_SYSTEM_PROMPT
except ImportError:
    from shared_lib.constants import LLM_SYSTEM_PROMPT


class LLMCorrector:
    def __init__(self, provider=None):
        self.provider = provider or config.LLM_PROVIDER
        self.model = config.LLM_MODEL
        self.enabled = config.USE_LLM_CORRECTION
        self.client = None
        self.llm = None
        self.genai_types = None

        if not self.enabled:
            print("LLM correction disabled.")
            return

        if self.provider in {"auto", "google"}:
            if self._try_google():
                self.provider = "google"
            else:
                print("Google Gemini không khả dụng, sẽ thử Ollama local nếu có...")
                if self._try_ollama():
                    self.provider = "ollama"
                else:
                    self.enabled = False
        elif self.provider == "ollama":
            if not self._try_ollama():
                self.enabled = False
        else:
            print(f"Provider không hỗ trợ: {self.provider}")
            self.enabled = False

    def _try_google(self):
        if not config.LLM_API_KEY:
            print("Chưa cấu hình GEMINI_API_KEY; bỏ qua Gemini.")
            return False
        try:
            from google import genai
            from google.genai import types

            self.genai_types = types
            self.client = genai.Client(api_key=config.LLM_API_KEY)
            return True
        except ImportError:
            print("Chưa cài thư viện google-genai. Chạy: pip install google-genai")
        except Exception as e:
            print(f"Google Gemini khởi tạo lỗi: {e}")
        return False

    def _try_ollama(self):
        try:
            from langchain_community.llms import Ollama

            self.llm = Ollama(
                model=self.model,
                base_url=config.LLM_BASE_URL or "http://localhost:11434",
            )
            return True
        except ImportError:
            print(
                "Chưa cài thư viện langchain-community. Chạy: pip install langchain-community"
            )
        except Exception as e:
            print(f"Ollama khởi tạo lỗi: {e}")
        return False

    def correct(self, text, history_context=None):
        if not self.enabled or not text:
            return text
        try:
            # Nếu có ngữ cảnh, bổ sung vào đầu để AI hiểu mạch hội thoại
            prompt_text = text
            if history_context:
                prompt_text = f"Ngữ cảnh các câu trước đó: {history_context}\nDựa vào ngữ cảnh trên, hãy dịch từ/câu sau sao cho logic và tự nhiên nhất: {text}"
                
            if self.provider == "google":
                response = self.client.models.generate_content(
                    model=self.model,
                    contents=prompt_text,
                    config=self.genai_types.GenerateContentConfig(
                        system_instruction=LLM_SYSTEM_PROMPT,
                        temperature=0.1,  
                        max_output_tokens=50,  
                    ),
                )
                return response.text.strip()
            elif self.provider == "ollama":
                if self.client is not None and self._check_google():
                    print("✅ Có mạng trở lại → quay về Google Gemini")
                    self.provider = "google"
                    return self.correct(text, history_context)
                prompt = f"{LLM_SYSTEM_PROMPT}\n\nInput: {prompt_text}\nCorrected:"
                return self.llm.invoke(prompt).strip()

        except Exception as e:
            print(f"LLM correction error: {e}")
            if self.provider == "google":
                print("Chuyển sang Ollama local nếu có...")
                if self._try_ollama():
                    self.provider = "ollama"
                    return self.correct(text)
            return text

    def _check_google(self):
        # Thử ping Google để kiểm tra mạng
        try:
            import requests
            requests.get("https://generativelanguage.googleapis.com", timeout=2)
            return True
        except:
            return False
    

# ============================OPENAI SYSTEM PROMPT NẾU SỬ DỤNG============================
# nếu sử dụng openai thì phải nhớ là system prompt cần phải truyền mỗi lần gọi , role là ai đang nói , content là nói vấn đề gì , system là người lập trình viên - ra lệnh cho AI , user là người dùng - gửi yêu cầu , assistant là AI - trả lời yêu cầu của user bằng cách coi lại lịch sử hội thoạt
# và nếu sử dụng thì các vai trò cần phải được viết thường không viết hoa nếu không sẽ lỗi {"role": "system", "content": "..."}  # ✅ Đúng
# OpenAI bọc nhiều lớp để hỗ trợ tính năng trả về nhiều câu trả lời cùng lúc (choices[]), còn Gemini không có tính năng đó nên trả thẳng .text luôn.
# openai cần phải lấy phần tử dầu tiên của choices[] để lấy câu trả lời chính, còn Gemini thì trả thẳng phần content của phản hồi. Đây là điểm khác biệt quan trọng khi xử lý phản hồi từ hai nhà cung cấp LLM khác nhau.


# ============================GOOGLE GEMINI SYSTEM PROMPT NẾU SỬ DỤNG============================
# google-generativeai Cấu hình toàn cục (Global). Thiết lập một lần, dùng cho mọi nơi , Khó dùng nếu em có nhiều API Key khác nhau trong cùng một app , Dành riêng cho các mô hình Gemini đời đầu

# =====================langchain ==============
# vì lý do nếu pip install langchain-community thì kéo theo hàng chục thư viện phụ thuộc khác
# tăng dung lượng app mobile lên đáng kể khỏi động chậm hơn chỉ để gọi 1 dòng du nhất
#
