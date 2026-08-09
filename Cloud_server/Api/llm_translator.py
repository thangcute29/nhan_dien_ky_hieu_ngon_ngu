# Cloud_server/Api/llm_translator.py

class LLMTranslator:
    def __init__(self, llm_corrector=None):
        self.llm_corrector = llm_corrector

    def translate(self, text, target_lang="vi"):
        if not text:
            return ""
        return text
