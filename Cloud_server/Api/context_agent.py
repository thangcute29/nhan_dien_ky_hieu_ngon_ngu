# Cloud_server/Api/context_agent.py

# Từ điển dịch từ vựng thủ ngữ cơ bản (100 từ vựng WLASL) sang Tiếng Việt
VI_DICTIONARY = {
    'accident': 'tai nạn', 'add': 'thêm vào', 'alone': 'cô đơn', 'animal': 'động vật',
    'apple': 'quả táo', 'appointment': 'cuộc hẹn', 'argue': 'tranh cãi', 'bad': 'xấu/tệ',
    'balance': 'cân bằng', 'bar': 'quán bar', 'basketball': 'bóng rổ', 'because': 'bởi vì',
    'bed': 'cái giường', 'before': 'trước khi', 'big': 'to lớn', 'bird': 'con chim',
    'black': 'màu đen', 'blanket': 'cái chăn', 'bowling': 'trò bowling', 'brother': 'anh/em trai',
    'call': 'gợi/gọi điện', 'candy': 'kẹo', 'catch': 'bắt lấy', 'champion': 'nhà vô địch',
    'change': 'thay đổi', 'cheat': 'gian lận', 'check': 'kiểm tra', 'cold': 'lạnh',
    'computer': 'máy tính', 'convince': 'thuyết phục', 'cool': 'mát mẻ/ngầu', 'corn': 'bắp/ngô',
    'country': 'đất nước', 'cousin': 'anh chị em họ', 'cry': 'khóc', 'dark': 'tối',
    'daughter': 'con gái', 'deaf': 'khiếm thính', 'delay': 'trì hoãn', 'delicious': 'ngon miệng',
    'different': 'khác biệt', 'discuss': 'thảo luận', 'dive': 'lặn', 'doctor': 'bác sĩ',
    'dog': 'con chó', 'drink': 'uống', 'dry': 'khô ráo', 'environment': 'môi trường',
    'example': 'ví dụ', 'family': 'gia đình', 'far': 'xa', 'fat': 'béo/mập',
    'fault': 'lỗi sai', 'fish': 'con cá', 'full': 'đầy/no', 'future': 'tương lai',
    'give': 'cho/tặng', 'go': 'đi', 'good': 'tốt', 'government': 'chính phủ',
    'graduate': 'tốt nghiệp', 'help': 'trợ giúp', 'hot': 'nóng', 'improve': 'cải thiện',
    'inform': 'thông báo', 'interest': 'sự quan tâm', 'join': 'tham gia', 'kiss': 'hôn',
    'language': 'ngôn ngữ', 'last': 'cuối cùng', 'later': 'sau này', 'laugh': 'cười',
    'leave': 'rời đi', 'letter': 'lá thư', 'like': 'thích', 'make': 'làm/tạo',
    'man': 'đàn ông', 'many': 'nhiều', 'mean': 'có nghĩa là', 'mother': 'người mẹ',
    'move': 'di chuyển', 'no': 'không', 'orange': 'quả cam', 'order': 'đặt hàng',
    'perspective': 'góc nhìn', 'pink': 'màu hồng', 'pizza': 'bánh pizza', 'play': 'chơi',
    'room': 'căn phòng', 'score': 'điểm số', 'shirt': 'áo sơ mi', 'short': 'ngắn/thấp',
    'take': 'lấy', 'tall': 'cao', 'thanksgiving': 'lễ tạ ơn', 'thin': 'gầy/mỏng',
    'trade': 'trao đổi', 'what': 'cái gì', 'who': 'ai', 'yes': 'vâng/có'
}

JA_DICTIONARY = {
    'apple': 'リンゴ', 'shirt': 'シャツ', 'hello': 'こんにちは', 'help': '助けて',
    'yes': 'はい', 'no': 'いいえ', 'cousin': 'いとこ', 'brother': '兄弟', 'mother': '母'
}

KO_DICTIONARY = {
    'apple': '사과', 'shirt': '셔츠', 'hello': '안녕하세요', 'help': '도와주세요',
    'yes': '네', 'no': '아니오', 'cousin': '사촌', 'brother': '형제', 'mother': '어머니'
}


class ContextAgent:
    def __init__(self, llm_corrector=None):
        self.llm_corrector = llm_corrector

    def process(self, action_word, target_lang='vi'):
        if not action_word:
            return ""

        import config

        # SỰ CỐ 4 FIX: Tách chuỗi từ vựng (ví dụ: "apple mother" -> ["apple", "mother"])
        words = action_word.lower().split()
        translated_words = []

        for w in words:
            if target_lang == 'vi':
                translated_words.append(VI_DICTIONARY.get(w, w))
            elif target_lang == 'ja':
                translated_words.append(JA_DICTIONARY.get(w, w))
            elif target_lang == 'ko':
                translated_words.append(KO_DICTIONARY.get(w, w))
            else:
                translated_words.append(w)

        final_str = " ".join(translated_words)

        # Chỉ gọi LLM khi API Key hợp lệ (không phải key mẫu "AIzaSy...")
        api_key = getattr(config, 'LLM_API_KEY', '')
        if self.llm_corrector and hasattr(self.llm_corrector, 'correct') and api_key and not api_key.startswith("AIzaSy..."):
            try:
                llm_out = self.llm_corrector.correct(final_str)
                if llm_out:
                    return llm_out
            except Exception:
                pass

        return final_str # Trả về bản dịch Tiếng Việt Offline mượt mà & chính xác!

