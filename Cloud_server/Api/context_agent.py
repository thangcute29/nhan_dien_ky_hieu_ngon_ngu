# Cloud_server/Api/context_agent.py

# ==============================================================================
# TỪ ĐIỂN DỊCH THUẬT ĐA NGÔN NGỮ THỦ NGỮ (SIGN LANGUAGE MULTILINGUAL DICTIONARY)
# Bao gồm: 100 từ vựng WLASL + Từ vựng giao tiếp ASL thông dụng + Bảng chữ cái ngón tay
# ==============================================================================

VI_DICTIONARY = {
    # --- 1. TOÀN BỘ 100 TỪ VỰNG HUẤN LUYỆN WLASL ---
    'accident': 'tai nạn', 'add': 'thêm vào', 'alone': 'cô đơn', 'animal': 'động vật',
    'apple': 'quả táo', 'appointment': 'cuộc hẹn', 'argue': 'tranh cãi', 'bad': 'xấu/tệ',
    'balance': 'cân bằng', 'bar': 'quán bar', 'basketball': 'bóng rổ', 'because': 'bởi vì',
    'bed': 'cái giường', 'before': 'trước khi', 'big': 'to lớn', 'bird': 'con chim',
    'black': 'màu đen', 'blanket': 'cái chăn', 'bowling': 'trò bowling', 'brother': 'anh/em trai',
    'call': 'gọi điện', 'candy': 'kẹo', 'catch': 'bắt lấy', 'champion': 'nhà vô địch',
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
    'trade': 'trao đổi', 'what': 'cái gì', 'who': 'ai', 'yes': 'vâng/có',
    'africa': 'châu Phi', 'all': 'tất cả', 'birthday': 'sinh nhật', 'but': 'nhưng',
    'can': 'có thể', 'chair': 'cái ghế', 'city': 'thành phố', 'clothes': 'quần áo',
    'color': 'màu sắc', 'cook': 'nấu ăn', 'cow': 'con bò', 'decide': 'quyết định',
    'enjoy': 'thích thú', 'finish': 'hoàn thành', 'forget': 'quên', 'hat': 'cái mũ',
    'hearing': 'thính giác', 'jacket': 'áo khoác', 'medicine': 'thuốc', 'meet': 'gặp',
    'now': 'bây giờ', 'paint': 'sơn/vẽ', 'paper': 'giấy', 'pull': 'kéo',
    'right': 'đúng/bên phải', 'same': 'giống nhau', 'secretary': 'thư ký',
    'table': 'cái bàn', 'tell': 'kể/nói', 'thursday': 'thứ năm',
    'wrong': 'sai', 'year': 'năm',

    # --- 2. CHÀO HỎI & GIAO TIẾP HÀNG NGÀY ---
    'hello': 'xin chào', 'hi': 'chào bạn', 'hey': 'này bạn', 'goodbye': 'tạm biệt',
    'bye': 'tạm biệt', 'thanks': 'cảm ơn', 'thank': 'cảm ơn', 'thankyou': 'cảm ơn bạn',
    'welcome': 'chào mừng', 'sorry': 'xin lỗi', 'excuse': 'xin phép', 'please': 'làm ơn',
    'ok': 'đồng ý', 'okay': 'được rồi', 'fine': 'ổn/tốt', 'nice': 'tuyệt vời',

    # --- 3. ĐẠI TỪ & CÂU HỎI ---
    'i': 'tôi', 'me': 'tôi', 'my': 'của tôi', 'you': 'bạn', 'your': 'của bạn',
    'he': 'anh ấy', 'she': 'cô ấy', 'we': 'chúng tôi', 'they': 'họ',
    'where': 'ở đâu', 'when': 'khi nào', 'why': 'tại sao', 'how': 'như thế nào',
    'which': 'cái nào', 'howmuch': 'bao nhiêu',

    # --- 4. GIA ĐÌNH & CON NGƯỜI ---
    'father': 'người bố', 'dad': 'bố', 'mom': 'mẹ', 'parent': 'cha mẹ',
    'sister': 'chị/em gái', 'son': 'con trai', 'baby': 'em bé', 'child': 'trẻ em',
    'friend': 'bạn bè', 'teacher': 'thầy cô giáo', 'student': 'học sinh', 'person': 'con người',
    'boy': 'cậu bé', 'girl': 'cô bé', 'woman': 'phụ nữ',

    # --- 5. HÀNH ĐỘNG & ĐỘNG TỪ THÔNG DỤNG ---
    'eat': 'ăn', 'sleep': 'ngủ', 'see': 'nhìn thấy', 'look': 'nhìn', 'hear': 'nghe',
    'listen': 'lắng nghe', 'speak': 'nói', 'talk': 'trò chuyện', 'walk': 'đi bộ',
    'run': 'chạy', 'study': 'học tập', 'learn': 'học hỏi', 'work': 'làm việc',
    'need': 'cần', 'want': 'muốn', 'love': 'yêu thương', 'hate': 'ghét',
    'buy': 'mua', 'sell': 'bán', 'pay': 'thanh toán', 'wait': 'chờ đợi',
    'stop': 'dừng lại', 'start': 'bắt đầu', 'open': 'mở ra', 'close': 'đóng lại',
    'read': 'đọc sách', 'write': 'viết', 'sing': 'hát', 'dance': 'nhảy múa',

    # --- 6. CẢM XÚC & TÍNH TỪ ---
    'happy': 'vui vẻ', 'sad': 'buồn bã', 'angry': 'tức giận', 'tired': 'mệt mỏi',
    'hungry': 'đói bụng', 'thirsty': 'khát nước', 'sick': 'ốm/bệnh', 'pain': 'đau đớn',
    'strong': 'khỏe mạnh', 'weak': 'yếu ớt', 'smart': 'thông minh', 'busy': 'bận rộn',
    'fast': 'nhanh chóng', 'slow': 'chậm chạp', 'easy': 'dễ dàng', 'hard': 'khó khăn',
    'clean': 'sạch sẽ', 'dirty': 'dơ bẩn', 'new': 'mới', 'old': 'cũ/già',

    # --- 7. ĐỒ VẬT, MÔI TRƯỜNG & THỨC ĂN ---
    'water': 'nước uống', 'food': 'thức ăn', 'rice': 'cơm/gạo', 'bread': 'bánh mì',
    'tea': 'trà', 'coffee': 'cà phê', 'milk': 'sữa', 'money': 'tiền',
    'house': 'ngôi nhà', 'home': 'nhà', 'school': 'trường học', 'hospital': 'bệnh viện',
    'car': 'xe hơi', 'bus': 'xe buýt', 'book': 'cuốn sách', 'pen': 'cây bút',
    'phone': 'điện thoại', 'time': 'thời gian', 'today': 'hôm nay', 'tomorrow': 'ngày mai',

    # --- 8. MÀU SẮC ---
    'red': 'màu đỏ', 'blue': 'màu xanh dương', 'green': 'màu xanh lá',
    'yellow': 'màu vàng', 'white': 'màu trắng', 'purple': 'màu tím', 'brown': 'màu nâu',

    # --- 9. TỪ TIẾNG VIỆT KHÔNG DẤU ĐÁNH VẦN FINGERSPELLING ---
    'chao': 'xin chào', 'xinchao': 'xin chào', 'camon': 'cảm ơn', 'yeu': 'yêu',
    'khong': 'không', 'dung': 'đúng', 'sai': 'sai', 'tam biet': 'tạm biệt', 'tambiet': 'tạm biệt'
}

JA_DICTIONARY = {
    'apple': 'リンゴ', 'shirt': 'シャツ', 'hello': 'こんにちは', 'hi': 'こんにちは',
    'help': '助けて', 'yes': 'はい', 'no': 'いいえ', 'cousin': 'いとこ',
    'brother': '兄弟', 'mother': '母', 'father': '父', 'thanks': 'ありがとう',
    'thankyou': 'ありがとうございます', 'bye': 'さようなら', 'goodbye': 'さようなら',
    'love': '愛しています', 'water': '水', 'eat': '食べる', 'drink': '飲む',
    'happy': '嬉しい', 'sad': '悲しい', 'friend': '友達', 'school': '学校'
}

KO_DICTIONARY = {
    'apple': '사과', 'shirt': '셔츠', 'hello': '안녕하세요', 'hi': '안녕',
    'help': '도와주세요', 'yes': '네', 'no': '아니오', 'cousin': '사촌',
    'brother': '형제', 'mother': '어머니', 'father': '아버지', 'thanks': '감사합니다',
    'thankyou': '감사합니다', 'bye': '안녕히 가세요', 'goodbye': '안녕히 가세요',
    'love': '사랑해요', 'water': '물', 'eat': '먹다', 'drink': '마시다',
    'happy': '행복해요', 'sad': '슬퍼요', 'friend': '친구', 'school': '학교'
}


from collections import deque

class ContextAgent:
    def __init__(self, llm_corrector=None):
        import config
        self.llm_corrector = llm_corrector
        self.history = deque(maxlen=getattr(config, 'CONTEXT_WINDOW_SIZE', 7))

    def _merge_fingerspelling(self, raw_tokens):
        """
        Tự động ghép các chữ cái rời (fingerspelling) thành từ hoàn chỉnh và xử lý ngắt chữ:
        - 'space' -> tạo khoảng trắng ngăn cách giữa các từ
        - 'del'   -> xóa ký tự trước đó
        - ['H', 'E', 'L', 'L', 'O', 'space', 'W', 'O', 'R', 'L', 'D'] -> ['hello', 'world']
        - ['A', 'P', 'P', 'L', 'E', 'mother'] -> ['apple', 'mother']
        """
        merged_tokens = []
        char_buf = []

        for token in raw_tokens:
            token_clean = token.strip()
            tok_lower = token_clean.lower()

            if tok_lower == 'space':
                if char_buf:
                    merged_tokens.append("".join(char_buf).lower())
                    char_buf = []
            elif tok_lower == 'del':
                if char_buf:
                    char_buf.pop()
                elif merged_tokens:
                    merged_tokens.pop()
            elif tok_lower in ('nothing', ''):
                continue
            elif len(token_clean) == 1 and token_clean.isalpha():
                char_buf.append(token_clean)
            else:
                if char_buf:
                    merged_tokens.append("".join(char_buf).lower())
                    char_buf = []
                if token_clean:
                    merged_tokens.append(token_clean.lower())

        if char_buf:
            merged_tokens.append("".join(char_buf).lower())

        return merged_tokens

    def process(self, action_word, target_lang='vi'):
        if not action_word:
            return ""

        import config

        # 1. Nếu đầu vào là một từ vựng đơn lẻ có sẵn trong từ điển (ví dụ: "apple", "mother", "bye")
        dictionaries = {'vi': VI_DICTIONARY, 'ja': JA_DICTIONARY, 'ko': KO_DICTIONARY}
        dict_obj = dictionaries.get(target_lang)
        
        clean_word = action_word.strip().lower()
        if dict_obj and clean_word in dict_obj:
            translated = dict_obj[clean_word]
            self.history.append(translated)
            return translated
        if target_lang == 'en' and " " not in clean_word:
            self.history.append(clean_word)
            return clean_word

        # 2. Xử lý chuỗi nhiều từ hoặc chữ cái đánh vần (Fingerspelling)
        raw_tokens = action_word.split()
        words = self._merge_fingerspelling(raw_tokens)
        translated_words = []

        for w in words:
            if target_lang == 'vi':
                # Nếu là từ vựng trong từ điển
                if w in VI_DICTIONARY:
                    translated_words.append(VI_DICTIONARY[w])
                elif len(w) == 1:
                    translated_words.append(w.upper())
                else:
                    translated_words.append(w)
            elif target_lang == 'ja':
                if w in JA_DICTIONARY:
                    translated_words.append(JA_DICTIONARY[w])
                elif len(w) == 1:
                    translated_words.append(w.upper())
                else:
                    translated_words.append(w)
            elif target_lang == 'ko':
                if w in KO_DICTIONARY:
                    translated_words.append(KO_DICTIONARY[w])
                elif len(w) == 1:
                    translated_words.append(w.upper())
                else:
                    translated_words.append(w)
            else:
                translated_words.append(w.upper() if len(w) == 1 else w)

        final_str = " ".join(translated_words)
        
        # Tạo chuỗi history
        history_context = " | ".join(self.history) if self.history else None

        # Chỉ gọi LLM khi API Key hợp lệ (không phải key mẫu "AIzaSy...")
        api_key = getattr(config, 'LLM_API_KEY', '')
        if self.llm_corrector and hasattr(self.llm_corrector, 'correct') and api_key and not api_key.startswith("AIzaSy..."):
            try:
                llm_out = self.llm_corrector.correct(final_str, history_context=history_context)
                if llm_out:
                    self.history.append(llm_out)
                    return llm_out
            except Exception:
                pass

        if final_str:
            self.history.append(final_str)
        return final_str # Trả về bản dịch mượt mà & chính xác!
