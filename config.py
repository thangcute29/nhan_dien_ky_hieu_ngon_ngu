# config.py
import os
import sys
import io

# Đảm bảo console Windows có thể in được tiếng Việt (tránh lỗi UnicodeEncodeError)
if hasattr(sys.stdout, 'buffer'):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# --- CẤU HÌNH ĐƯỜNG DẪN THƯ MỤC GỐC ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = BASE_DIR

# Đường dẫn tới khu vực dữ liệu nghiên cứu
RESEARCH_DATA_DIR = os.path.join(PROJECT_ROOT, 'Research_and_Data', 'Dataset')

# ----------------- DETECTION -----------------
DETECTION_DIR = os.path.join(RESEARCH_DATA_DIR, 'Detection')
DETECTION_IMAGES_DIR = os.path.join(DETECTION_DIR, 'Hands', 'Hands')   # Thư mục ảnh
DETECTION_CSV = os.path.join(DETECTION_DIR, 'HandInfo.csv')             # File CSV thông tin ảnh Detection

# ----------------- CLASSIFICATION (Multi-Label) -----------------
CLASSIFICATION_DIR = os.path.join(RESEARCH_DATA_DIR, 'Classification')
CLASSIFICATION_IMAGES_DIR = os.path.join(CLASSIFICATION_DIR, 'images')
CLASSIFICATION_TRAIN_CSV = os.path.join(CLASSIFICATION_DIR, 'train.csv')
CLASSIFICATION_VAL_CSV = os.path.join(CLASSIFICATION_DIR, 'val.csv')

# ----------------- FEATURES -----------------
FEATURES_DIR = os.path.join(RESEARCH_DATA_DIR, 'Features')
FEATURES_RAW_SUBDIR = 'asl-alphabet-train'
FEATURES_RAW_DIR = os.path.join(FEATURES_DIR, FEATURES_RAW_SUBDIR)

# ----------------- SEQUENCES -----------------
SEQUENCES_DIR = os.path.join(RESEARCH_DATA_DIR, 'Sequences')
# Thư mục gốc chứa video phân loại theo lớp từ vựng (apple, book, cat,...)
SEQUENCES_DATASET_DIR = os.path.join(SEQUENCES_DIR, 'archive', 'dataset', 'SL')
# Thư mục chứa các video thô theo mã ID số (00335.mp4, 00336.mp4,...)
SEQUENCES_VIDEOS_DIR = os.path.join(SEQUENCES_DIR, 'videos')
# Dữ liệu CSV bổ sung (nếu có)
SEQUENCES_CSV = os.path.join(SEQUENCES_DIR, 'hand_gestures.csv')

# ----------------- EDGE CASES -----------------
EDGE_CASES_DIR = os.path.join(RESEARCH_DATA_DIR, '..', 'edge_cases')  # Nằm cùng cấp Dataset
EDGE_CASES_PENDING = os.path.join(EDGE_CASES_DIR, 'pending')
EDGE_CASES_LABELED = os.path.join(EDGE_CASES_DIR, 'labeled')



#              AI AGENT CONFIG
# ----------------- LLM CONFIG -----------------
# Khai báo nhà cung cấp dịch vụ mà ứng dụng sẽ sử dụng để gọi API LLM (ví dụ: "openai", "azure", "google", "auto")
LLM_PROVIDER = "google"          # Ưu tiên dùng Gemini của Google; nếu Gemini không khả dụng thì sẽ thử Ollama local nếu có
# Dán mã API Key lấy từ Google AI Studio (hãy đảm bảo giữ bí mật và không commit vào Git)
LLM_API_KEY = "AIzaSy..."        # Dán mã API Key lấy từ Google AI Studio 
#Chỉ định mô hình sẽ thực hiện công việc sửa lỗi văn bản (ví dụ: "gpt-3.5-turbo", "gemini-1.5-flash", "gemini-2.0-pro")
LLM_MODEL = "gemini-1.5-flash"   # Sử dụng bản Flash để tốc độ phản hồi nhanh nhất cho bài nhận diện
#Xác định URL cho thư viện local llm (ollama,v.v) nếu không sử dụng dịch vụ đám mây (để gọi API local), nếu để None sẽ dùng URL mặc định của Ollama
LLM_BASE_URL = None              # Gemini dùng thư viện riêng hoặc URL mặc định của Google, nên để None
#Đây là cờ bật/tắt tính năng sửa lỗi văn bản bằng LLM. Nếu False, phần sửa lỗi sẽ bị bỏ qua và trả về văn bản gốc (có thể dùng để so sánh hiệu quả sửa lỗi)
USE_LLM_CORRECTION = True
#Số lượng câu gần nhất (context window) được gửi đến LLM để sửa lỗi. Với Gemini, em có thể tăng lên 7-10 câu mà vẫn chạy rất nhanh, giúp cải thiện độ chính xác sửa lỗi bằng cách cung cấp nhiều ngữ cảnh hơn.
CONTEXT_WINDOW_SIZE = 7          # Với Gemini, em có thể tăng lên 7-10 câu mà vẫn chạy rất nhanh                

#=======================ngôn ngữ hỗ trợ dịch thuật========================
# config.py
TARGET_LANGUAGE = "vi"   # vi, en, ja, ko, zh, fr...

# --- CẤU HÌNH THỜI GIAN THỰC (REAL-TIME CONFIG) ---
AI_FPS = 10
PROCESS_INTERVAL = 1.0 / AI_FPS  # 0.1 giây giữa mỗi lần xử lý AI
BUFFER_THRESHOLD = 1            # Đã hạ từ 3 xuống 1 ký tự cho từ tiếng Anh ngắn (I, He, Is)
COOLDOWN_TIME = 0.3             # Khoảng lặng 300ms để đóng gói cụm danh từ

# --- CẤU HÌNH NGÔN NGỮ (LOCALIZATION CONFIG) ---
DEFAULT_PROVIDER = "google"      # Định hướng dùng Gemini làm mặc định
TARGET_LANGUAGE = 'vi'          # Ngôn ngữ đích hiển thị cho người dùng là Tiếng Việt
CONTEXT_WINDOW_SIZE = 7         # Giữ nguyên cửa sổ 7 câu gần nhất để AI Agent suy luận ngữ cảnh

# --- HỆ THỐNG ĐƯỜNG DẪN THƯ MỤC (PATH CONFIG) ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Đường dẫn bộ Dataset dùng chung tại Local (App/Web mượn đọc)
SHARED_LIB_DIR = os.path.join(BASE_DIR, "Shared_lib")
SHARED_ASSETS_DIR = os.path.join(SHARED_LIB_DIR, "assets")

# Đường dẫn kho dữ liệu trên Cloud Server
CLOUD_DB_DIR = os.path.join(BASE_DIR, "Cloud_server", "Database")
EDGE_CASES_DIR = os.path.join(CLOUD_DB_DIR, "edge_cases")

# 3 Phân khu chính của kho dữ liệu lỗi để học máy
PENDING_DIR = os.path.join(EDGE_CASES_DIR, "Pending")
LABELED_DIR = os.path.join(EDGE_CASES_DIR, "Labeled")
UNVERIFIED_LEARNING_DIR = os.path.join(EDGE_CASES_DIR, "Unverified_Learning")