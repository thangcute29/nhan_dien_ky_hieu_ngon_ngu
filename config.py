# config.py
import os
import sys

# Đảm bảo console Windows có thể in được tiếng Việt (tránh lỗi UnicodeEncodeError)
if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except (OSError, ValueError):
        pass

# --- CẤU HÌNH ĐƯỜNG DẪN THƯ MỤC GỐC ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = BASE_DIR

# Đường dẫn tới khu vực dữ liệu nghiên cứu
RESEARCH_DATA_DIR = os.path.join(PROJECT_ROOT, 'Research_and_Data', 'Dataset')


# ----------------- CLASSIFICATION (Multi-Label) -----------------
CLASSIFICATION_DIR = os.path.join(RESEARCH_DATA_DIR, 'Classification')
CLASSIFICATION_IMAGES_DIR = os.path.join(CLASSIFICATION_DIR, 'images')
CLASSIFICATION_TRAIN_CSV = os.path.join(CLASSIFICATION_DIR, 'train.csv')
CLASSIFICATION_VAL_CSV = os.path.join(CLASSIFICATION_DIR, 'val.csv')

# ----------------- DETECTION (YOLO Hand Detection) -----------------
DETECTION_DIR = os.path.join(RESEARCH_DATA_DIR, 'Detection')

# ----------------- FEATURES (Dynamic Auto-Detect) -----------------
FEATURES_DIR = os.path.join(RESEARCH_DATA_DIR, 'Features')
if os.path.exists(os.path.join(FEATURES_DIR, 'train')):
    FEATURES_RAW_DIR = os.path.join(FEATURES_DIR, 'train')
else:
    subdirs = [os.path.join(FEATURES_DIR, d) for d in os.listdir(FEATURES_DIR) if os.path.isdir(os.path.join(FEATURES_DIR, d))] if os.path.exists(FEATURES_DIR) else []
    FEATURES_RAW_DIR = subdirs[0] if subdirs else os.path.join(FEATURES_DIR, 'asl-alphabet-train')

# ----------------- SEQUENCES (Universal Directory) -----------------
SEQUENCES_DIR = os.path.join(RESEARCH_DATA_DIR, 'Sequences')
WLASL_VIDEOS_DIR = os.path.join(SEQUENCES_DIR, 'videos')
WLASL_METADATA_PATH = os.path.join(SEQUENCES_DIR, 'nslt_1000.json')
WLASL_CLASS_LIST_PATH = os.path.join(SEQUENCES_DIR, 'wlasl_class_list.txt')
# Giữ nguyên cây ``processed`` cũ để đối chiếu; không trộn nhãn lỗi với WLASL.
SEQUENCES_PROCESSED_DIR = os.path.join(SEQUENCES_DIR, 'processed_wlasl1000')
SEQUENCES_CUSTOM_DIR = os.path.join(SEQUENCES_DIR, 'custom_enrollment')   # Thư mục lưu video mẫu nạp mới của Gia sư AI

#              AI AGENT CONFIG
# ----------------- LLM CONFIG -----------------
# Khai báo nhà cung cấp dịch vụ mà ứng dụng sẽ sử dụng để gọi API LLM (ví dụ: "openai", "azure", "google", "auto")
LLM_PROVIDER = "google"          # Ưu tiên dùng Gemini của Google; nếu Gemini không khả dụng thì sẽ thử Ollama local nếu có
# Dán mã API Key lấy từ Google AI Studio (hãy đảm bảo giữ bí mật và không commit vào Git)
LLM_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_API_KEY = LLM_API_KEY # Alias cho công cụ mới
#Chỉ định mô hình sẽ thực hiện công việc sửa lỗi văn bản (ví dụ: "gpt-3.5-turbo", "gemini-1.5-flash", "gemini-2.0-pro")
LLM_MODEL = "gemini-2.5-flash"
USE_LLM_CORRECTION = True
#Số lượng câu gần nhất (context window) được gửi đến LLM để sửa lỗi. Với Gemini, em có thể tăng lên 7-10 câu mà vẫn chạy rất nhanh, giúp cải thiện độ chính xác sửa lỗi bằng cách cung cấp nhiều ngữ cảnh hơn.
CONTEXT_WINDOW_SIZE = 7          # Với Gemini, em có thể tăng lên 7-10 câu mà vẫn chạy rất nhanh                

# --- CẤU HÌNH THỜI GIAN THỰC (REAL-TIME CONFIG) ---
AI_FPS = 10
PROCESS_INTERVAL = 1.0 / AI_FPS  # 0.1 giây giữa mỗi lần xử lý AI
BUFFER_THRESHOLD = 1            # Đã hạ từ 3 xuống 1 ký tự cho từ tiếng Anh ngắn (I, He, Is)
COOLDOWN_TIME = 0.3             # Khoảng lặng 300ms để đóng gói cụm danh từ
USE_YOLO_HAND_CROPS = False     # WLASL GRU dùng MediaPipe toàn khung; bật crop sẽ lệch preprocessing.

# --- CẤU HÌNH NGÔN NGỮ (LOCALIZATION CONFIG) ---
TARGET_LANGUAGE = 'vi'          # Ngôn ngữ đích hiển thị cho người dùng là Tiếng Việt

# --- HỆ THỐNG ĐƯỜNG DẪN THƯ MỤC (PATH CONFIG) ---
# Đường dẫn bộ Dataset dùng chung tại Local (App/Web mượn đọc)
SHARED_LIB_DIR = os.path.join(BASE_DIR, "Shared_lib")
SHARED_ASSETS_DIR = os.path.join(SHARED_LIB_DIR, "Assets")

# Đường dẫn kho dữ liệu trên Cloud Server
CLOUD_DB_DIR = os.path.join(BASE_DIR, "Cloud_server", "Database")
EDGE_CASES_DIR = os.path.join(CLOUD_DB_DIR, "edge_cases")

# 3 Phân khu chính của kho dữ liệu lỗi để học máy
PENDING_DIR = os.path.join(EDGE_CASES_DIR, "Pending")
LABELED_DIR = os.path.join(EDGE_CASES_DIR, "Labeled")
UNVERIFIED_LEARNING_DIR = os.path.join(EDGE_CASES_DIR, "Unverified_Learning")
