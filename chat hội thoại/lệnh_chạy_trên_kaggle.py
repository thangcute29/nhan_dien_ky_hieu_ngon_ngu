"""
HƯỚNG DẪN ĐƯA TOÀN BỘ PROJECT LÊN KAGGLE ĐỂ TRAIN BẰNG GPU MIỄN PHÍ P100/T4x2

Kaggle có ưu điểm tuyệt vời là bạn có thể upload nguyên cả thư mục code dưới dạng "Kaggle Dataset", 
sau đó chạy thẳng các file Python bằng Bash lệnh (!python ...) cực kỳ dễ dàng.

BƯỚC 1: CHUẨN BỊ Ở MÁY CÁ NHÂN (LOCAL)
1. Gom toàn bộ thư mục 'Sign_language' (không cần file npy cũ nếu bạn định trích xuất lại trên Kaggle).
2. Nén thành file ZIP (VD: Sign_language_project.zip)

BƯỚC 2: UPLOAD LÊN KAGGLE
1. Vào Kaggle -> Datasets -> New Dataset -> Upload file ZIP lên. Đặt tên là "sign_language_ngon_ngu_ky_hieu".
2. Vào Kaggle -> Notebooks -> New Notebook. 
3. Bật GPU: Ở menu góc phải (Session options) -> Accelerator -> Chọn GPU T4 x2 hoặc P100.
4. Ở cột phải, mục Data -> Add Data -> Your Datasets -> Chọn cái dataset "sign_language_ngon_ngu_ky_hieu" vừa up.

BƯỚC 3: DƯỚI ĐÂY LÀ CÁC LỆNH (CELLS) BẠN CẦN COPY VÀ CHẠY TRÊN KAGGLE NOTEBOOK
"""

# ==============================================================================
# CELL 1: Copy dữ liệu từ thư mục Read-only sang thư mục Working
# (Kaggle Dataset mặc định nằm ở thư mục /kaggle/input/ và chỉ ĐỌC. 
# Do đó ta phải copy qua /kaggle/working/ để Code có thể tạo ra file checkpoint, file npy...)
# ==============================================================================
import os
import shutil

# 1. Đường dẫn Dataset của bạn trên Kaggle
SOURCE_DIR = '/kaggle/input/sign_language_ngon_ngu_ky_hieu'
WORKING_DIR = '/kaggle/working/SIGN_LANGUAGE'

# Dự phòng nếu Kaggle tự động đổi gạch dưới _ thành gạch ngang -
if not os.path.exists(SOURCE_DIR):
    input_folders = os.listdir('/kaggle/input')
    if input_folders:
        SOURCE_DIR = os.path.join('/kaggle/input', input_folders[0])

print(f"🎯 Nguồn dữ liệu gốc: {SOURCE_DIR}")

# 2. Xóa thư mục tạm cũ (nếu có) và tạo mới
if os.path.exists(WORKING_DIR):
    shutil.rmtree(WORKING_DIR)
os.makedirs(WORKING_DIR, exist_ok=True)

# 3. CHIẾN THUẬT COPY THÔNG MINH
# A. Các thư mục Code (nhẹ) thì COPY THẬT để có thể chỉnh sửa/chạy được
CODE_WHITELIST = [
    'Cloud_server', 'Data_preparation', 'Shared_lib', 'Tools', 
    'Demo_ui', 'config.py', 'Retrain.py', 'Requirments.txt', 'requirements.txt'
]

print("🔄 Đang copy source code...")
for item in CODE_WHITELIST:
    src = os.path.join(SOURCE_DIR, item)
    dst = os.path.join(WORKING_DIR, item)
    if os.path.exists(src):
        if os.path.isdir(src): shutil.copytree(src, dst)
        else: shutil.copy2(src, dst)

# B. Các thư mục Dữ liệu/Model nặng thì TẠO SHORTCUT ẢO (Symlink) 
# -> Dung lượng = 0 MB, Thời gian copy = 0 giây, nhưng Code vẫn đọc được bình thường!
SYMLINK_LIST = ['Research_and_Data', 'YOLOv8', 'yolov8n.pt']

print("🔗 Đang tạo Shortcut (Symlink) cho thư mục Dữ liệu siêu nặng...")
for item in SYMLINK_LIST:
    src = os.path.join(SOURCE_DIR, item)
    dst = os.path.join(WORKING_DIR, item)
    if os.path.exists(src):
        os.symlink(src, dst)

# 4. Chuyển vị trí làm việc của Python vào gốc dự án mới
os.chdir(WORKING_DIR)
print("✅ Hoàn tất! Sẵn sàng chiến đấu.")
print("📍 Thư mục làm việc hiện tại:", os.getcwd())


# ==============================================================================
# CELL 2: Cài đặt các thư viện cần thiết (Kaggle đã có sẵn TF, chỉ cần cài thêm mediapipe)
# (Chạy lệnh shell bằng dấu chấm than ! ở đầu)
# ==============================================================================
# !pip install mediapipe opencv-python tensorflow==2.15.0


# ==============================================================================
# CELL 3: TRÍCH XUẤT KEYPOINTS (Nếu bạn chưa làm ở Local)
# ==============================================================================
# (Kaggle chạy lệnh python nguyên bản nhờ config.py của bạn tự động định tuyến đường dẫn hoàn hảo)
# !python Data_preparation/Prepare_sequences.py


# ==============================================================================
# CELL 4: HUẤN LUYỆN GRU VỚI GPU CỦA KAGGLE
# ==============================================================================
# !python Cloud_server/Trainer/train_scripts/train_gru.py


# ==============================================================================
# CELL 5: TẢI MODEL VỀ SAU KHI TRAIN XONG
# ==============================================================================
"""
Sau khi Cell 4 chạy xong 200 epochs và lưu mô hình tốt nhất vào thư mục:
/kaggle/working/Sign_language/Cloud_server/Trainer/runs/gru_checkpoints/...

Bạn nhìn sang cột bên Phải của Kaggle, mục "Output" (/kaggle/working/). 
Bạn sẽ thấy thư mục Sign_language. Bấm mở ra tìm file mô hình .h5, 
nhấn dấu 3 chấm -> Download là tải về máy!
"""
