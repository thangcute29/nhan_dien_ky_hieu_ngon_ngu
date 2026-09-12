import os
import sys
import shutil
import time
import re
from google import genai

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import config

# Cấu hình API Key
if not hasattr(config, 'GEMINI_API_KEY') or not config.GEMINI_API_KEY.strip():
    print("❌ Lỗi: Bạn chưa cấu hình GEMINI_API_KEY hợp lệ trong file config.py")
    sys.exit(1)

client = genai.Client(api_key=config.GEMINI_API_KEY)

# Đường dẫn thư mục
CUSTOM_ENROLL_DIR = os.path.join(BASE_DIR, "Research_and_Data", "Dataset", "Sequences", "custom_enrollment")
LONG_SEGMENTS_DIR = os.path.join(CUSTOM_ENROLL_DIR, "long_video_segments")
TRAIN_DIR = os.path.join(config.SEQUENCES_PROCESSED_DIR, "train")

def get_existing_vocabulary():
    """Lấy danh sách các từ vựng đã có trong dataset."""
    vocab = set()
    if os.path.exists(TRAIN_DIR):
        vocab.update([d for d in os.listdir(TRAIN_DIR) if os.path.isdir(os.path.join(TRAIN_DIR, d))])
    if os.path.exists(CUSTOM_ENROLL_DIR):
        vocab.update([d for d in os.listdir(CUSTOM_ENROLL_DIR) if os.path.isdir(os.path.join(CUSTOM_ENROLL_DIR, d)) and d != "long_video_segments"])
    return vocab

def identify_sign_with_gemini(video_path):
    """Gửi video lên Gemini 1.5 Flash/Pro để nhận diện từ vựng."""
    print(f"   ⏳ Đang tải video lên AI: {os.path.basename(video_path)}...")
    try:
        video_file = client.files.upload(file=video_path)
        
        # Chờ video xử lý xong trên hệ thống của Google
        while not video_file.state or video_file.state.name == "PROCESSING":
            time.sleep(1)
            video_file = client.files.get(name=video_file.name)
            
        if video_file.state.name == "FAILED":
            print(f"   ❌ Tải video lên thất bại.")
            return None

        # Gọi mô hình Gemini
        prompt = (
            "You are an expert in American Sign Language (ASL). "
            "Watch this short video and tell me what ONE word or short phrase the person is signing. "
            "Return ONLY the English word, capitalized, with no punctuation or extra text. "
            "Example output: Hello"
        )
        
        print(f"   🧠 AI đang xem và phân tích...")
        response = client.models.generate_content(
            model=config.LLM_MODEL,
            contents=[video_file, prompt],
        )
        
        # Xóa file trên Cloud sau khi dùng xong
        client.files.delete(name=video_file.name)
        
        raw_text = response.text.strip()
        
        # Làm sạch chuỗi trả về (Xóa các ký tự đặc biệt, thay khoảng trắng bằng _)
        clean_text = re.sub(r'[^a-zA-Z0-9\s]', '', raw_text)
        clean_text = clean_text.strip().replace(' ', '_').capitalize()
        
        return clean_text if clean_text else None

    except Exception as e:
        print(f"   ❌ Lỗi khi gọi Gemini API: {e}")
        return None

def run_auto_builder():
    print("=" * 70)
    print("🤖 BẮT ĐẦU QUY TRÌNH GẮN NHÃN & AUTO DATASET BUILDER (BẰNG GEMINI)")
    print("=" * 70)
    
    if not os.path.exists(LONG_SEGMENTS_DIR):
        print(f"❌ Không tìm thấy thư mục: {LONG_SEGMENTS_DIR}")
        return
        
    vocab = get_existing_vocabulary()
    print(f"📚 Kho từ vựng hiện tại đang có: {len(vocab)} từ.")
    
    video_folders = [f for f in os.listdir(LONG_SEGMENTS_DIR) if os.path.isdir(os.path.join(LONG_SEGMENTS_DIR, f))]
    if not video_folders:
        print("✅ Không có video cắt nào cần xử lý.")
        return
        
    total_processed = 0
    new_words = 0
    
    for vid_folder in video_folders:
        src_dir = os.path.join(LONG_SEGMENTS_DIR, vid_folder)
        segments = [f for f in os.listdir(src_dir) if f.endswith(('.mp4', '.avi'))]
        
        print(f"\n📂 Đang quét thư mục video: {vid_folder} ({len(segments)} segments)")
        
        for seg_file in segments:
            seg_path = os.path.join(src_dir, seg_file)
            print(f"\n▶️ Đang xử lý: {seg_file}")
            
            label = identify_sign_with_gemini(seg_path)
            
            if not label or len(label) > 30: # Bỏ qua nếu label trả về quá dài (có thể AI bị ảo giác)
                print(f"   ⚠️ Không thể nhận diện nhãn hợp lệ cho video này. Bỏ qua.")
                continue
                
            print(f"   🎯 Kết quả từ AI: [{label}]")

            approval = input(
                "   Xác nhận nhãn này? [Y] đồng ý, [N] bỏ qua, hoặc nhập nhãn đúng: "
            ).strip()
            if approval.lower() in {'n', 'no'}:
                print("   ⏭️ Đã bỏ qua; video gốc vẫn được giữ nguyên.")
                continue
            if approval and approval.lower() not in {'y', 'yes'}:
                corrected = re.sub(r'[^a-zA-Z0-9_-]', '', approval.strip().replace(' ', '_'))
                if not corrected:
                    print("   ⚠️ Nhãn sửa không hợp lệ. Bỏ qua.")
                    continue
                label = corrected.lower()
            
            # Tạo thư mục đích
            dst_dir = os.path.join(CUSTOM_ENROLL_DIR, label)
            os.makedirs(dst_dir, exist_ok=True)
            
            if label not in vocab:
                vocab.add(label)
                new_words += 1
                print(f"   🌟 Đã phát hiện TỪ MỚI: {label}")
                
            # Đổi tên file để tránh trùng lặp
            new_filename = f"{label}_{int(time.time())}_{seg_file}"
            dst_path = os.path.join(dst_dir, new_filename)
            
            shutil.move(seg_path, dst_path)
            total_processed += 1
            print(f"   ✅ Đã phân loại vào: custom_enrollment/{label}/{new_filename}")
            
        # Dọn dẹp thư mục trống
        if not os.listdir(src_dir):
            os.rmdir(src_dir)
            print(f"🗑️ Đã xóa thư mục trống: {vid_folder}")
            
    print("\n" + "=" * 70)
    print(f"🎉 HOÀN TẤT! Đã xử lý {total_processed} đoạn video. Tìm thấy {new_words} từ vựng mới!")
    print("=" * 70)
    
    if total_processed > 0:
        print("\n✅ Các nhãn đã được người dùng xác nhận. Hãy kiểm tra lại dữ liệu rồi chạy:")
        print("   python Cloud_server/Trainer/Retrain.py")

if __name__ == "__main__":
    run_auto_builder()
