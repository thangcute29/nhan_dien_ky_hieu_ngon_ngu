# cloud_server/trainer/train_scripts/train_yolo.py
#train cho detection bàn tay bằng YOLOv8
"""
Huấn luyện YOLOv8-nano để phát hiện bàn tay.
Sử dụng dữ liệu từ data/detection/data.yaml
"""
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
import config
from ultralytics import YOLO

def main():
    print("=== Huấn luyện YOLOv8 phát hiện bàn tay ===")
    
    # Đường dẫn file data.yaml
    data_yaml = os.path.join(config.DETECTION_DIR, 'data.yaml')
    if not os.path.exists(data_yaml):
        print(f"❌ Không tìm thấy {data_yaml}")
        return False # Trả về False nếu lỗi không tìm thấy file dữ liệu

    # Kiểm tra checkpoint để resume
    weights_dir = os.path.join(config.PROJECT_ROOT, "Cloud_server", "Trainer", "runs", "yolo_hands", "weights")
    last_ckpt = os.path.join(weights_dir, "last.pt")
    
    if os.path.exists(last_ckpt):
        print(f"🔄 Tìm thấy checkpoint '{last_ckpt}', đang resume...")
        model = YOLO(last_ckpt)
        resume_flag = True
    else:
        print("🚀 Không tìm thấy checkpoint, bắt đầu train từ đầu với yolov8n...")
        model = YOLO("yolov8n.pt")
        resume_flag = False
    
    # Huấn luyện
    results = model.train(
        data=data_yaml,
        epochs=30,
        imgsz=640,
        batch=16,
        device='cuda' if os.system('nvidia-smi') == 0 else 'cpu',
        workers=2,
        cache=False,
        # Dùng đường dẫn TUYỆT ĐỐI neo theo config.PROJECT_ROOT, không phụ thuộc
        # vào thư mục làm việc (cwd) lúc chạy script -> luôn khớp với convert_to_mobile.py
        project=os.path.join(config.PROJECT_ROOT, "Cloud_server", "Trainer", "runs"),
        name="yolo_hands",
        exist_ok=True,
        patience=5, #chốt chặn sớm nếu không cải thiện để tránh overfitting
        save_period=5,  #theo dõi tiến trình, so sánh kết quả ở các giai đoạn khác nhau , có thể khôi phục lại quá trình huấn luyện từ điểm gần nhất nếu bị gián đoạn, thay vì phải chạy lại từ đầu
        resume=resume_flag
    )
    
    # Dọn dẹp rác (xoá các file epoch*.pt và last.pt)
    import glob
    try:
        for f in glob.glob(os.path.join(weights_dir, "epoch*.pt")):
            os.remove(f)
        if os.path.exists(last_ckpt):
            os.remove(last_ckpt)
        print("🧹 Đã tự động dọn dẹp các file checkpoint tạm thời.")
    except Exception as e:
        print(f"⚠️ Lỗi khi dọn dẹp file tạm: {e}")
        
    # Đọc chỉ số mAP50 (Độ chính xác phát hiện box bàn tay) sau khi kết thúc huấn luyện
    best_map50 = results.box.map50
    print(f"\n[AI YOLO] Độ chính xác cao nhất đạt được (mAP50): {best_map50 * 100:.2f}%")
    
    # CHỐT CHẶN 2: CHẤT LƯỢNG ĐẦU RA PHẢI TRÊN 90%
    # ĐỘ CHÍNH XÁC CAO NHẤT ĐẠT ĐƯỢC >= 90%
    if best_map50 >= 0.90:
        # 1. Dùng os.path.join từ PROJECT_ROOT để tuyệt đối không bao giờ lỗi đường dẫn
        best_path = os.path.join(config.PROJECT_ROOT, 'Cloud_server', 'Trainer', 'runs', 'yolo_hands', 'weights', 'best.pt')
        dest_path = os.path.join(config.SHARED_ASSETS_DIR, 'hand_det_yolo.pt') #đường dẫn lưu file dùng chung cho Mobile App trong assets
        
        # 2. Tiến hành bốc file thô về khu vực tập kết của Server
        if os.path.exists(best_path):
            import shutil
            os.makedirs(os.path.dirname(dest_path), exist_ok=True)
            shutil.copy(best_path, dest_path)
            print(f"✅ Kết quả đạt chuẩn chất lượng! Model gốc đã lưu tại Shared assets: {dest_path}")
            
        return True # Báo hiệu cho Retrain.py kích hoạt tool nén sang .tflite
    else:
        print(f"❌ Kết quả huấn luyện THẤT BẠI ({best_map50 * 100:.2f}% < 90%). Hủy bỏ để học lại.")
        return False
    

if __name__ == "__main__":
    main()