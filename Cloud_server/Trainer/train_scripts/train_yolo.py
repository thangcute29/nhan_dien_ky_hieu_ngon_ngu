"""
Huấn luyện YOLOv8-nano để phát hiện bàn tay.
Sử dụng dữ liệu từ data/detection/data.yaml
"""
import os
import sys
import torch
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
import config
from ultralytics import YOLO

def main():
    print("=== Huấn luyện YOLOv8 phát hiện bàn tay ===")
    
    # Đường dẫn file data.yaml
    data_yaml = os.path.join(config.DETECTION_DIR, 'data.yaml')
    if not os.path.exists(data_yaml):
        print(f"❌ Không tìm thấy {data_yaml}")
        return False

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
        
    # SỬA LỖI: Dùng torch.cuda để kiểm tra GPU an toàn trên Windows
    device_type = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print(f"🖥️ Đang sử dụng thiết bị huấn luyện: {device_type.upper()}")
    
    # Huấn luyện
    results = model.train(
        data=data_yaml,
        epochs=30,
        imgsz=640,
        batch=16,
        device=device_type,
        workers=2,
        cache=False,
        project=os.path.join(config.PROJECT_ROOT, "Cloud_server", "Trainer", "runs"),
        name="yolo_hands",
        exist_ok=True,
        patience=5,
        save_period=5,
        resume=resume_flag
    )
    
    best_map50 = results.box.map50
    print(f"\n[AI YOLO] Độ chính xác cao nhất đạt được (mAP50): {best_map50 * 100:.2f}%")
    
    if best_map50 >= 0.90:
        best_path = os.path.join(config.PROJECT_ROOT, 'Cloud_server', 'Trainer', 'runs', 'yolo_hands', 'weights', 'best.pt')
        dest_path = os.path.join(config.SHARED_ASSETS_DIR, 'hand_det_yolo.pt')
        
        if os.path.exists(best_path):
            import shutil
            os.makedirs(os.path.dirname(dest_path), exist_ok=True)
            shutil.copy(best_path, dest_path)
            print(f"✅ Kết quả đạt chuẩn chất lượng! Model gốc đã lưu tại Shared assets: {dest_path}")

            # Only remove resumable checkpoints after a model passes the gate.
            import glob
            for checkpoint in glob.glob(os.path.join(weights_dir, "epoch*.pt")):
                os.remove(checkpoint)
            if os.path.exists(last_ckpt):
                os.remove(last_ckpt)
            
        return True
    else:
        print(f"❌ Kết quả huấn luyện THẤT BẠI ({best_map50 * 100:.2f}% < 90%). Hủy bỏ để học lại.")
        return False
    
if __name__ == '__main__':
    main()
