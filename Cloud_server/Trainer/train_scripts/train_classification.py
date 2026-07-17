import os
import sys
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image
import pandas as pd
from tqdm import tqdm

# Cấu hình đường dẫn (Lùi ra 3 cấp để về thư mục gốc chứa config.py)
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
import config

# ==========================================
# 1. BỘ ĐỌC DỮ LIỆU ĐA NHÃN (CUSTOM DATASET)
# ==========================================
class MultiLabelHandDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        self.df = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform
        
        self.tasks = ['gender', 'skinColor', 'accessories', 'nailPolish', 'aspectOfHand']
        
        self.label_encoders = {}
        self.num_classes_dict = {}
        
        for task in self.tasks:
            unique_vals = sorted(self.df[task].astype(str).unique().tolist())
            self.label_encoders[task] = {val: idx for idx, val in enumerate(unique_vals)}
            self.num_classes_dict[task] = len(unique_vals)
            
        os.makedirs(config.SHARED_ASSETS_DIR, exist_ok=True)
        classes_json_path = os.path.join(config.SHARED_ASSETS_DIR, 'classification_classes.json')
        with open(classes_json_path, 'w', encoding='utf-8') as f:
            json.dump(self.label_encoders, f, ensure_ascii=False, indent=4)
            
    def __len__(self):
        return len(self.df)
        
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.img_dir, row['imageName'])
        
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
            
        labels = {}
        for task in self.tasks:
            val_str = str(row[task])
            labels[task] = torch.tensor(self.label_encoders[task][val_str], dtype=torch.long)
            
        return image, labels

# ==========================================
# 2. KIẾN TRÚC MẠNG ĐA ĐẦU (ĐÃ THÊM EFFICIENTNET)
# ==========================================
class MultiHeadModel(nn.Module):
    def __init__(self, num_classes_dict):
        super(MultiHeadModel, self).__init__()
        
        # Sử dụng cố định EfficientNet-B0 vì phù hợp và tối ưu nhất cho thiết bị di động
        print("=> Xương sống: EfficientNet-B0 ( trích suất đặc trưng và Đồng bộ dự án, cực nhẹ cho Mobile)")
        backbone = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        self.features = backbone.features
        self.pool = nn.AdaptiveAvgPool2d(1)
        in_features = 1280

        self.heads = nn.ModuleDict()
        for task_name, num_classes in num_classes_dict.items():
            self.heads[task_name] = nn.Sequential(
                nn.Flatten(),
                nn.Linear(in_features, 256),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(256, num_classes)
            )
            
    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        outputs = {}
        for task_name, head in self.heads.items():
            outputs[task_name] = head(x)
        return outputs

# ==========================================
# 3. VÒNG LẶP HUẤN LUYỆN (CÓ EARLY STOPPING & HỌC LẠI)
# ==========================================
def train_model():
    print("=== Khởi chạy Train Classification ===")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Thiết bị: {device}")
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    train_dataset = MultiLabelHandDataset(config.CLASSIFICATION_TRAIN_CSV, config.CLASSIFICATION_IMAGES_DIR, transform)
    val_dataset = MultiLabelHandDataset(config.CLASSIFICATION_VAL_CSV, config.CLASSIFICATION_IMAGES_DIR, transform)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    # --- CẤU HÌNH DỪNG SỚM & ĐIỀU KIỆN ---
    EPOCHS = 100          # Cho số Epoch lớn lên vì có Early Stopping cản lại rồi
    PATIENCE_LIMIT = 5    # Số lần liên tiếp không đổi thì dừng
    TARGET_ACCURACY = 90.0 # Bắt buộc phải trên 90%
    
    attempt = 1
    while True: # Vòng lặp HỌC LẠI
        print(f"\n[***] LẦN THỬ THỨ {attempt} [***]")
        model = MultiHeadModel(train_dataset.num_classes_dict).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        
        best_val_acc = 0.0
        patience_counter = 0
        os.makedirs(config.SHARED_ASSETS_DIR, exist_ok=True)
        model_save_path = os.path.join(config.SHARED_ASSETS_DIR, 'multi_head_best.pth')
        
        # --- THÊM CHỨC NĂNG LƯU & KHÔI PHỤC (RESUME) ---
        save_period = 5
        start_epoch = 0
        checkpoint_path = os.path.join(config.SHARED_ASSETS_DIR, 'classification_last.pth')
        
        if os.path.exists(checkpoint_path):
            print("[*] Đang khôi phục quá trình train bị dừng...")
            checkpoint = torch.load(checkpoint_path)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            best_val_acc = checkpoint['best_val_acc']
            patience_counter = checkpoint['patience_counter']
            print(f"[*] Tiếp tục train từ Epoch {start_epoch + 1}")
            
        for epoch in range(start_epoch, EPOCHS):
            # --- TRAIN ---
            model.train()
            running_loss = 0.0
            
            for images, labels_dict in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]"):
                images = images.to(device)
                for task in labels_dict:
                    labels_dict[task] = labels_dict[task].to(device)
                    
                optimizer.zero_grad()
                outputs = model(images)
                
                total_loss = sum([criterion(outputs[task], labels_dict[task]) for task in outputs])
                total_loss.backward()
                optimizer.step()
                running_loss += total_loss.item()
                
            train_loss = running_loss / len(train_loader)
            
            # --- VALIDATION (Đánh giá) ---
            model.eval()
            val_loss = 0.0
            correct_preds = 0
            total_preds = 0
            
            with torch.no_grad():
                for images, labels_dict in val_loader:
                    images = images.to(device)
                    for task in labels_dict:
                        labels_dict[task] = labels_dict[task].to(device)
                        
                    outputs = model(images)
                    
                    batch_loss = sum([criterion(outputs[task], labels_dict[task]) for task in outputs])
                    val_loss += batch_loss.item()
                    
                    # Tính tổng số dự đoán đúng trên tất cả các nhãn để làm cơ sở tính Accuracy
                    for task in outputs:
                        _, preds = torch.max(outputs[task], 1)
                        correct_preds += torch.sum(preds == labels_dict[task]).item()
                        total_preds += labels_dict[task].size(0)
                        
            val_loss = val_loss / len(val_loader)
            val_acc = (correct_preds / total_preds) * 100.0
            
            print(f"Epoch {epoch+1}: Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
            
            # --- EARLY STOPPING LOGIC ---
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
                torch.save(model.state_dict(), model_save_path)
                print(f"  -> (+) Lưu mô hình tốt nhất! (Acc: {best_val_acc:.2f}%)")
            else:
                patience_counter += 1
                print(f"  -> (-) Không cải thiện. Lần cảnh báo: {patience_counter}/{PATIENCE_LIMIT}")
                
            # --- LƯU LẠI TIẾN TRÌNH VÀ SAVE PERIOD ---
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_acc': best_val_acc,
                'patience_counter': patience_counter
            }, checkpoint_path)
            
            if (epoch + 1) % save_period == 0:
                periodic_path = os.path.join(config.SHARED_ASSETS_DIR, f'classification_epoch_{epoch+1}.pth')
                torch.save(model.state_dict(), periodic_path)
                print(f"  -> Đã lưu mô hình định kỳ tại Epoch {epoch+1}")
                
            if patience_counter >= PATIENCE_LIMIT:
                print(f"\n[!] EARLY STOPPING: Dừng sớm tại Epoch {epoch+1} vì kết quả không đổi sau {PATIENCE_LIMIT} lần.")
                break # Thoát khỏi vòng lặp Epoch hiện tại
                
        # --- KIỂM TRA ĐIỀU KIỆN SAU KHI DỪNG ---
        print(f"\nKết thúc lần thử {attempt}. Độ chính xác Max: {best_val_acc:.2f}%")
        
        # Xóa file checkpoint tiến trình và rác lưu định kỳ
        import glob
        if os.path.exists(checkpoint_path):
            os.remove(checkpoint_path)
        for f in glob.glob(os.path.join(config.SHARED_ASSETS_DIR, 'classification_epoch_*.pth')):
            os.remove(f)
            
        if best_val_acc >= TARGET_ACCURACY:
            print(f"🎉 THÀNH CÔNG! Độ chính xác {best_val_acc:.2f}% đạt tiêu chuẩn >= {TARGET_ACCURACY}%.")
            print(f"Đã chốt mô hình tại: {model_save_path}")
            break # Thoát hoàn toàn vòng lặp While, kết thúc chương trình
        else:
            print(f"⚠️ THẤT BẠI: {best_val_acc:.2f}% < {TARGET_ACCURACY}%. Sẽ xóa não AI và học lại từ đầu...")
            attempt += 1

if __name__ == "__main__":
    train_model()
