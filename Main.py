# Main.py (Root Central Launcher)
"""
BẢNG ĐIỀU KHIỂN KHỞI ĐỘNG TRUNG TÂM DỰ ÁN NGÔN NGỮ KÝ HIỆU
Cho phép lựa chọn các chế độ chạy ứng dụng dễ dàng:
[1] Ứng dụng Tiêu chuẩn Core Mobile/Desktop (Mobile_app/Src/Main.py)
[2] Giao diện Phụ đề Netflix Điện ảnh Báo cáo (Demo_ui/App.py)
[3] Khởi động Máy chủ API Backend (Cloud_server/Api/Main.py)
"""
import sys
import os
import subprocess

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

def main():
    while True:
        print("\n=======================================================", flush=True)
        print(" 🚀 HỆ THỐNG NHẬN DIỆN NGÔN NGỮ KÝ HIỆU - CENTRAL LAUNCHER", flush=True)
        print("=======================================================", flush=True)
        print(" [1] Chạy Ứng dụng Tiêu chuẩn Core (Mobile_app/Src/Main.py)", flush=True)
        print(" [2] Chạy Giao diện Phụ đề Netflix Điện ảnh (Demo_ui/App.py)", flush=True)
        print(" [3] Khởi động Máy chủ Backend API Server (Cloud_server/Api/Main.py)", flush=True)
        print(" [Q] Thoát chương trình", flush=True)
        print("-------------------------------------------------------", flush=True)
        
        choice = input("👉 Nhập lựa chọn của bạn (1/2/3/Q): ").strip().lower()
        
        if choice == '1':
            from Mobile_app.Src.Main import SignLanguageApp
            app = SignLanguageApp()
            app.run()
        elif choice == '2':
            from Demo_ui.App import SignLanguageDemoUI
            app = SignLanguageDemoUI()
            app.run()
        elif choice == '3':
            print("🚀 Đang khởi động Cloud API Server...", flush=True)
            subprocess.run([
                sys.executable, "-m", "uvicorn", "Cloud_server.Api.Main:app",
                "--host", "127.0.0.1", "--port", "8000"
            ], cwd=BASE_DIR)
        elif choice in ('q', 'quit', 'exit'):
            print("👋 Cảm ơn bạn đã sử dụng hệ thống Ngôn ngữ Ký hiệu!", flush=True)
            break

if __name__ == "__main__":
    main()
