# mobile_app/src/main.py
import cv2
import requests
import threading
import time
from inference_engine.predictor import SignLanguagePredictor
from inference_engine.llm_corrector import LLMCorrector
from inference_engine.llm_translator import LLMTranslator  
from inference_engine.context_agent import ContextAgent
from tts_handler.speaker import TextToSpeech
from virtual_cam.virtual_camera import VirtualCamera
from shared_lib.constants import ACTION_CLASSES

CONFIDENCE_THRESHOLD = 0.6
SERVER_URL = "http://localhost:8000"

class SignLanguageApp:
    def __init__(self):
        self.predictor = SignLanguagePredictor(
            yolo_path="assets/hand_det_yolo.tflite",
            feature_path="assets/feature_extractor.tflite",
            gru_path="assets/action_recognizer.tflite"
        )
        self.tts = TextToSpeech()
        self.virtual_cam = VirtualCamera()
        self.cap = cv2.VideoCapture(0)

#=======================khởi tạo LLMCorrector và ContextAgent========================
        self.llm_corrector = LLMCorrector()  # Tạo instance LLMCorrector
        self.context_agent = ContextAgent(llm_corrector=self.llm_corrector)  # Truyền LLMCorrector vào ContextAgent để dùng chung client
        self.translator = LLMTranslator(llm_corrector=self.llm_corrector)  # Tạo instance LLMTranslator
    
    
        self.edge_case_buffer = []
        self.last_action = None


#=======================Giới hạn tốc độ xử lý AI =========================
        self.last_process_time = 0
        self.process_interval = 0.1  # Giới hạn xử lý mỗi 100ms (10fps)
        # Chỉnh tùy máy: nếu máy mạnh có thể giảm xuống 0.05 (20fps), nếu máy yếu có thể tăng lên 0.2 (5fps) để giảm tải CPU
        # 1.0 / 10 = 10fps → cân bằng
        # 1.0 / 20 = 20fps → mượt hơn nhưng tốn CPU hơn
        # 1.0 / 5 = 5fps → tiết kiệm CPU nhưng có thể thấy giật lag        

    def send_edge_case(self, video_frames, predicted_label):
        """Gửi video bị nhận diện sai lên server để gán nhãn lại."""
        # Tạo file video tạm
        if not video_frames:
            return
        h, w = video_frames[0].shape[:2]
        temp_file = "temp_edge.mp4"
        out = cv2.VideoWriter(temp_file, cv2.VideoWriter_fourcc(*'mp4v'), 10, (w, h))
        for f in video_frames:
            out.write(f)
        out.release()

        with open(temp_file, 'rb') as f:
            files = {'video': f}
            data = {'label': predicted_label}
            try:
                requests.post(f"{SERVER_URL}/upload_edge_case", files=files, data=data)
                print("Edge case sent.")
            except Exception as e:
                print("Failed to send edge case:", e)

    def run(self):
        print("=== Ứng dụng nhận diện ngôn ngữ ký hiệu ===")
        print("Phím tắt:")
        print("  [Q] Thoát")
        print("  [V] Tiếng Việt")
        print("  [E] Tiếng Anh")
        print("  [J] Tiếng Nhật")
        print("  [K] Tiếng Hàn")
        print("==========================================")

        # Lưu kết quả cuối để hiển thị liên tục lên màn hình
        last_display_text = ""


        while True:
            ret, frame = self.cap.read()
            if not ret:
                break

            static_char, action, conf, bbox = self.predictor.process_frame(frame)

            # Vẽ bounding box và thông tin lên frame
            display = frame.copy()
            if bbox:
                x1, y1, x2, y2 = bbox
                cv2.rectangle(display, (x1, y1), (x2, y2), (0,255,0), 2)
            if action:
                cv2.putText(display, f"{action} ({conf:.2f})", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
                if conf < CONFIDENCE_THRESHOLD:
                    self.edge_case_buffer.append(frame.copy())
                    if len(self.edge_case_buffer) >= self.predictor.seq_len:
                        threading.Thread(target=self.send_edge_case,
                                         args=(self.edge_case_buffer.copy(), action)).start()
                        self.edge_case_buffer = []
                else:
                    self.edge_case_buffer = []
                    # Trước: self.tts.speak(action) — nói thẳng, không sửa
                    # Sau:   đưa qua agent → sửa lỗi → dịch → mới nói
                    if action != self.last_action:
                         final_text = self.agent.process(action_word=action)
                        if final_text:
                            self.tts.speak(final_text)     # Nói tiếng Việt ✅
                            cv2.putText(                   # Hiển thị lên màn hình
                                display, final_text, (10, 70),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2
                            )
                        self.last_action = action

            # Gửi frame ra camera ảo
            self.virtual_cam.send_frame(display)

            # BẬT CỬA SỔ HIỂN THỊ ĐỂ BẮT SỰ KIỆN PHÍM 'q' KHI TEST
            cv2.imshow("Sign Language App", display)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break

            # Giới hạn tốc độ xử lý để tránh quá tải CPU
            time.sleep(0.03)



        self.cap.release()
        self.virtual_cam.close()

if __name__ == "__main__":
    app = SignLanguageApp()
    app.run()







# tham khảo thêm về độ fps và tối ưu hiệu suất:
    #Phim chiếu rạp          24fps   → mắt người thấy mượt 
    #Game bình thường        30fps   → mượt
    #Game cao cấp             60fps   → rất mượt, đặc biệt với game hành động nhanh
    #Video call (Zoom)        30fps   → mượt
    #Video call (Google Meet) 15-30fps → mượt nếu đủ băng thông
    #Để đạt được 30fps, mỗi khung hình phải được xử lý trong khoảng 33ms. Nếu xử lý mất nhiều thời gian hơn, fps sẽ giảm xuống.
    #Nhận diện ký hiệu tay  10fps   → ĐỦ DÙNG, vì ngôn ngữ ký hiệu không yêu cầu chuyển động quá nhanh như game hành động, và việc xử lý hình ảnh phức tạp có thể mất nhiều thời gian hơn. 10fps là mức tối thiểu để đảm bảo trải nghiệm người dùng không bị giật lag quá nhiều, đồng thời vẫn cho phép mô hình có đủ thời gian để phân tích và đưa ra dự đoán chính xác. Nếu có thể tối ưu để đạt 15-20fps thì sẽ càng tốt, nhưng 10fps là mức chấp nhận được cho ứng dụng này.
