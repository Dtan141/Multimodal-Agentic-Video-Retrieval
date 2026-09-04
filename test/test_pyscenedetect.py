import os
import cv2
from PIL import Image
from scenedetect import detect, ContentDetector

# --- CẤU HÌNH ---
VIDEO_PATH = "video/thoisu_1.mp4" # Đảm bảo file này tồn tại
OUTPUT_DIR = "ex_debug_pyscenedetect/thoisu_1"

# Tạo thư mục lưu trữ
os.makedirs(os.path.join(OUTPUT_DIR, "keyframes"), exist_ok=True)

print(f"🎬 Đang phân tích video: {VIDEO_PATH} bằng PySceneDetect...")

# 1. CHẠY PYSCENEDETECT
# ContentDetector dùng ngưỡng mặc định là 27.0. 
# Ngưỡng càng thấp -> Cắt càng vụn; Ngưỡng càng cao -> Cắt càng ít
scene_list = detect(VIDEO_PATH, ContentDetector(threshold=27.0))

print(f"✅ Đã phát hiện {len(scene_list)} scenes (shots).")

# 2. LƯU LOG FRAME
with open(os.path.join(OUTPUT_DIR, "01_shot_frames.txt"), "w") as f:
    for i, scene in enumerate(scene_list):
        start_frame = scene[0].get_frames()
        end_frame = scene[1].get_frames()
        f.write(f"Shot {i+1:03d}: Frame {start_frame} -> {end_frame}\n")

# 3. TRÍCH XUẤT KEYFRAME (Frame ở giữa)
# 3. TRÍCH XUẤT KEYFRAME (Chuẩn xác 100% bằng Sequential Read)
print("📸 Đang trích xuất keyframes (Chế độ chính xác tuyệt đối)...")

# Gom tất cả các frame ở giữa (mid_frame) cần cắt vào một danh sách và sắp xếp tăng dần
target_frames = []
for scene in scene_list:
    start_frame = scene[0].get_frames()
    end_frame = scene[1].get_frames()
    mid_frame = (start_frame + end_frame) // 2
    target_frames.append(mid_frame)

target_frames.sort()

cap = cv2.VideoCapture(VIDEO_PATH)
current_frame = 0
target_idx = 0

# Đọc lướt qua toàn bộ video từ đầu đến cuối
while cap.isOpened() and target_idx < len(target_frames):
    ret, frame = cap.read()
    if not ret:
        break # Kết thúc video
        
    # Nếu frame hiện tại đang khớp với frame mục tiêu cần cắt
    if current_frame == target_frames[target_idx]:
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame_rgb)
        
        # Lưu file (Đảm bảo tên shot bắt đầu từ 1)
        img.save(os.path.join(OUTPUT_DIR, "keyframes", f"shot_{target_idx+1:04d}.jpg"))
        
        # Chuyển sang mục tiêu tiếp theo
        target_idx += 1
        
    current_frame += 1

cap.release()
print(f"🎉 Hoàn tất! Đã trích xuất chính xác {target_idx} keyframes. Hãy kiểm tra thư mục: {OUTPUT_DIR}")