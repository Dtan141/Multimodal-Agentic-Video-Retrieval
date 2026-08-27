import os
import cv2
import shutil
import numpy as np
import imagehash
from PIL import Image
import gc
from sklearn.cluster import DBSCAN

# ============================================================
# CẤU HÌNH HUGGING FACE
# ============================================================
HF_REPO_ID = "Chillguy2026/dataset_video"
TARGET_FOLDERS = ["Videos_L25"]

# ============================================================
# CẤU HÌNH ĐƯỜNG DẪN
# ============================================================
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_WORKSPACE = os.path.join(CURRENT_DIR, "workspace")
# Đầu vào và Đầu ra
LOCAL_DATASET_DIR = os.path.join(BASE_WORKSPACE, "hf_downloaded_videos")
KEYFRAMES_VECTOR_FOLDER = os.path.join(BASE_WORKSPACE, "keyframes_vector")
LOCAL_TEMP_FOLDER = os.path.join(
    BASE_WORKSPACE,
    "temp_vector_processing",
)

os.makedirs(LOCAL_DATASET_DIR, exist_ok=True)
os.makedirs(KEYFRAMES_VECTOR_FOLDER, exist_ok=True)
os.makedirs(LOCAL_TEMP_FOLDER, exist_ok=True)

# ============================================================
# CẤU HÌNH THUẬT TOÁN
# ============================================================
FRAME_INTERVAL = 20          # Lấy mẫu mỗi 20 frame
PHASH_EPS = 6                # Ngưỡng khoảng cách Hamming 
JPEG_QUALITY = 85

VIDEO_EXTENSIONS = (".mp4", ".avi", ".mkv",)

# ============================================================
# HELPER
# ============================================================
def process_vector_pipeline():
    # 1. Tìm tất cả các file video trong thư mục
    video_paths = []
    for root, dirs, files in os.walk(LOCAL_DATASET_DIR):
        for f in files:
            if f.lower().endswith(('.mp4', '.avi', '.mkv')):
                video_paths.append(os.path.join(root, f))

    if not video_paths:
        print(f"⚠️ Không tìm thấy video nào trên Drive '{LOCAL_DATASET_DIR}'.")
        return

    for drive_video_path in video_paths:
        video_file = os.path.basename(drive_video_path)
        raw_video_id = os.path.splitext(video_file)[0]
        folder_chua_vid = os.path.basename(os.path.dirname(drive_video_path))

        if raw_video_id.startswith(f"{folder_chua_vid}_"):
            base_name = raw_video_id
        else:
            base_name = f"{folder_chua_vid}_{raw_video_id}"
            
        print(f"\n[{folder_chua_vid} / {video_file}] Đang xử lý Vector Extraction...")
        
        # Copy file về SSD local để tránh nghẽn I/O khi đọc frame
        local_video_path = os.path.join(LOCAL_TEMP_FOLDER, video_file)
        print(" ⏳ Đang copy video xuống SSD để đọc tuần tự...")
        shutil.copy2(drive_video_path, local_video_path)
        
        out_kf_dir = os.path.join(KEYFRAMES_VECTOR_FOLDER, folder_chua_vid, f"{base_name}_vector_frames")
        os.makedirs(out_kf_dir, exist_ok=True)
        
        # Thư mục nháp để chứa frame ứng viên tạm thời
        tmp_frames_dir = os.path.join(LOCAL_TEMP_FOLDER, f"tmp_{base_name}_frames")
        os.makedirs(tmp_frames_dir, exist_ok=True)

        try:
            print(f"  -> BƯỚC 1: Đọc tuần tự & Lấy mẫu (Mỗi {FRAME_INTERVAL} frame)...")
            cap = cv2.VideoCapture(local_video_path)
            
            current_frame = 0
            candidates = []
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Cứ 20 frame thì bốc ra làm ứng cử viên
                if current_frame % FRAME_INTERVAL == 0:
                    # Lưu tạm ra ổ cứng để không làm tràn RAM nếu video quá dài
                    tmp_img_path = os.path.join(tmp_frames_dir, f"tmp_{current_frame:05d}.jpg")
                    cv2.imwrite(tmp_img_path, frame)
                    
                    # Tính toán pHash ngay lập tức
                    pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    h = imagehash.phash(pil_img)
                    
                    # Chuyển pHash (8x8 bool) thành mảng số nguyên nhị phân 64 chiều cho DBSCAN
                    bin_hash = h.hash.flatten().astype(int)
                    
                    candidates.append({
                        "frame_idx": current_frame,
                        "img_path": tmp_img_path,
                        "hash_bin": bin_hash
                    })
                
                current_frame += 1
            cap.release()
            
            num_candidates = len(candidates)
            print(f"     ✅ Bốc được {num_candidates} frame ứng cử viên.")
            
            if num_candidates == 0:
                continue

            print("  -> BƯỚC 2: Phân cụm pHash (DBSCAN) & Chọn Top-1 Trung tâm...")
            
            # Chuyển đổi dữ liệu cho thuật toán phân cụm
            X = np.array([c["hash_bin"] for c in candidates])
            
            # Ngưỡng Hamming trong DBSCAN tính theo tỷ lệ (phash_eps / 64)
            dbscan_eps = float(PHASH_EPS) / 64.0
            
            db = DBSCAN(eps=dbscan_eps, min_samples=1, metric='hamming', n_jobs=-1)
            labels = db.fit_predict(X)
            
            # Gom các ứng viên vào cụm (cluster)
            clusters = {}
            for idx, lab in enumerate(labels):
                clusters.setdefault(lab, []).append(candidates[idx])
            
            # Tiến hành lọc Top-1 cho từng cụm
            kept_count = 0
            for lab, items in clusters.items():
                M = len(items)
                if M == 1:
                    # Cụm chỉ có 1 ảnh -> Chắc chắn được chọn
                    best_candidate = items[0]
                else:
                    # Cụm có nhiều ảnh -> Chấm điểm độ trung tâm (Centrality)
                    H = np.zeros((M, M), dtype=np.float32)
                    h_list = [item["hash_bin"] for item in items]
                    
                    # Tính ma trận khoảng cách Hamming nội bộ cụm
                    for i in range(M):
                        for j in range(i+1, M):
                            # Số bit khác nhau chia cho 64
                            dist = np.count_nonzero(h_list[i] != h_list[j]) / 64.0
                            H[i, j] = H[j, i] = dist
                            
                    sim_matrix = 1.0 - H
                    centrality_scores = sim_matrix.sum(axis=1)
                    
                    # Lấy index của ảnh có độ trung tâm cao nhất
                    best_idx = np.argmax(centrality_scores)
                    best_candidate = items[best_idx]
                
                # Lưu file chiến thắng sang ổ Google Drive theo đúng chuẩn ID
                frame_id = best_candidate["frame_idx"]
                final_name = f"{base_name}_{frame_id:05d}.jpg"
                final_path = os.path.join(out_kf_dir, final_name)
                
                shutil.copy2(best_candidate["img_path"], final_path)
                kept_count += 1
                
            print(f"     ✅ Phân thành {len(clusters)} cụm. Giữ lại {kept_count}/{num_candidates} frames.")
            print(f"🎉 Hoàn tất! Ảnh đã lưu vào: {out_kf_dir}")

        except Exception as e:
            print(f"❌ Lỗi khi xử lý {video_file}: {e}")
            
        finally:
            # Dọn rác SSD: Xóa video và các ảnh ứng cử viên tạm
            if os.path.exists(local_video_path):
                os.remove(local_video_path)
            if os.path.exists(tmp_frames_dir):
                shutil.rmtree(tmp_frames_dir)
                
            # Giải phóng RAM
            gc.collect()
            print("-" * 60)

if __name__ == "__main__":
    process_vector_pipeline()