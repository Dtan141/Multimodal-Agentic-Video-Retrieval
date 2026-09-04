import cv2
import os
import shutil
import numpy as np
import time

# --- CẤU HÌNH ĐƯỜNG DẪN ---
INPUT_FOLDER = "test_keyframes_momo" # Đổi tên này thành thư mục chứa ảnh test của bạn
BENCHMARK_FOLDER = "benchmark_filter"

# --- NGƯỠNG CẤU HÌNH (THRESHOLDS) ---
PIXEL_DIFF_THRESHOLD = 15.0    # Càng nhỏ càng giữ nhiều ảnh (0-255)
HISTOGRAM_THRESHOLD = 0.8     # Càng gần 1.0 càng lọc gắt (0.0 - 1.0)
BLUR_THRESHOLD = 50.0          # Càng lớn càng yêu cầu ảnh phải cực kỳ sắc nét

def setup_folders():
    if os.path.exists(BENCHMARK_FOLDER):
        shutil.rmtree(BENCHMARK_FOLDER)
    
    folders = {
        "method_1": os.path.join(BENCHMARK_FOLDER, "1_pixel_diff"),
        "method_2": os.path.join(BENCHMARK_FOLDER, "2_histogram"),
        "method_3": os.path.join(BENCHMARK_FOLDER, "3_blur_laplacian")
    }
    
    for f in folders.values():
        os.makedirs(f, exist_ok=True)
        
    return folders

def get_image_files(folder):
    files = [f for f in os.listdir(folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    files.sort() # Sắp xếp theo tên để đảm bảo thứ tự thời gian
    return files

def run_benchmark():
    if not os.path.exists(INPUT_FOLDER):
        print(f"❌ Không tìm thấy thư mục đầu vào: {INPUT_FOLDER}")
        return

    image_files = get_image_files(INPUT_FOLDER)
    total_frames = len(image_files)
    
    if total_frames == 0:
        print(f"⚠️ Thư mục '{INPUT_FOLDER}' không có ảnh nào!")
        return

    print(f"🚀 Bắt đầu Benchmark trên {total_frames} frames...")
    out_folders = setup_folders()
    
    # Biến lưu trữ kết quả
    kept_counts = {"method_1": 0, "method_2": 0, "method_3": 0}
    time_taken = {"method_1": 0.0, "method_2": 0.0, "method_3": 0.0}

    # ==========================================
    # PHƯƠNG PHÁP 1: PIXEL DIFFERENCE (Độ chênh lệch Pixel)
    # ==========================================
    print(" -> Đang chạy Phương pháp 1: Pixel Difference...")
    start_time = time.time()
    prev_frame_gray = None
    
    for filename in image_files:
        img_path = os.path.join(INPUT_FOLDER, filename)
        frame = cv2.imread(img_path)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        keep_frame = False
        if prev_frame_gray is None:
            keep_frame = True # Luôn giữ frame đầu tiên
        else:
            # Tính trung bình độ lệch tuyệt đối giữa 2 ảnh
            diff = cv2.absdiff(prev_frame_gray, gray)
            mean_diff = np.mean(diff)
            
            if mean_diff > PIXEL_DIFF_THRESHOLD:
                keep_frame = True
        
        if keep_frame:
            cv2.imwrite(os.path.join(out_folders["method_1"], filename), frame)
            prev_frame_gray = gray
            kept_counts["method_1"] += 1
            
    time_taken["method_1"] = time.time() - start_time

    # ==========================================
    # PHƯƠNG PHÁP 2: HISTOGRAM (Biểu đồ màu sắc)
    # ==========================================
    print(" -> Đang chạy Phương pháp 2: Histogram Comparison...")
    start_time = time.time()
    prev_hist = None
    
    for filename in image_files:
        img_path = os.path.join(INPUT_FOLDER, filename)
        frame = cv2.imread(img_path)
        
        # Tính toán Histogram cho kênh màu HSV (Tốt hơn RGB)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None, [50, 60], [0, 180, 0, 256])
        cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)
        
        keep_frame = False
        if prev_hist is None:
            keep_frame = True
        else:
            similarity = cv2.compareHist(prev_hist, hist, cv2.HISTCMP_CORREL)
            if similarity < HISTOGRAM_THRESHOLD:
                keep_frame = True
                
        if keep_frame:
            cv2.imwrite(os.path.join(out_folders["method_2"], filename), frame)
            prev_hist = hist
            kept_counts["method_2"] += 1
            
    time_taken["method_2"] = time.time() - start_time

    # ==========================================
    # PHƯƠNG PHÁP 3: LAPLACIAN (Lọc ảnh mờ)
    # Lưu ý: PP này không loại ảnh trùng, chỉ loại ảnh nhòe
    # ==========================================
    print(" -> Đang chạy Phương pháp 3: Laplacian Blur Filter...")
    start_time = time.time()
    
    for filename in image_files:
        img_path = os.path.join(INPUT_FOLDER, filename)
        frame = cv2.imread(img_path)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Tính phương sai của phương pháp Laplacian
        variance = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        if variance >= BLUR_THRESHOLD:
            cv2.imwrite(os.path.join(out_folders["method_3"], filename), frame)
            kept_counts["method_3"] += 1
            
    time_taken["method_3"] = time.time() - start_time

    # ==========================================
    # IN BÁO CÁO THỐNG KÊ
    # ==========================================
    print("\n" + "="*60)
    print("📊 BÁO CÁO THỐNG KÊ KẾT QUẢ BENCHMARK")
    print("="*60)
    print(f"📁 Thư mục gốc: {INPUT_FOLDER}")
    print(f"🖼️ Tổng số frame gốc: {total_frames}\n")
    
    print(f"{'Phương pháp':<25} | {'Số frame giữ lại':<18} | {'Tỷ lệ nén':<12} | {'Thời gian chạy'}")
    print("-" * 75)
    
    for method_id, method_name in [("method_1", "1. Pixel Difference"), 
                                   ("method_2", "2. Histogram"), 
                                   ("method_3", "3. Laplacian (Lọc Mờ)")]:
        kept = kept_counts[method_id]
        ratio = ((total_frames - kept) / total_frames) * 100 if total_frames > 0 else 0
        t_taken = time_taken[method_id]
        print(f"{method_name:<25} | {kept:<4} / {total_frames:<11} | 📉 {ratio:05.2f}% | ⏱️ {t_taken:.3f}s")
    print("="*75)
    print(f"✅ Kết quả đã được lưu tại thư mục: {BENCHMARK_FOLDER}/")

if __name__ == "__main__":
    run_benchmark()