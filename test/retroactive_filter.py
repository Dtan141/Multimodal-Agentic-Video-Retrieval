import os
import json
import cv2
import numpy as np

# --- CẤU HÌNH ĐƯỜNG DẪN ---
# Thay đổi BASE_WORKSPACE cho khớp với đường dẫn thực tế của bạn
BASE_WORKSPACE = r"G:\.shortcut-targets-by-id\11I5_AMfAufb6crT2hzGrLEI3tMsTsKjX\AIC2026"
METADATA_FOLDER = os.path.join(BASE_WORKSPACE, "metadata", "L28")
KEYFRAMES_META_FOLDER = os.path.join(BASE_WORKSPACE, "keyframes_meta", "L28")

HISTOGRAM_THRESHOLD = 0.90  # Ngưỡng gom nhóm (Bạn có thể tinh chỉnh)

def process_retroactive_filter():
    if not os.path.exists(METADATA_FOLDER):
        print(f"❌ Không tìm thấy thư mục metadata: {METADATA_FOLDER}")
        return

    json_files = [f for f in os.listdir(METADATA_FOLDER) if f.endswith('.json')]
    
    if not json_files:
        print(f"⚠️ Không tìm thấy file JSON nào trong {METADATA_FOLDER}")
        return

    total_kept = 0
    total_deleted = 0

    print(f"🚀 BẮT ĐẦU LỌC LẠI FRAME CHO {len(json_files)} VIDEO TRONG THƯ MỤC L28...")
    print("="*60)

    for json_file in json_files:
        json_path = os.path.join(METADATA_FOLDER, json_file)
        video_id = os.path.splitext(json_file)[0]

        skip_list = [
            "L28_V011", "L28_V005", "L28_V002", "L28_V010", 
            "L28_V022", "L28_V009", "L28_V021", "L28_V007", 
            "L28_V014", "L28_V018", "L28_V015", "L28_V023", 
            "L28_V017", "L28_V020", "L28_V016", "L28_V024",
            "L28_V013", "L28_V006", "L28_V004", "L28_V001", 
            "L28_V008"
        ]
        
        if video_id in skip_list:
            print(f"⏭️ Bỏ qua {video_id} vì đã chạy xong ở lần trước.")
            continue
        
        # Thư mục chứa ảnh thực tế của video này
        kf_dir = os.path.join(KEYFRAMES_META_FOLDER, f"{video_id}_keyframes")
        
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        vid_kept = 0
        vid_deleted = 0

        # Duyệt qua từng segment
        for seg in data.get("segments", []):
            kf_data = seg.get("keyframe", {})
            
            # Xử lý tương thích ngược: nếu keyframe đang lưu dạng List thì chuyển thành list, nếu Dict thì lấy keys
            if isinstance(kf_data, dict):
                frame_filenames = list(kf_data.keys())
            else:
                frame_filenames = kf_data
                
            if not frame_filenames:
                continue

            # Sắp xếp theo tên để đảm bảo đúng thứ tự thời gian
            frame_filenames.sort()

            basket = []
            prev_hist = None
            winners = []

            # 1. Đọc và gom giỏ chờ
            for filename in frame_filenames:
                img_path = os.path.join(kf_dir, filename)
                if not os.path.exists(img_path):
                    continue

                frame = cv2.imread(img_path)

                if frame is None:
                    print(f"   ⚠️ Bỏ qua ảnh lỗi/không thể đọc: {filename}")
                    continue
                
                # Tính đặc trưng
                hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                hist = cv2.calcHist([hsv], [0, 1], None, [50, 60], [0, 180, 0, 256])
                cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)
                
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                lap_score = cv2.Laplacian(gray, cv2.CV_64F).var()

                if not basket:
                    basket.append((filename, img_path, lap_score))
                    prev_hist = hist
                else:
                    sim = cv2.compareHist(prev_hist, hist, cv2.HISTCMP_CORREL)
                    if sim >= HISTOGRAM_THRESHOLD:
                        # Gom chung giỏ
                        basket.append((filename, img_path, lap_score))
                        prev_hist = hist
                    else:
                        # Chốt giỏ cũ, tìm quán quân
                        winner = max(basket, key=lambda x: x[2])
                        winners.append(winner[0])
                        
                        # Xóa file vật lý của những kẻ thua cuộc
                        for item in basket:
                            if item[0] != winner[0]:
                                os.remove(item[1])
                                vid_deleted += 1
                        
                        # Mở giỏ mới
                        basket = [(filename, img_path, lap_score)]
                        prev_hist = hist

            # Xả giỏ cuối cùng của segment
            if basket:
                winner = max(basket, key=lambda x: x[2])
                winners.append(winner[0])
                for item in basket:
                    if item[0] != winner[0]:
                        os.remove(item[1])
                        vid_deleted += 1

            # 2. Cập nhật lại Metadata của Segment (Theo chuẩn Dict bạn yêu cầu)
            new_keyframe_dict = {}
            for w in winners:
                new_keyframe_dict[w] = {
                    "object": [],
                    "ocr": []
                }
                vid_kept += 1
                
            seg["keyframe"] = new_keyframe_dict

        # 3. Ghi đè lại file JSON
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
            
        print(f"✅ {video_id}: Giữ {vid_kept} frames | Xóa {vid_deleted} rác.")
        total_kept += vid_kept
        total_deleted += vid_deleted

    print("="*60)
    print("🎉 HOÀN TẤT DỌN DẸP!")
    print(f"📊 Tổng số frame giữ lại: {total_kept}")
    print(f"🗑️ Tổng số frame đã xóa (tiết kiệm ổ cứng): {total_deleted}")

if __name__ == "__main__":
    process_retroactive_filter()