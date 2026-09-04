import os
import json
import torch
from sentence_transformers import SentenceTransformer, util

# --- CẤU HÌNH ĐƯỜNG DẪN ---
DEBUG_FOLDER = "debug_outputs"
OUTPUT_FOLDER = "metadata"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

device = "cuda:0" if torch.cuda.is_available() else "cpu"

# 1. CHỈ KHỞI TẠO MÔ HÌNH EMBEDDING (Bỏ qua Florence & TransNet)
print("🚀 Đang tải mô hình Embedding (all-MiniLM-L6-v2)...")
embedder = SentenceTransformer('all-MiniLM-L6-v2').to(device)

# 2. HÀM GOM CỤM (Có in log điểm số để bạn dễ quan sát)
def group_shots_to_segments(shots, threshold):
    if not shots: return []

    segments = []
    current_segment = {
        "segment_id": "seg_0001",
        "start_time": shots[0]["start_time"],
        "end_time": shots[0]["end_time"],
        "shots": [shots[0]]
    }
    
    seg_counter = 1
    
    print(f"\n--- BẮT ĐẦU GOM CỤM (THRESHOLD = {threshold}) ---")
    for i in range(1, len(shots)):
        prev_caption = shots[i-1]["caption"]
        curr_caption = shots[i]["caption"]
        
        embeddings = embedder.encode([prev_caption, curr_caption], convert_to_tensor=True)
        cosine_score = util.cos_sim(embeddings[0], embeddings[1]).item()
        
        # In log để xem điểm số thực tế
        print(f"[{current_segment['shots'][-1]['shot_id']} <-> {shots[i]['shot_id']}] "
              f"Score: {cosine_score:.4f} "
              f"({'✅ GOM' if cosine_score >= threshold else '✂️ CẮT'})")
        
        if cosine_score >= threshold:
            current_segment["shots"].append(shots[i])
            current_segment["end_time"] = shots[i]["end_time"]
        else:
            segments.append(current_segment)
            seg_counter += 1
            current_segment = {
                "segment_id": f"seg_{seg_counter:04d}",
                "start_time": shots[i]["start_time"],
                "end_time": shots[i]["end_time"],
                "shots": [shots[i]]
            }
            
    segments.append(current_segment)
    print("--- KẾT THÚC GOM CỤM ---\n")
    
    for seg in segments:
        seg["segment_caption"] = " ".join([s["caption"] for s in seg["shots"]])
        for s in seg["shots"]:
            s.pop("image", None) # Đảm bảo dọn dẹp data thừa nếu có
            
    return segments

# 3. LUỒNG CHẠY TEST TỪ FILE JSON
def test_threshold(video_id, test_threshold):
    # Đọc data từ file đã lưu
    raw_json_path = os.path.join(DEBUG_FOLDER, video_id, "02_raw_captions.json")
    
    if not os.path.exists(raw_json_path):
        print(f"❌ Không tìm thấy file {raw_json_path}")
        return
        
    with open(raw_json_path, "r", encoding="utf-8") as f:
        shots_data = json.load(f)
        
    print(f"📥 Đã tải {len(shots_data)} shots từ cache của video '{video_id}'.")
    
    # Chạy hàm gom cụm
    segments_data = group_shots_to_segments(shots_data, threshold=test_threshold)
    
    # Xuất ra file mới để kiểm tra (tên file chứa luôn mức threshold để dễ so sánh)
    final_data = {
        "video_id": video_id,
        "segments": segments_data
    }
    
    output_json_path = os.path.join(OUTPUT_FOLDER, f"{video_id}_tuned_th{test_threshold}.json")
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(final_data, f, ensure_ascii=False, indent=4)
        
    print(f"🎉 Kết quả: Từ {len(shots_data)} shots ban đầu, đã gom lại thành {len(segments_data)} segments.")
    print(f"💾 Đã lưu kết quả tại: {output_json_path}\n")

if __name__ == "__main__":
    # --- BẠN THAY ĐỔI THÔNG SỐ Ở ĐÂY ĐỂ TEST ---
    VIDEO_NAME = "momo" 
    
    # Thử chạy liên tục nhiều mức threshold khác nhau để so sánh
    test_threshold(VIDEO_NAME, test_threshold=0.4)
    test_threshold(VIDEO_NAME, test_threshold=0.35)