import cv2
import torch
import math
from PIL import Image
from transformers import AutoProcessor, AutoModel

# --- 1. HÀM TRÍCH XUẤT FRAME TỪ SEGMENT ---
def extract_frames_from_segment(video_path, start_time, end_time, min_frames=3, max_step=20):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    start_frame = int(start_time * fps)
    end_frame = int(end_time * fps)
    total_frames = end_frame - start_frame
    
    # Tính toán số lượng frame cần lấy dựa trên quy tắc của bạn
    # Công thức: N = max(min_frames, ceil(total_frames / max_step) + 1)
    num_frames = max(min_frames, math.ceil(total_frames / max_step) + 1)
    
    # Tính khoảng cách thực tế giữa các frame
    step = total_frames // (num_frames - 1) if num_frames > 1 else 0
    
    target_frames = [start_frame + i * step for i in range(num_frames)]
    # Đảm bảo frame cuối cùng không vượt quá end_frame
    target_frames[-1] = min(target_frames[-1], end_frame)
    
    extracted_images = []
    timestamps = []
    
    for f_idx in target_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, f_idx)
        ret, frame = cap.read()
        if ret:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            extracted_images.append(Image.fromarray(frame_rgb))
            timestamps.append(round(f_idx / fps, 2))
            
    cap.release()
    return extracted_images, timestamps

# --- 2. LUỒNG TÌM KIẾM MÔ PHỎNG (VỚI SIGLIP) ---
def simulate_search(video_path, start_time, end_time, text_query):
    print("📸 Đang trích xuất frames từ segment...")
    images, timestamps = extract_frames_from_segment(video_path, start_time, end_time)
    print(f"✅ Đã trích xuất {len(images)} frames tại các mốc (giây): {timestamps}")
    
    print("\n🚀 Đang tải mô hình SigLIP 2...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_id = "google/siglip-base-patch16-224" # Dùng bản base để test cho nhẹ
    model = AutoModel.from_pretrained(model_id).to(device).eval()
    processor = AutoProcessor.from_pretrained(model_id)
    
    print(f"🔍 Đang tìm kiếm frame khớp với câu: '{text_query}'")
    # Tiền xử lý Text và Image
    inputs = processor(text=[text_query], images=images, padding="max_length", return_tensors="pt").to(device)
    
    with torch.no_grad():
        # Lấy vector nhúng (embeddings)
        image_embeddings = model.get_image_features(**inputs)
        text_embeddings = model.get_text_features(**inputs)
        
        # Chuẩn hóa vector (Normalize)
        image_embeddings = image_embeddings / image_embeddings.norm(p=2, dim=-1, keepdim=True)
        text_embeddings = text_embeddings / text_embeddings.norm(p=2, dim=-1, keepdim=True)
        
        # Tính độ tương đồng Cosine (Cosine Similarity)
        similarities = torch.matmul(text_embeddings, image_embeddings.t()).squeeze(0)
    
    # Tìm frame có điểm cao nhất
    best_idx = torch.argmax(similarities).item()
    best_score = similarities[best_idx].item()
    best_time = timestamps[best_idx]
    
    print("\n📊 KẾT QUẢ XẾP HẠNG:")
    for i, score in enumerate(similarities):
        print(f" - Frame {timestamps[i]}s : Score = {score.item():.4f}")
        
    print(f"\n🏆 FRAME CHIẾN THẮNG: {best_time}s (Score: {best_score:.4f})")
    
    # Hiển thị ảnh chiến thắng (Nếu chạy trên Colab)
    # images[best_idx].show() 

# --- CHẠY THỬ ---
if __name__ == "__main__":
    # Giả sử metadata đã trả về segment từ 13.2s đến 20.6s
    simulate_search(
        video_path="videos/momo.mp4", 
        start_time=13.2, 
        end_time=20.6, 
        text_query="A person sitting at a table in a dark hallway"
    )