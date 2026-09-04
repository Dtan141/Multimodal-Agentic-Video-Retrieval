import streamlit as st
import cv2
import torch
import math
from PIL import Image
from transformers import AutoProcessor, AutoModel

st.set_page_config(layout="wide", page_title="SigLIP Frame Search")

# --- 1. TẢI MÔ HÌNH VÀ CACHE VÀO RAM/VRAM ---
# Dùng @st.cache_resource để Streamlit không tải lại mô hình mỗi khi bạn gõ chữ
@st.cache_resource
def load_model():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model_id = "google/siglip-base-patch16-224"
    model = AutoModel.from_pretrained(model_id).to(device).eval()
    processor = AutoProcessor.from_pretrained(model_id)
    return model, processor, device

# --- 2. HÀM TRÍCH XUẤT FRAME ---
def extract_frames_from_segment(video_path, start_time, end_time, min_frames=3, max_step=20):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    start_frame = int(start_time * fps)
    end_frame = int(end_time * fps)
    total_frames = end_frame - start_frame
    
    num_frames = max(min_frames, math.ceil(total_frames / max_step) + 1)
    step = total_frames // (num_frames - 1) if num_frames > 1 else 0
    
    target_frames = [start_frame + i * step for i in range(num_frames)]
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

# --- 3. GIAO DIỆN STREAMLIT ---
st.title("🔍 SigLIP Local Search Engine")

# Load model (Sẽ tốn khoảng 10-20s cho lần chạy đầu tiên)
with st.spinner("Đang tải mô hình SigLIP..."):
    model, processor, device = load_model()

# Tạo Form nhập liệu
col1, col2 = st.columns([1, 2])
with col1:
    st.markdown("### Cấu hình Segment")
    video_path = st.text_input("Đường dẫn Video", value="videos/momo.mp4")
    start_time = st.number_input("Start Time (s)", value=13.2, step=0.1)
    end_time = st.number_input("End Time (s)", value=20.6, step=0.1)

with col2:
    st.markdown("### Truy vấn (Query)")
    text_query = st.text_input("Nhập câu tìm kiếm:", value="A person sitting at a table in a dark hallway")
    search_btn = st.button("🚀 Tìm kiếm Top Frames", use_container_width=True)

# --- 4. XỬ LÝ TÌM KIẾM ---
if search_btn:
    with st.spinner("Đang trích xuất frame và chấm điểm..."):
        images, timestamps = extract_frames_from_segment(video_path, start_time, end_time)
        
        # Tiền xử lý và Embed
        inputs = processor(text=[text_query], images=images, padding="max_length", return_tensors="pt").to(device)
        
        with torch.no_grad():
            # Chỉ truyền 'pixel_values' cho hàm xử lý ảnh
            image_embeddings = model.get_image_features(pixel_values=inputs["pixel_values"])
            
            # Chỉ truyền 'input_ids' và 'attention_mask' cho hàm xử lý chữ
            text_embeddings = model.get_text_features(
                input_ids=inputs["input_ids"], 
                attention_mask=inputs.get("attention_mask")
            )
            
            # Chuẩn hóa vector
            image_embeddings = image_embeddings / image_embeddings.norm(p=2, dim=-1, keepdim=True)
            text_embeddings = text_embeddings / text_embeddings.norm(p=2, dim=-1, keepdim=True)
            
            # Tính toán khoảng cách
            similarities = torch.matmul(text_embeddings, image_embeddings.t()).squeeze(0)
        
        # Lấy Top 10
        actual_k = min(10, len(images))
        top_scores, top_indices = torch.topk(similarities, actual_k)
        
        st.success(f"Đã xử lý xong {len(images)} frames! Hiển thị Top {actual_k}:")
        
        # Vẽ lưới ảnh bằng st.columns
        cols = st.columns(5) # Chia thành 5 cột mỗi hàng
        for i in range(actual_k):
            idx = top_indices[i].item()
            score = top_scores[i].item()
            time_mark = timestamps[idx]
            
            # Đẩy ảnh vào đúng cột tương ứng (xuống dòng tự động)
            with cols[i % 5]:
                st.image(images[idx], caption=f"Rank {i+1} | {time_mark}s\nScore: {score:.4f}", use_container_width=True)