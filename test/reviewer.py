import streamlit as st
import json
import os
from PIL import Image

# --- HÀM HỖ TRỢ: TẠO PHỤ ĐỀ WEBVTT TỰ ĐỘNG ---
def format_vtt_time(seconds):
    """Chuyển đổi giây (float) sang định dạng HH:MM:SS.mmm của WebVTT"""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds - int(seconds)) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"

def generate_segment_subtitles(segments, output_path="temp_segments.vtt"):
    """Tạo file phụ đề WebVTT để phủ chữ lên video"""
    vtt_content = "WEBVTT\n\n"
    for i, seg in enumerate(segments):
        start = format_vtt_time(seg['start_time'])
        end = format_vtt_time(seg['end_time'])
        
        # Cắt ngắn caption để không che hết màn hình video
        short_cap = " ".join(seg.get('segment_caption', '').split()[:12]) + "..."
        
        vtt_content += f"{i+1}\n{start} --> {end}\n"
        vtt_content += f"📌 {seg['segment_id']} | {short_cap}\n\n"
        
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(vtt_content)
    return output_path

# --- CẤU HÌNH GIAO DIỆN ---
st.set_page_config(layout="wide", page_title="AIC Metadata Reviewer Ultimate")
st.title("🎬 Trình kiểm tra Metadata Phân đoạn Video (Ultimate)")

# --- 1. CHỌN FILE JSON ---
metadata_dir = "metadata"
if not os.path.exists(metadata_dir):
    st.error(f"Không tìm thấy thư mục '{metadata_dir}'.")
    st.stop()

json_files = [f for f in os.listdir(metadata_dir) if f.endswith('.json')]
if not json_files:
    st.warning("Chưa có file JSON nào trong thư mục metadata.")
    st.stop()

selected_file = st.sidebar.selectbox("📂 Chọn Video để kiểm tra", json_files)

with open(os.path.join(metadata_dir, selected_file), 'r', encoding='utf-8') as f:
    data = json.load(f)

video_id = data.get("video_id", "Unknown")
video_path = data.get("video_path", "")
segments = data.get("segments", [])
total_duration = segments[-1]["end_time"] if segments else 1

st.sidebar.markdown(f"**Tổng số Segments:** {len(segments)}")
st.sidebar.markdown(f"**Tổng thời lượng:** {total_duration}s")

# --- 2. VẼ THANH TIMELINE TỔNG QUÁT BẰNG HTML/CSS ---
st.markdown("### ⏱️ Timeline Phân đoạn (Global Video Timeline)")

# Ép HTML trên 1 dòng để tránh lỗi Markdown Block
timeline_html = '<div style="width: 100%; height: 35px; display: flex; border-radius: 8px; overflow: hidden; margin-bottom: 20px; border: 1px solid #555;">'
colors = ["#4B8BBE", "#306998", "#FFE873", "#FFD43B", "#646464"]

for i, seg in enumerate(segments):
    duration = seg['end_time'] - seg['start_time']
    width_pct = (duration / total_duration) * 100
    color = colors[i % len(colors)]
    tooltip = f"{seg['segment_id']}: {seg['start_time']}s - {seg['end_time']}s"
    label = seg['segment_id'] if width_pct > 5 else "" 
    timeline_html += f'<div style="width: {width_pct}%; background-color: {color}; text-align: center; line-height: 35px; color: #111; font-weight: bold; font-size: 12px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; cursor: pointer;" title="{tooltip}">{label}</div>'

timeline_html += "</div>"
st.markdown(timeline_html, unsafe_allow_html=True)

# --- 3. BỐ CỤC CHÍNH (MASTER - DETAIL) ---
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("📑 Danh sách Segments")
    segment_options = {f"{seg['segment_id']} ({seg['start_time']}s - {seg['end_time']}s)": seg for seg in segments}
    selected_seg_label = st.radio("Chọn Segment để xem chi tiết:", list(segment_options.keys()))

if selected_seg_label:
    selected_seg = segment_options[selected_seg_label]
    start_s = int(selected_seg['start_time'])
    
    with col2:
        st.subheader(f"🔍 Đang xem: {selected_seg['segment_id']}")
        
        # --- PLAYER TRỰC TIẾP TÍCH HỢP PHỤ ĐỀ OVERLAY ---
        if os.path.exists(video_path):
            # Tạo file phụ đề đè lên video
            vtt_path = generate_segment_subtitles(segments)
            
            # Truyền phụ đề vào player
            st.video(video_path, start_time=start_s, subtitles=vtt_path)
            
            st.caption(f"Trình phát đang tự động bật **Phụ đề (CC)** để hiển thị ID của Segment đè lên video. Đã tua tới mốc **{start_s} giây**.")
        else:
            st.error(f"Không tìm thấy file video gốc tại đường dẫn: {video_path}")
            
        st.info(f"**Caption tổng hợp:** {selected_seg.get('segment_caption', '')}")
        st.markdown("---")
        
        # --- CHI TIẾT CÁC SHOT BÊN TRONG ---
        st.markdown("### 🎞️ Các Shots thành phần (Keyframes)")
        shots = selected_seg.get("shots", [])
        
        for shot in shots:
            with st.container():
                shot_col1, shot_col2 = st.columns([1, 3])
                
                with shot_col1:
                    img_path = os.path.join("debug_outputs", video_id, "keyframes", f"{shot['shot_id']}.jpg")
                    if os.path.exists(img_path):
                        img = Image.open(img_path)
                        st.image(img, use_container_width=True)
                    else:
                        st.warning("Không có ảnh")
                
                with shot_col2:
                    st.write(f"**{shot['shot_id']}** (`{shot['start_time']}s` - `{shot['end_time']}s`)")
                    st.write(f"📝 *Caption:* {shot['caption']}")
            st.markdown("<hr style='margin: 10px 0; border-top: 1px dashed #ccc;'>", unsafe_allow_html=True)