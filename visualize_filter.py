import streamlit as st
import os
import cv2
import numpy as np
from PIL import Image

# --- CẤU HÌNH GIAO DIỆN STREAMLIT ---
st.set_page_config(page_title="Keyframe Filter Viewer", layout="wide")

# --- ĐƯỜNG DẪN DỮ LIỆU ---
INPUT_FOLDER = "test_keyframes_momo"
TUNE_FOLDER = "tune_tournament_momo"

def get_image_files(folder):
    """Hàm lấy danh sách ảnh đã được sắp xếp."""
    if not os.path.exists(folder):
        return []
    files = [f for f in os.listdir(folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    files.sort()
    return files

def main():
    st.title("👁️ Trình Trực Quan Hóa Bộ Lọc")
    st.markdown("Chọn một ngưỡng (Threshold) bên dưới. Các frame sáng màu là **được giữ lại**, các frame bị tối đi là **đã bị lọc bỏ**.")

    # 1. Kiểm tra thư mục gốc
    all_frames = get_image_files(INPUT_FOLDER)
    if not all_frames:
        st.error(f"❌ Không tìm thấy ảnh gốc trong thư mục '{INPUT_FOLDER}'. Hãy chạy file benchmark trước!")
        return

    # 2. Lấy danh sách các Threshold đã chạy
    if not os.path.exists(TUNE_FOLDER):
        st.error(f"❌ Không tìm thấy thư mục '{TUNE_FOLDER}'. Hãy chạy script tune_histogram.py trước!")
        return

    threshold_folders = [f for f in os.listdir(TUNE_FOLDER) if f.startswith("thresh_")]
    threshold_folders.sort()

    if not threshold_folders:
        st.warning("⚠️ Chưa có thư mục kết quả nào trong `tune_histogram`. Hãy đảm bảo bạn đã chạy script tuning.")
        return

    # 3. Tạo thanh chọn (Selectbox) cho Threshold
    selected_thresh_folder = st.selectbox(
        "🎛️ Chọn Ngưỡng (Threshold) để xem kết quả:",
        threshold_folders,
        index=len(threshold_folders)//2 # Mặc định chọn ở giữa
    )

    # 4. Xác định các frame được giữ lại ở Threshold này
    kept_folder_path = os.path.join(TUNE_FOLDER, selected_thresh_folder)
    kept_frames = set(get_image_files(kept_folder_path))

    # Thống kê nhanh
    st.markdown(f"**📊 Thống kê:** Giữ lại `{len(kept_frames)}` / `{len(all_frames)}` frames (Loại bỏ **{len(all_frames) - len(kept_frames)}** frames).")
    st.divider()

    # 5. Hiển thị Grid
    cols_per_row = 6  # Số lượng ảnh trên 1 hàng
    cols = st.columns(cols_per_row)

    for idx, filename in enumerate(all_frames):
        col = cols[idx % cols_per_row]
        img_path = os.path.join(INPUT_FOLDER, filename)
        
        # Đọc ảnh bằng OpenCV và chuyển sang RGB cho Streamlit
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        is_kept = filename in kept_frames

        if not is_kept:
            # LÀM TỐI ẢNH: Nhân ma trận pixel với 0.25 (Giảm độ sáng xuống còn 25%)
            # Ép kiểu về uint8 để tránh lỗi hiển thị
            img = (img * 0.25).astype(np.uint8)
            
            # (Tùy chọn) Vẽ thêm chữ "REMOVED" mờ mờ lên ảnh cho rõ ràng
            cv2.putText(img, "REMOVED", (10, img.shape[0] - 20), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 50, 50), 2)

        # Căn chỉnh hiển thị
        with col:
            st.image(img, use_container_width=True)
            if is_kept:
                st.caption(f"✅ {filename}")
            else:
                st.caption(f"❌ *{filename}*")

if __name__ == "__main__":
    main()