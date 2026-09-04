
import streamlit as st
from pathlib import Path
from PIL import Image


# ============================================================
# CONFIG
# ============================================================

OUTPUT_DIR = Path("./test_optimize_img_comparison")

METHOD_FOLDERS = [
    "00_original",
    "jpeg_q95",
    "jpeg_q90",
    "jpeg_q85",
    "jpeg_q80",
    "jpeg_q90_optimized",
    "resize_1920_jpeg_q90_optimized",
    "webp_q90",
    "resize_1920_webp_q90",
    "png_lossless",
]

METHOD_NAMES = {
    "00_original": "Original",
    "jpeg_q95": "JPEG Q95",
    "jpeg_q90": "JPEG Q90",
    "jpeg_q85": "JPEG Q85",
    "jpeg_q80": "JPEG Q80",
    "jpeg_q90_optimized": "JPEG Q90 + Optimize",
    "resize_1920_jpeg_q90_optimized": "Resize 1920 + JPEG Q90",
    "webp_q90": "WebP Q90",
    "resize_1920_webp_q90": "Resize 1920 + WebP Q90",
    "png_lossless": "PNG Lossless",
}

SUPPORTED_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
    ".bmp",
}


# ============================================================
# HELPER
# ============================================================

def get_image_names():
    """
    Lấy danh sách stem ảnh từ toàn bộ các folder.
    Ví dụ:
        L21_V001_00123.jpg
        L21_V001_00123.webp

    => L21_V001_00123
    """
    names = set()

    for folder_name in METHOD_FOLDERS:
        folder = OUTPUT_DIR / folder_name

        if not folder.exists():
            continue

        for file in folder.iterdir():
            if (
                file.is_file()
                and file.suffix.lower() in SUPPORTED_EXTENSIONS
            ):
                names.add(file.stem)

    return sorted(names)


def find_image(folder_name, image_name):
    """
    Tìm ảnh theo stem, không phụ thuộc extension.
    """
    folder = OUTPUT_DIR / folder_name

    if not folder.exists():
        return None

    for ext in SUPPORTED_EXTENSIONS:
        path = folder / f"{image_name}{ext}"

        if path.exists():
            return path

    return None


def get_size_kb(path):
    return path.stat().st_size / 1024


# ============================================================
# STREAMLIT
# ============================================================

st.set_page_config(
    page_title="Keyframe Comparison",
    layout="wide",
)

st.title("Keyframe Optimization Comparison")

image_names = get_image_names()

if not image_names:
    st.error(
        f"Không tìm thấy ảnh trong folder:\n\n{OUTPUT_DIR.resolve()}"
    )
    st.stop()


# ============================================================
# SELECT IMAGE
# ============================================================

selected_image = st.selectbox(
    "Chọn ảnh",
    image_names,
)


# ============================================================
# DISPLAY GRID
# ============================================================

NUM_COLUMNS = 3

for start_idx in range(
    0,
    len(METHOD_FOLDERS),
    NUM_COLUMNS,
):
    cols = st.columns(NUM_COLUMNS)

    current_methods = METHOD_FOLDERS[
        start_idx:start_idx + NUM_COLUMNS
    ]

    for col, folder_name in zip(
        cols,
        current_methods,
    ):
        with col:
            image_path = find_image(
                folder_name,
                selected_image,
            )

            if image_path is None:
                st.warning(
                    f"Không có ảnh\n{folder_name}"
                )
                continue

            image = Image.open(image_path)

            st.image(
                image,
                use_container_width=True,
            )

            # Tên phương pháp ngay dưới ảnh
            st.markdown(
                f"### {METHOD_NAMES.get(folder_name, folder_name)}"
            )

            width, height = image.size
            size_kb = get_size_kb(image_path)

            st.caption(
                f"{width} × {height} | "
                f"{size_kb:.1f} KB"
            )

