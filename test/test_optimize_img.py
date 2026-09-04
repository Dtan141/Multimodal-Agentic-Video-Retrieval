import argparse
import csv
import shutil
from pathlib import Path

import cv2


# ============================================================
# CẤU HÌNH THỬ NGHIỆM
# ============================================================

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

JPEG_QUALITIES = [95, 90, 85, 80]
WEBP_QUALITY = 90
MAX_KEYFRAME_WIDTH = 1920


# ============================================================
# HÀM HỖ TRỢ
# ============================================================

def get_image_files(input_dir: Path):
    """Lấy toàn bộ ảnh trong input folder, không duyệt recursive."""
    return sorted(
        path
        for path in input_dir.iterdir()
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )


def resize_max_width(image, max_width=MAX_KEYFRAME_WIDTH):
    """
    Chỉ resize khi chiều rộng ảnh lớn hơn max_width.
    Giữ nguyên aspect ratio.
    """
    height, width = image.shape[:2]

    if width <= max_width:
        return image

    scale = max_width / width
    new_height = round(height * scale)

    return cv2.resize(
        image,
        (max_width, new_height),
        interpolation=cv2.INTER_AREA,
    )


def get_file_size_kb(path: Path):
    """Trả về kích thước file theo KB."""
    return path.stat().st_size / 1024


def save_jpeg(image, output_path: Path, quality: int, optimize=False):
    """Lưu JPEG với quality tùy chọn."""
    params = [
        cv2.IMWRITE_JPEG_QUALITY,
        quality,
    ]

    if optimize:
        params.extend([
            cv2.IMWRITE_JPEG_OPTIMIZE,
            1,
        ])

    success = cv2.imwrite(
        str(output_path),
        image,
        params,
    )

    if not success:
        raise RuntimeError(f"Không thể ghi ảnh: {output_path}")


def save_webp(image, output_path: Path, quality: int):
    """Lưu WebP lossy."""
    success = cv2.imwrite(
        str(output_path),
        image,
        [
            cv2.IMWRITE_WEBP_QUALITY,
            quality,
        ],
    )

    if not success:
        raise RuntimeError(f"Không thể ghi ảnh: {output_path}")


def save_png(image, output_path: Path):
    """Lưu PNG lossless để làm mốc so sánh."""
    success = cv2.imwrite(
        str(output_path),
        image,
        [
            cv2.IMWRITE_PNG_COMPRESSION,
            3,
        ],
    )

    if not success:
        raise RuntimeError(f"Không thể ghi ảnh: {output_path}")


def append_result(
    rows,
    source_name,
    method,
    output_path,
    original_size_kb,
    width,
    height,
):
    output_size_kb = get_file_size_kb(output_path)

    if original_size_kb > 0:
        reduction_percent = (
            1 - output_size_kb / original_size_kb
        ) * 100
    else:
        reduction_percent = 0

    rows.append({
        "source_image": source_name,
        "method": method,
        "output_file": str(output_path),
        "width": width,
        "height": height,
        "size_kb": round(output_size_kb, 2),
        "original_size_kb": round(original_size_kb, 2),
        "size_reduction_percent": round(reduction_percent, 2),
    })


# ============================================================
# XỬ LÝ MỘT ẢNH
# ============================================================

def process_image(image_path: Path, output_dir: Path, rows):
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)

    if image is None:
        print(f"⚠️ Không đọc được ảnh: {image_path.name}")
        return

    original_height, original_width = image.shape[:2]
    original_size_kb = get_file_size_kb(image_path)

    stem = image_path.stem

    print(
        f"\n🖼️ {image_path.name} "
        f"({original_width}x{original_height}, "
        f"{original_size_kb:.1f} KB)"
    )

    # --------------------------------------------------------
    # 0. ORIGINAL
    # Copy nguyên file để tiện so sánh trực tiếp
    # --------------------------------------------------------
    method_dir = output_dir / "00_original"
    method_dir.mkdir(parents=True, exist_ok=True)

    original_copy = method_dir / image_path.name
    shutil.copy2(image_path, original_copy)

    append_result(
        rows,
        image_path.name,
        "original",
        original_copy,
        original_size_kb,
        original_width,
        original_height,
    )

    # --------------------------------------------------------
    # 1. JPEG Quality 95 / 90 / 85 / 80
    # Giữ nguyên resolution
    # --------------------------------------------------------
    for quality in JPEG_QUALITIES:
        method_name = f"jpeg_q{quality}"
        method_dir = output_dir / method_name
        method_dir.mkdir(parents=True, exist_ok=True)

        output_path = method_dir / f"{stem}.jpg"

        save_jpeg(
            image,
            output_path,
            quality=quality,
            optimize=False,
        )

        append_result(
            rows,
            image_path.name,
            method_name,
            output_path,
            original_size_kb,
            original_width,
            original_height,
        )

    # --------------------------------------------------------
    # 2. JPEG Quality 90 + optimize
    # Giữ nguyên resolution
    # --------------------------------------------------------
    method_name = "jpeg_q90_optimized"
    method_dir = output_dir / method_name
    method_dir.mkdir(parents=True, exist_ok=True)

    output_path = method_dir / f"{stem}.jpg"

    save_jpeg(
        image,
        output_path,
        quality=90,
        optimize=True,
    )

    append_result(
        rows,
        image_path.name,
        method_name,
        output_path,
        original_size_kb,
        original_width,
        original_height,
    )

    # --------------------------------------------------------
    # 3. Resize max width 1920 + JPEG Q90 optimized
    #
    # Ảnh <= 1920px: giữ nguyên resolution.
    # Ảnh > 1920px : resize xuống width=1920.
    # --------------------------------------------------------
    resized = resize_max_width(
        image,
        max_width=MAX_KEYFRAME_WIDTH,
    )

    resized_height, resized_width = resized.shape[:2]

    method_name = "resize_1920_jpeg_q90_optimized"
    method_dir = output_dir / method_name
    method_dir.mkdir(parents=True, exist_ok=True)

    output_path = method_dir / f"{stem}.jpg"

    save_jpeg(
        resized,
        output_path,
        quality=90,
        optimize=True,
    )

    append_result(
        rows,
        image_path.name,
        method_name,
        output_path,
        original_size_kb,
        resized_width,
        resized_height,
    )

    # --------------------------------------------------------
    # 4. WebP Quality 90
    # Giữ nguyên resolution
    # --------------------------------------------------------
    method_name = "webp_q90"
    method_dir = output_dir / method_name
    method_dir.mkdir(parents=True, exist_ok=True)

    output_path = method_dir / f"{stem}.webp"

    save_webp(
        image,
        output_path,
        quality=WEBP_QUALITY,
    )

    append_result(
        rows,
        image_path.name,
        method_name,
        output_path,
        original_size_kb,
        original_width,
        original_height,
    )

    # --------------------------------------------------------
    # 5. Resize max width 1920 + WebP Q90
    # --------------------------------------------------------
    method_name = "resize_1920_webp_q90"
    method_dir = output_dir / method_name
    method_dir.mkdir(parents=True, exist_ok=True)

    output_path = method_dir / f"{stem}.webp"

    save_webp(
        resized,
        output_path,
        quality=WEBP_QUALITY,
    )

    append_result(
        rows,
        image_path.name,
        method_name,
        output_path,
        original_size_kb,
        resized_width,
        resized_height,
    )

    # --------------------------------------------------------
    # 6. PNG lossless
    #
    # Không khuyến nghị dùng cho keyframe video,
    # nhưng lưu thêm để làm baseline chất lượng lossless.
    # --------------------------------------------------------
    method_name = "png_lossless"
    method_dir = output_dir / method_name
    method_dir.mkdir(parents=True, exist_ok=True)

    output_path = method_dir / f"{stem}.png"

    save_png(
        image,
        output_path,
    )

    append_result(
        rows,
        image_path.name,
        method_name,
        output_path,
        original_size_kb,
        original_width,
        original_height,
    )


# ============================================================
# LƯU REPORT
# ============================================================

def save_report(rows, output_dir: Path):
    csv_path = output_dir / "comparison.csv"

    fieldnames = [
        "source_image",
        "method",
        "output_file",
        "width",
        "height",
        "size_kb",
        "original_size_kb",
        "size_reduction_percent",
    ]

    with open(
        csv_path,
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(rows)

    return csv_path


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Demo so sánh các phương án lưu/nén keyframe "
            "cho Object Detection và OCR."
        )
    )

    input_dir = Path("test_keyframes_momo")
    output_dir = Path("test_optimize_img_output")

    if not input_dir.exists():
        raise FileNotFoundError(
            f"Không tồn tại input folder: {input_dir}"
        )

    if not input_dir.is_dir():
        raise NotADirectoryError(
            f"Input không phải folder: {input_dir}"
        )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    image_files = get_image_files(input_dir)

    if not image_files:
        print(
            f"⚠️ Không tìm thấy ảnh hợp lệ trong: "
            f"{input_dir}"
        )
        return

    print("=" * 70)
    print("KEYFRAME COMPRESSION / RESIZE COMPARISON")
    print("=" * 70)
    print(f"Input : {input_dir.resolve()}")
    print(f"Output: {output_dir.resolve()}")
    print(f"Số ảnh: {len(image_files)}")
    print("=" * 70)

    rows = []

    for index, image_path in enumerate(
        image_files,
        start=1,
    ):
        print(
            f"\n[{index}/{len(image_files)}]",
            end="",
        )

        try:
            process_image(
                image_path,
                output_dir,
                rows,
            )

        except Exception as exc:
            print(
                f"❌ Lỗi xử lý "
                f"{image_path.name}: {exc}"
            )

    csv_path = save_report(
        rows,
        output_dir,
    )

    print("\n" + "=" * 70)
    print("✅ HOÀN TẤT")
    print(f"Output : {output_dir.resolve()}")
    print(f"Report : {csv_path.resolve()}")
    print("=" * 70)

    print(
        "\nCác folder được tạo:"
        "\n  00_original"
        "\n  jpeg_q95"
        "\n  jpeg_q90"
        "\n  jpeg_q85"
        "\n  jpeg_q80"
        "\n  jpeg_q90_optimized"
        "\n  resize_1920_jpeg_q90_optimized"
        "\n  webp_q90"
        "\n  resize_1920_webp_q90"
        "\n  png_lossless"
    )


if __name__ == "__main__":
    main()