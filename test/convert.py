import os
import json

# Đường dẫn tới thư mục chứa các file metadata JSON cũ
# (Hãy thay đổi đường dẫn này trỏ tới thư mục L28 hoặc metadata của bạn)
TARGET_FOLDER = r"G:\.shortcut-targets-by-id\11I5_AMfAufb6crT2hzGrLEI3tMsTsKjX\AIC2026\metadata\L28"

def convert_metadata_format(folder_path):
    converted_count = 0
    skipped_count = 0
    
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.endswith(".json"):
                file_path = os.path.join(root, file)
                
                try:
                    # Đọc file JSON
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    
                    is_modified = False
                    
                    # Kiểm tra xem file có chứa mảng "segments" hay không
                    if "segments" in data:
                        for seg in data["segments"]:
                            # Kiểm tra nếu 'keyframe' đang là list (cấu trúc cũ)
                            if "keyframe" in seg and isinstance(seg["keyframe"], list):
                                new_keyframe_dict = {}
                                
                                # Duyệt qua từng tên file ảnh trong list cũ
                                for img_name in seg["keyframe"]:
                                    new_keyframe_dict[img_name] = {
                                        "object": [],
                                        "ocr": []
                                    }
                                
                                # Cập nhật lại thuộc tính keyframe bằng Dictionary mới
                                seg["keyframe"] = new_keyframe_dict
                                is_modified = True
                    
                    # Nếu có thay đổi, ghi đè lại định dạng mới vào đúng file đó
                    if is_modified:
                        with open(file_path, 'w', encoding='utf-8') as f:
                            json.dump(data, f, ensure_ascii=False, indent=4)
                        converted_count += 1
                        print(f"✅ Đã chuyển đổi thành công: {file}")
                    else:
                        skipped_count += 1
                        
                except Exception as e:
                    print(f"❌ Lỗi khi xử lý file {file}: {e}")
                    
    print("\n" + "="*50)
    print(f"🎉 TỔNG KẾT:")
    print(f" - Đã chuyển đổi: {converted_count} file.")
    print(f" - Bỏ qua (đã đúng format từ trước): {skipped_count} file.")
    print("="*50)

if __name__ == "__main__":
    print(f"Bắt đầu quét thư mục: {TARGET_FOLDER}\n")
    convert_metadata_format(TARGET_FOLDER)