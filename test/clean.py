import os
import json

# Trỏ đúng vào thư mục metadata của bạn
TARGET_FOLDER = r"G:\.shortcut-targets-by-id\11I5_AMfAufb6crT2hzGrLEI3tMsTsKjX\AIC2026\metadata\L28"

def clean_redundant_keys(folder_path):
    cleaned_count = 0
    skipped_count = 0
    
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.endswith(".json"):
                file_path = os.path.join(root, file)
                
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    
                    is_modified = False
                    
                    if "segments" in data:
                        for seg in data["segments"]:
                            # Xóa 'object' nếu nó tồn tại ở cấp độ segment
                            if "object" in seg:
                                del seg["object"]
                                is_modified = True
                            
                            # Xóa 'ocr' nếu nó tồn tại ở cấp độ segment
                            if "ocr" in seg:
                                del seg["ocr"]
                                is_modified = True
                    
                    # Lưu lại nếu có sự thay đổi
                    if is_modified:
                        with open(file_path, 'w', encoding='utf-8') as f:
                            json.dump(data, f, ensure_ascii=False, indent=4)
                        cleaned_count += 1
                        print(f"✅ Đã dọn dẹp thành công: {file}")
                    else:
                        skipped_count += 1
                        
                except Exception as e:
                    print(f"❌ Lỗi khi xử lý file {file}: {e}")
                    
    print("\n" + "="*50)
    print(f"🎉 TỔNG KẾT:")
    print(f" - Đã dọn dẹp (xóa key thừa): {cleaned_count} file.")
    print(f" - Bỏ qua (file đã sạch): {skipped_count} file.")
    print("="*50)

if __name__ == "__main__":
    print(f"Bắt đầu quét thư mục: {TARGET_FOLDER}\n")
    clean_redundant_keys(TARGET_FOLDER)