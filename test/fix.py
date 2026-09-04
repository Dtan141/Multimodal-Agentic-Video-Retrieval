import os
import json

# Đường dẫn tới thư mục metadata
TARGET_FOLDER = r"G:\.shortcut-targets-by-id\11I5_AMfAufb6crT2hzGrLEI3tMsTsKjX\AIC2026\metadata\L28"

def add_new_paths(folder_path):
    updated_count = 0
    
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.endswith(".json"):
                file_path = os.path.join(root, file)
                
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    
                    # Lấy tên video từ trường video_id (ví dụ: "L28_V001")
                    # Nếu không có sẵn, sẽ lấy tên file JSON bỏ đuôi .json
                    video_id = data.get("video_id", file.replace(".json", ""))
                    
                    # Tạo dictionary mới để gom các thông tin chung lên đầu
                    new_data = {}
                    
                    # 1. Giữ nguyên các key cơ bản ở đầu
                    for key in ["video_id", "type", "video_path", "fps"]:
                        if key in data:
                            new_data[key] = data[key]
                            
                    # 2. Thêm 2 trường mới theo yêu cầu
                    new_data["keyframes_folder_path"] = f"L28/{video_id}_keyframes"
                    new_data["metadata_path"] = f"L28/{video_id}.json"
                    
                    # 3. Đưa các trường còn lại (bao gồm mảng 'segments' khổng lồ) xuống cuối
                    for key, value in data.items():
                        if key not in new_data:
                            new_data[key] = value
                            
                    # Ghi đè lại file
                    with open(file_path, 'w', encoding='utf-8') as f:
                        json.dump(new_data, f, ensure_ascii=False, indent=4)
                        
                    updated_count += 1
                    print(f"✅ Đã thêm trường cho file: {file}")
                        
                except Exception as e:
                    print(f"❌ Lỗi khi xử lý file {file}: {e}")
                    
    print("\n" + "="*50)
    print(f"🎉 TỔNG KẾT: Đã cập nhật thành công {updated_count} file JSON.")
    print("="*50)

if __name__ == "__main__":
    print(f"Bắt đầu cập nhật metadata: {TARGET_FOLDER}\n")
    add_new_paths(TARGET_FOLDER)