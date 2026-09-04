import tensorflow as tf
import sys
import os
import json
import cv2
import torch
import math
import numpy as np
from PIL import Image

# --- BƯỚC LỌC AN TOÀN CHO FLORENCE-2 ---
import transformers
import transformers.dynamic_module_utils as dynamic_utils
from transformers import AutoProcessor, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer, util

# Lưu lại hàm gốc của transformers
orig_get_imports = dynamic_utils.get_imports

# Tạo hàm mới: Vẫn chạy hàm gốc nhưng lén xóa flash_attn khỏi kết quả
def custom_get_imports(filename):
    imports = orig_get_imports(filename)
    if imports is not None and "flash_attn" in imports:
        imports.remove("flash_attn")
    return imports

# Ghi đè hàm mới vào hệ thống
dynamic_utils.get_imports = custom_get_imports
# -----------------------------------------

# --- CẤU HÌNH ĐƯỜNG DẪN DỮ LIỆU ---
INPUT_FOLDER = "videos"       # Thư mục chứa video đầu vào
OUTPUT_FOLDER = "metadata"    # Thư mục chứa JSON đầu ra

os.makedirs(INPUT_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# --- THÊM CẤU HÌNH DEBUG Ở ĐẦU FILE  ---
DEBUG_FOLDER = "debug_outputs"
os.makedirs(DEBUG_FOLDER, exist_ok=True)

# Thêm TransNetV2 vào đường dẫn hệ thống (đảm bảo folder TransNetV2 nằm cùng thư mục gốc)
sys.path.append(os.path.join(os.getcwd(), 'TransNetV2', 'inference'))
from transnetv2 import TransNetV2

# --- KHỞI TẠO MÔ HÌNH ---
print("🚀 Đang khởi tạo và kiểm tra phần cứng cho các mô hình AI...")

# 1. KIỂM TRA GPU CHO PYTORCH (Dành cho Florence-2 & MiniLM)
if torch.cuda.is_available():
    device = "cuda:0"
    torch_dtype = torch.float16
    gpu_name = torch.cuda.get_device_name(0)
    print(f"  ✅ [PyTorch] Đã nhận diện GPU: {gpu_name}")
else:
    device = "cpu"
    torch_dtype = torch.float32
    print("  ⚠️ [PyTorch] KHÔNG tìm thấy GPU! Florence-2 sẽ chạy trên CPU (Cảnh báo: Rất chậm!)")

# 2. KIỂM TRA GPU CHO TENSORFLOW (Dành cho TransNetV2)
tf_gpus = tf.config.list_physical_devices('GPU')
if tf_gpus:
    print(f"  ✅ [TensorFlow] Đã nhận diện được {len(tf_gpus)} GPU cho TransNetV2.")
else:
    print("  ⚠️ [TensorFlow] KHÔNG tìm thấy GPU! TransNetV2 sẽ chạy trên CPU.")
print("-" * 50)

# 1. TransNetV2
print("1/3. Đang tải TransNetV2...")
transnet = TransNetV2()

# 2. Florence-2
# Lưu ý: Chuyển sang bản large-ft để sinh caption mượt và chi tiết hơn
florence_model_id = "microsoft/Florence-2-large-ft" 
print(f"2/3. Đang tải {florence_model_id} trên {device}...")
florence_model = AutoModelForCausalLM.from_pretrained(
    florence_model_id, 
    torch_dtype=torch_dtype, 
    trust_remote_code=True
).to(device).eval()
florence_processor = AutoProcessor.from_pretrained(florence_model_id, trust_remote_code=True)

# # 3. Sentence Transformer (BGE-M3)
# print("3/3. Đang tải BAAI/bge-m3...")
# embedder = SentenceTransformer('BAAI/bge-m3').to(device)

# 3. Sentence Transformer (all-MiniLM-L6-v2)
print("3/3. Đang tải all-MiniLM-L6-v2...")
embedder = SentenceTransformer('all-MiniLM-L6-v2').to(device)

print("✅ TẢI MÔ HÌNH THÀNH CÔNG RỰC RỠ!\n")


# --- CÁC HÀM XỬ LÝ CỐT LÕI ---
def extract_keyframes_from_shots(video_path, scenes):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    keyframes = []
    
    for shot_idx, (start_frame, end_frame) in enumerate(scenes):
        mid_frame = (start_frame + end_frame) // 2
        cap.set(cv2.CAP_PROP_POS_FRAMES, mid_frame)
        ret, frame = cap.read()
        
        if ret:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(frame_rgb)
            keyframes.append({
                "shot_id": f"shot_{shot_idx+1:04d}",
                "start_time": round(start_frame / fps, 2),
                "end_time": round(end_frame / fps, 2),
                "image": pil_img
            })
    cap.release()
    return keyframes, fps

def generate_caption(image):
    prompt = "<MORE_DETAILED_CAPTION>"
    inputs = florence_processor(text=prompt, images=image, return_tensors="pt").to(device, torch_dtype)
    
    with torch.no_grad():
        generated_ids = florence_model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=1024,
            do_sample=False,
            num_beams=3
        )
    
    generated_text = florence_processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
    parsed_answer = florence_processor.post_process_generation(generated_text, task=prompt, image_size=(image.width, image.height))
    return parsed_answer[prompt]

def group_shots_to_segments(shots, threshold=0.75):
    if not shots: return []

    segments = []
    current_segment = {
        "segment_id": "seg_0001",
        "start_time": shots[0]["start_time"],
        "end_time": shots[0]["end_time"],
        "shots": [shots[0]]
    }
    
    seg_counter = 1
    for i in range(1, len(shots)):
        prev_caption = shots[i-1]["caption"]
        curr_caption = shots[i]["caption"]
        
        embeddings = embedder.encode([prev_caption, curr_caption], convert_to_tensor=True)
        cosine_score = util.cos_sim(embeddings[0], embeddings[1]).item()
        
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
    
    # Gom caption, dọn dẹp biến image khỏi bộ nhớ RAM để xuất JSON
    for seg in segments:
        seg["segment_caption"] = " ".join([s["caption"] for s in seg["shots"]])
        for s in seg["shots"]:
            s.pop("image", None) 
            
    return segments

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

# --- LUỒNG CHẠY CHÍNH ---
# ----------------------------------------------------------------
def process_all_videos():
    video_files = [f for f in os.listdir(DRIVE_INPUT_FOLDER) if f.endswith(('.mp4', '.avi', '.mkv'))]
    
    if not video_files:
        print(f"⚠️ Không tìm thấy video nào trên Drive '{DRIVE_INPUT_FOLDER}'.")
        return

    for video_file in video_files:
        print(f"\n[{video_file}] Đang bắt đầu xử lý...")
        drive_video_path = os.path.join(DRIVE_INPUT_FOLDER, video_file)
        local_video_path = os.path.join(LOCAL_TEMP_FOLDER, video_file)
        video_id = os.path.splitext(video_file)[0]
        
        # --- BƯỚC 0: KÉO VIDEO TỪ DRIVE XUỐNG LOCAL SSD ---
        print(" ⏳ Đang copy video từ Drive xuống máy tính để xử lý nhanh...")
        shutil.copy2(drive_video_path, local_video_path)
        print(" ✅ Copy hoàn tất!")
        
        # Khởi tạo thư mục lưu kết quả thẳng lên Drive
        vid_debug_dir = os.path.join(DRIVE_DEBUG_FOLDER, video_id)
        os.makedirs(vid_debug_dir, exist_ok=True)
        kf_debug_dir = os.path.join(vid_debug_dir, "keyframes")
        os.makedirs(kf_debug_dir, exist_ok=True)
        
        try:
            # ---> LƯU Ý: Chuyển video_path thành local_video_path để đọc <---
            print("  -> Cắt shot bằng TransNetV2...")
            video_frames, single_frame_predictions, all_frame_predictions = transnet.predict_video(local_video_path)
            scenes = transnet.predictions_to_scenes(single_frame_predictions)
            
            with open(os.path.join(vid_debug_dir, "01_shot_frames.txt"), "w") as f:
                for idx, (s, e) in enumerate(scenes):
                    f.write(f"Shot {idx+1:03d}: Frame {s} -> {e}\n")
            
            print(f"  -> Trích xuất keyframes ({len(scenes)} shots)...")
            keyframes, fps = extract_keyframes_from_shots(local_video_path, scenes)
            
            # Lưu ảnh thẳng lên Drive
            for kf in keyframes:
                img_path = os.path.join(kf_debug_dir, f"{kf['shot_id']}.jpg")
                kf["image"].save(img_path)
            
            print("  -> Sinh caption bằng Florence-2...")
            shots_data = []
            
            for kf in keyframes: 
                caption = generate_caption(kf["image"])
                shots_data.append({
                    "shot_id": kf["shot_id"],
                    "start_time": kf["start_time"],
                    "end_time": kf["end_time"],
                    "image": kf["image"],
                    "caption": caption
                })
                
            debug_shots_data = [{k: v for k, v in s.items() if k != 'image'} for s in shots_data]
            with open(os.path.join(vid_debug_dir, "02_raw_captions.json"), "w", encoding="utf-8") as f:
                json.dump(debug_shots_data, f, ensure_ascii=False, indent=4)
                
            print("  -> Gom cụm segment (Threshold = 0.4)...")
            segments_data = group_shots_to_segments(shots_data, threshold=0.4)
            
            print("  -> Lưu kết quả JSON cuối cùng lên Drive...")
            final_data = {
                "video_id": video_id,
                "video_path": f"drive_id_or_path_here", # Cập nhật sau nếu cần
                "fps": fps,
                "segments": segments_data
            }
            
            output_json_path = os.path.join(DRIVE_OUTPUT_FOLDER, f"{video_id}.json")
            with open(output_json_path, "w", encoding="utf-8") as f:
                json.dump(final_data, f, ensure_ascii=False, indent=4)
                
            print(f"🎉 Hoàn tất! File đã được lưu tự động lên Drive: {output_json_path}")

        except Exception as e:
            print(f"❌ Lỗi khi xử lý {video_file}: {e}")
            
        finally:
            # --- BƯỚC CUỐI: XÓA FILE LOCAL ĐỂ DỌN RÁC ---
            if os.path.exists(local_video_path):
                os.remove(local_video_path)
                print(" 🧹 Đã xóa file video tạm trên máy.")

if __name__ == "__main__":
    process_all_videos()