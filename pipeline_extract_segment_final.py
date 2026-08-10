import tensorflow as tf
import sys
import os
import json
import cv2
import torch
import math
import shutil
import gc
from PIL import Image

import transformers.dynamic_module_utils as dynamic_utils
from transformers import AutoProcessor, AutoModelForCausalLM, PretrainedConfig
from sentence_transformers import SentenceTransformer, util

from huggingface_hub import snapshot_download

PretrainedConfig.forced_bos_token_id = None

orig_get_imports = dynamic_utils.get_imports
def custom_get_imports(filename):
    imports = orig_get_imports(filename)
    if imports is not None and "flash_attn" in imports:
        imports.remove("flash_attn")
    return imports
dynamic_utils.get_imports = custom_get_imports

# --- CONFIG PARAMETERS ---
HISTOGRAM_THRESHOLD = 0.90
JPEG_QUALITY = 85

SEGMENT_SIMILARITY_THRESHOLD = 0.4
KEYFRAME_MAX_INTERVAL = 20
MIN_KEYFRAMES_PER_SEGMENT = 3

# --- THAY TÊN FOLDER MUỐN CHẠY ---
TARGET_FOLDERS = ["Videos_L21", "Videos_L22", "Videos_L23", "Videos_L24", "Videos_L25", "Videos_L26"] 
# "Videos_L23", "Videos_L24", "Videos_L25", "Videos_L26", 
# "Videos_L27", "Videos_L28", "Videos_L29", "Videos_L30"
HF_REPO_ID = "lauralaurus/AIC2026"

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__)) # Mọi file output sẽ được lưu trong workspace nằm cùng cấp với file code này
BASE_WORKSPACE = os.path.join(CURRENT_DIR, "workspace") 
LOCAL_DATASET_DIR = os.path.join(BASE_WORKSPACE, "hf_downloaded_videos")
DRIVE_OUTPUT_FOLDER = os.path.join(BASE_WORKSPACE, "metadata")
DRIVE_KEYFRAMES_META_FOLDER = os.path.join(BASE_WORKSPACE, "keyframes_meta")
LOCAL_TEMP_FOLDER = "temp_processing_videos"

os.makedirs(LOCAL_DATASET_DIR, exist_ok=True)
os.makedirs(DRIVE_OUTPUT_FOLDER, exist_ok=True)
os.makedirs(DRIVE_KEYFRAMES_META_FOLDER, exist_ok=True)
os.makedirs(LOCAL_TEMP_FOLDER, exist_ok=True)


# --- IMPORT TRANSNETV2 ---
sys.path.append(os.path.join(os.getcwd(), 'TransNetV2', 'inference'))
from transnetv2 import TransNetV2

# --- KHỞI TẠO MÔ HÌNH ---
print("🚀 Đang khởi tạo và kiểm tra phần cứng cho các mô hình AI...")
if torch.cuda.is_available():
    device = "cuda:0"
    torch_dtype = torch.float16
    print(f"  ✅ [PyTorch] Đã nhận diện GPU: {torch.cuda.get_device_name(0)}")
else:
    device = "cpu"
    torch_dtype = torch.float32
    print("  ⚠️ [PyTorch] KHÔNG tìm thấy GPU!")

tf_gpus = tf.config.list_physical_devices('GPU')
if tf_gpus:
    print(f"  ✅ [TensorFlow] Đã nhận diện được GPU.")
else:
    print("  ⚠️ [TensorFlow] KHÔNG tìm thấy GPU! TransNetV2 sẽ chạy trên CPU.")
print("-" * 50)

print("1/3. Đang tải TransNetV2...")
transnet = TransNetV2()

florence_model_id = "microsoft/Florence-2-base-ft"
print(f"2/3. Đang tải {florence_model_id} trên {device}...")
florence_model = AutoModelForCausalLM.from_pretrained(florence_model_id, torch_dtype=torch_dtype, trust_remote_code=True).to(device).eval()
florence_processor = AutoProcessor.from_pretrained(florence_model_id, trust_remote_code=True)

print("3/3. Đang tải all-MiniLM-L6-v2...")
embedder = SentenceTransformer('all-MiniLM-L6-v2').to(device)
print("✅ TẢI MÔ HÌNH THÀNH CÔNG!\n")


# --- CÁC HÀM XỬ LÝ  ---
def extract_keyframes_from_shots(video_path, scenes):
    " Trích keyframe từ mỗi shot và trả về danh sách keyframes cùng fps của video "
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
    " Sinh caption cho ảnh bằng Florence-2 "
    prompt = "<MORE_DETAILED_CAPTION>"
    inputs = florence_processor(text=prompt, images=image, return_tensors="pt").to(device, torch_dtype)
    with torch.no_grad():
        generated_ids = florence_model.generate(
            input_ids=inputs["input_ids"], pixel_values=inputs["pixel_values"],
            max_new_tokens=1024, do_sample=False, num_beams=3
        )
    generated_text = florence_processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
    return florence_processor.post_process_generation(generated_text, task=prompt, image_size=(image.width, image.height))[prompt]

def group_shots_to_segments(shots, threshold=SEGMENT_SIMILARITY_THRESHOLD):
    " Gom các shot thành segment dựa trên độ tương đồng cosine của caption "
    if not shots: return []

    captions = [shot["caption"] for shot in shots]
    
    embeddings = embedder.encode(
        captions,
        convert_to_tensor=True
    )

    segments = []
    current_segment = {
        "segment_id": "seg_0001",
        "start_time": shots[0]["start_time"],
        "end_time": shots[0]["end_time"],
        "shots": [shots[0]]
    }
    
    seg_counter = 1

    for i in range(1, len(shots)):
        similarity = util.cos_sim(
            embeddings[i - 1],
            embeddings[i]
        ).item()

        if similarity >= threshold:
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
    
    # Định dạng cấu trúc segment
    for seg in segments:
        seg["segment_caption"] = " ".join([s["caption"] for s in seg["shots"]])
        seg["speech"] = []
        seg["keyframe"] = {}
    
        seg.pop("shots", None) 
            
    return segments

def save_tournament_winner(winner_data, base_name, seg_id, kf_meta_dir, seg_ref):
    """Hàm lưu frame xuất sắc nhất vào ổ cứng và update JSON"""
    frame_idx, frame_img = winner_data[0], winner_data[1]
    
    # Format: <vid>_<frameid>.jpg
    img_name = f"{base_name}_{frame_idx:05d}.jpg"
    img_path = os.path.join(kf_meta_dir, img_name)
    
    cv2.imwrite(
        img_path,
        frame_img,
        [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY]
    )
    
    seg_ref["keyframe"][img_name] = {
        "object": [],
        "ocr": []
    }

# --- MAIN FLOW ---
def process_all_videos():
    for folder_name in TARGET_FOLDERS:
        print(f"\n{'='*60}")
        print(f"📥 ĐANG TẢI VÀ KIỂM TRA DỮ LIỆU FOLDER: {folder_name}")
        print(f"{'='*60}")
        
        # 1. Tải folder vid từ Hugging Face về ổ cứng local
        try:
            folder_path = snapshot_download(
                repo_id=HF_REPO_ID,
                repo_type="dataset",
                allow_patterns=f"{folder_name}/*",
                local_dir=LOCAL_DATASET_DIR
            )
            target_input_folder = os.path.join(folder_path, folder_name)
        except Exception as e:
            print(f"❌ Lỗi khi tải dữ liệu từ Hugging Face cho {folder_name}: {e}")
            continue

        video_paths = []
        for root, dirs, files in os.walk(target_input_folder):
            for f in files:
                if f.endswith(('.mp4', '.avi', '.mkv')):
                    video_paths.append(os.path.join(root, f))

        if not video_paths:
            print(f"⚠️ Không tìm thấy video nào '{target_input_folder}'.")
            continue
        # TEST
        video_paths = video_paths[:2] 
        print(f"🧪 Chế độ TEST: Chỉ xử lý {len(video_paths)} video.")

        for vid_path in video_paths:
            video_file = os.path.basename(vid_path)
            folder_chua_vid = folder_name 

            # Lấy trực tiếp tên video gốc làm định danh (Ví dụ: L21_V001)
            base_name = os.path.splitext(video_file)[0]
                
            # Kiểm tra xem file JSON metadata đã tồn tại ở đích chưa
            out_json_dir = os.path.join(DRIVE_OUTPUT_FOLDER, folder_chua_vid)
            expected_json_path = os.path.join(out_json_dir, f"{base_name}.json")
            
            if os.path.exists(expected_json_path):
                print(f"⏩ [SKIP] Video {video_file} đã được xử lý (đã thấy file {base_name}.json). Bỏ qua...")
                continue # Nếu đã có file JSON metadata thì bỏ qua video này
            
            print(f"\n[{folder_chua_vid} / {video_file}] Đang bắt đầu xử lý...")
            local_video_path = os.path.join(LOCAL_TEMP_FOLDER, video_file)
            
            print(" ⏳ Đang copy video xuống SSD để tối ưu tốc độ...")
            shutil.copy2(vid_path, local_video_path)
            
            kf_meta_dir = os.path.join(DRIVE_KEYFRAMES_META_FOLDER, folder_chua_vid, f"{base_name}_keyframes")
            os.makedirs(kf_meta_dir, exist_ok=True)
            
            try:
                print("  -> Cắt shot bằng TransNetV2...")
                _, single_frame_predictions, _ = transnet.predict_video(local_video_path)
                scenes = transnet.predictions_to_scenes(single_frame_predictions)
                
                print(f"  -> Trích xuất ảnh đại diện ({len(scenes)} shots) để Florence đọc...")
                keyframes, fps = extract_keyframes_from_shots(local_video_path, scenes)
                
                print("  -> Sinh caption bằng Florence-2...")
                shots_data = []
                for kf in keyframes: 
                    caption = generate_caption(kf["image"])
                    shots_data.append({
                        "start_time": kf["start_time"], 
                        "end_time": kf["end_time"],
                        "caption": caption
                    })
                    
                print("  -> Gom cụm segment ")
                segments_data = group_shots_to_segments(shots_data, threshold=SEGMENT_SIMILARITY_THRESHOLD)
                
                print("  -> 📸 Đang trích xuất Keyframe (Lấy mẫu toán học -> Lọc Histogram)...")
                
                # =================================================================
                # BƯỚC 1: LẬP DANH SÁCH FRAME ỨNG CỬ VIÊN 
                # =================================================================
                target_frames_info = []
                
                for seg in segments_data:
                    if fps <= 0:
                        raise ValueError(f"Invalid FPS: {fps}")
                    start_frame = int(seg['start_time'] * fps)
                    end_frame = int(seg['end_time'] * fps)
                    total_frames = end_frame - start_frame
                    
                    if total_frames <= 0: continue
                    
                    # Công thức lấy keyframe trong segment: Tối thiểu 3 frame, max cách 20 frame
                    num_frames = max(MIN_KEYFRAMES_PER_SEGMENT, math.ceil(total_frames / KEYFRAME_MAX_INTERVAL) + 1)
                    step = total_frames // (num_frames - 1) if num_frames > 1 else 0
                    
                    for i in range(num_frames):
                        f_idx = min(start_frame + i * step, end_frame)
                        target_frames_info.append({
                            'frame_idx': f_idx,
                            'seg_id': seg['segment_id'],
                            'seg_ref': seg
                        })
                        
                # Sắp xếp mảng theo thứ tự thời gian để tiện cho việc đọc tuần tự 1 lần
                target_frames_info.sort(key=lambda x: x['frame_idx'])
                
                # =================================================================
                # BƯỚC 2: ĐỌC VIDEO VÀ ÁP DỤNG BỘ LỌC KÉP ĐỂ LOẠI BỎ ẢNH TRÙNG
                # =================================================================
                cap = cv2.VideoCapture(local_video_path)
                current_frame = 0
                target_idx = 0
                total_targets = len(target_frames_info)
                
                basket = []
                prev_hist = None
                current_seg_id = None
                
                while cap.isOpened() and target_idx < total_targets:
                    ret, frame = cap.read()
                    if not ret: break
                    
                    # Chỉ xử lý khi video chạy đến đúng frame nằm trong danh sách ứng cử viên
                    while target_idx < total_targets and current_frame == target_frames_info[target_idx]['frame_idx']:
                        info = target_frames_info[target_idx]
                        seg_id = info['seg_id']
                        seg_ref = info['seg_ref']
                        
                        # 1. Đo lường đặc trưng
                        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                        hist = cv2.calcHist([hsv], [0, 1], None, [50, 60], [0, 180, 0, 256])
                        cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)
                        
                        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                        lap_score = cv2.Laplacian(gray, cv2.CV_64F).var()
                        
                        # 2. Xả giỏ nếu thuật toán đã bước sang Segment mới
                        if current_seg_id != seg_id:
                            if basket:
                                winner = max(basket, key=lambda x: x[2])
                                save_tournament_winner(winner, base_name, current_seg_id, kf_meta_dir, winner[3])
                                basket = []
                            current_seg_id = seg_id
                            prev_hist = None
                            
                        # 3. Logic Lọc: Gom giỏ chờ & Loại trùng lặp
                        if not basket:
                            basket.append((current_frame, frame.copy(), lap_score, seg_ref))
                            prev_hist = hist
                        else:
                            sim = cv2.compareHist(prev_hist, hist, cv2.HISTCMP_CORREL)
                            if sim >= HISTOGRAM_THRESHOLD:
                                # Histogram giống nhau -> Thuộc cùng bối cảnh tĩnh -> Gom vào giỏ
                                basket.append((current_frame, frame.copy(), lap_score, seg_ref))
                                prev_hist = hist
                            else:
                                # Histogram thay đổi -> Có hành động mới -> Chốt giỏ cũ, chọn 1 tấm nét nhất
                                winner = max(basket, key=lambda x: x[2])
                                save_tournament_winner(winner, base_name, current_seg_id, kf_meta_dir, winner[3])
                                
                                # Cho frame hiện tại vào giỏ mới
                                basket = [(current_frame, frame.copy(), lap_score, seg_ref)]
                                prev_hist = hist
                                
                        target_idx += 1
                    
                    current_frame += 1
                    
                # Xả giỏ cuối cùng khi video kết thúc
                if basket:
                    winner = max(basket, key=lambda x: x[2])
                    save_tournament_winner(winner, base_name, current_seg_id, kf_meta_dir, winner[3])
                    
                cap.release()
                print(f"     ✅ Đã hoàn tất! Ứng cử viên được lấy theo quy tắc và đã qua bộ lọc trùng lặp.")
                
                print("  -> Lưu kết quả JSON Metadata...")
                
                # --- CẤU TRÚC JSON ---
                video_rel_path = os.path.join(folder_chua_vid, video_file).replace("\\", "/")
                
                final_data = {
                    "video_id": base_name,
                    "type": "video",
                    "video_path": video_rel_path,
                    "keyframes_folder_path": f"{folder_chua_vid}/{base_name}_keyframes", 
                    "metadata_path": f"{folder_chua_vid}/{base_name}.json",
                    "fps": round(fps, 2),
                    "segments": segments_data
                }
                
                out_json_dir = os.path.join(DRIVE_OUTPUT_FOLDER, folder_chua_vid)
                os.makedirs(out_json_dir, exist_ok=True)
                output_json_path = os.path.join(out_json_dir, f"{base_name}.json")
                
                with open(output_json_path, "w", encoding="utf-8") as f:
                    json.dump(final_data, f, ensure_ascii=False, indent=4)
                    
                print(f"🎉 Hoàn tất! File metadata đã được lưu: {output_json_path}")

            except Exception as e:
                print(f"❌ Lỗi khi xử lý {video_file}: {e}")
                
            finally:
                # 1. Dọn rác ổ cứng
                if os.path.exists(local_video_path):
                    os.remove(local_video_path)
                    print(" 🧹 Đã dọn dẹp file video tạm trên SSD.")
                    
                # 2. Ép Python thu gom các biến/tensor rác không dùng đến trong RAM
                gc.collect() 
                
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    
                print(" ♻️ Đã reset và giải phóng hoàn toàn VRAM & RAM. Sẵn sàng cho video tiếp theo!\n" + "="*50)

if __name__ == "__main__":
    process_all_videos()