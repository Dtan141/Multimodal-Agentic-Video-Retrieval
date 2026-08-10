import os
import json
import torch
import numpy as np
import ollama
import time
import re
import math
from PIL import Image
from rank_bm25 import BM25Okapi
from transformers import AutoProcessor, AutoModel
from sentence_transformers import CrossEncoder
from siglip2_embedder import Siglip2Embedder

class DualStreamSearcher:
    @staticmethod
    def get_unique_key(video_id, filename):
        # Sửa lỗi double prefix L01_L01_ nếu có
        if video_id.startswith("L01_L01_"):
            video_id = video_id.replace("L01_L01_", "L01_", 1)
        
        # Lấy số ID cuối cùng
        stem = filename.split('.')[0]
        pure_fid = int(stem.split('_')[-1])
        
        return f"{video_id}_{pure_fid}"
    
    def __init__(self, vector_npz_path, vector_jsonl_path, metadata_dir, device="cuda:0"):
        self.device = device
        print("🚀 Khởi tạo Dual-Stream Searcher...")
        
        # 1. LOAD LUỒNG VECTOR (SigLIP)
        print(" -> Đang nạp Vector Embeddings...")
        npz_data = np.load(vector_npz_path)
        self.image_vectors = npz_data['vectors']
        
        self.vector_metadata = []
        with open(vector_jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                self.vector_metadata.append(json.loads(line))
                
        print(" -> Đang tải mô hình SigLIP 2 thông qua Wrapper...")
        self.siglip = Siglip2Embedder(device=self.device, use_fp16=True)

        # 2. LOAD LUỒNG METADATA (BM25)
        print(" -> Đang nạp JSON Metadata và xây dựng BM25...")
        self.segments_data = [] # Lưu thông tin các segment
        self.frame_to_caption = {} # Dictionary ánh xạ: Tên ảnh -> Caption
        
        for json_file in os.listdir(metadata_dir):
            if not json_file.endswith('.json'): continue
            with open(os.path.join(metadata_dir, json_file), 'r', encoding='utf-8') as f:
                data = json.load(f)
                video_id = data.get('video_id', '')
                
                for seg in data.get('segments', []):
                    seg_caption = seg.get('segment_caption', '').strip()
                    if not seg_caption: continue
                    
                    self.segments_data.append({
                        'video_id': video_id,
                        'segment_id': seg['segment_id'],
                        'caption': seg_caption,
                        'frames': list(seg.get('keyframe', {}).keys())
                    })
                    
                    # Ánh xạ phục vụ Cross-Encoder sau này
                    for frame_name in seg.get('keyframe', {}).keys():
                        unique_key = self.get_unique_key(video_id, frame_name)
                        self.frame_to_caption[unique_key] = seg_caption

        # Tokenize (tách từ) để chạy BM25
        tokenized_corpus = [seg['caption'].lower().split(" ") for seg in self.segments_data]
        self.bm25 = BM25Okapi(tokenized_corpus)
        
        # 3. LOAD CROSS-ENCODER
        print(" -> Đang tải Cross-Encoder...")
        self.cross_model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2', device=self.device)
        print("✅ HỆ THỐNG ĐÃ SẴN SÀNG!\n")

    # ==========================================
    # CÁC HÀM TÌM KIẾM CỐT LÕI
    # ==========================================
    def search_vector_stream(self, query, top_k=100):
        # 1. Mã hóa câu Query thành vector
        query_vec = self.siglip.encode_texts([query]).cpu().numpy()[0]
        
        # 2. Tính Cosine Similarity bằng Dot Product (nhờ L2 Norm)
        scores = np.dot(self.image_vectors, query_vec)
        
        # 3. Lấy Top K
        top_indices = np.argsort(scores)[::-1][:top_k]
        
        results = []
        for rank, idx in enumerate(top_indices):
            meta = self.vector_metadata[idx]
            results.append({
                'frame_name': meta['filename'],
                'video_id': meta['video_id'],
                'score': float(scores[idx]),
                'rank': rank + 1
            })
        return results

    def search_metadata_stream(self, query, top_k=100):
        # 1. Chạy BM25
        tokenized_query = query.lower().split(" ")
        bm25_scores = self.bm25.get_scores(tokenized_query)
        
        # 2. Lấy Top Segments
        top_seg_indices = np.argsort(bm25_scores)[::-1]
        
        # 3. Trải phẳng các frames từ các Segment thắng cuộc (chỉ lấy cho đến khi đủ Top K)
        results = []
        current_rank = 1
        
        for idx in top_seg_indices:
            if bm25_scores[idx] <= 0: break # Bỏ qua các segment không chứa từ khóa nào
            seg = self.segments_data[idx]
            
            for frame_name in seg['frames']:
                results.append({
                    'frame_name': frame_name,
                    'video_id': seg['video_id'],
                    'score': float(bm25_scores[idx]),
                    'rank': current_rank
                })
                current_rank += 1
                if len(results) >= top_k:
                    return results
        return results

    # ==========================================
    # PIPELINE GỘP & RERANK
    # ==========================================
    def run_pipeline(self, query, rrf_k=60, cross_top_n=20):
        print(f"🔍 Đang truy vấn: '{query}'")
        
        # --- CHẶNG 1: TÌM KIẾM VECTOR & METADATA (Đo riêng rẽ) ---
        # 1.1 Đo luồng Vector (SigLIP)
        t_vec_start = time.time()
        vec_res = self.search_vector_stream(query, top_k=100)
        t_vec_end = time.time()
        t_vector = t_vec_end - t_vec_start
        print(f"⏱️ [1.1] Thời gian Search Vector (SigLIP): {t_vector:.4f}s")
        
        # 1.2 Đo luồng Metadata (BM25)
        t_meta_start = time.time()
        meta_res = self.search_metadata_stream(query, top_k=100)
        t_meta_end = time.time()
        t_meta = t_meta_end - t_meta_start
        print(f"⏱️ [1.2] Thời gian Search Metadata (BM25): {t_meta:.4f}s")
        
        # (Tùy chọn) In tổng chặng 1
        print(f"⏱️ Tổng thời gian truy xuất thô: {t_vector + t_meta:.4f}s")

        print(f"\n[DEBUG] Vector Stream returned {len(vec_res)} frames:")
        print([item['frame_name'] for item in vec_res[:10]]) # Chỉ in Top 10 đầu để không làm tràn màn hình
        
        print(f"\n[DEBUG] Metadata Stream returned {len(meta_res)} frames:")
        print([item['frame_name'] for item in meta_res[:10]])
        
        # --- CHẶNG 2: GỘP & RRF ---
        t1 = time.time()
        merged_dict = {}
        # Bước 2: Gộp bằng RRF
        merged_dict = {}

        
        def add_to_rrf(results, source):
            for item in results:
                pure_fid = self.get_unique_key(item['video_id'], item['frame_name']).split('_')[-1]
                vid = item['video_id']
                
                # Sửa lỗi dư thừa "L01_L01_" của luồng vector nếu có
                if vid.startswith("L01_L01_"):
                    vid = vid.replace("L01_L01_", "L01_", 1)
                    
                # Tạo khóa duy nhất: VD "L01_V001_0"
                unique_key = f"{vid}_{pure_fid}"
                
                if unique_key not in merged_dict:
                    merged_dict[unique_key] = {
                        'rank_vec': float('inf'), 
                        'rank_meta': float('inf'), 
                        'video_id': vid,
                        'pure_frame_id': pure_fid,
                        'frame_name': item['frame_name'], # Giữ lại 1 tên gốc để sau này copy ảnh
                        'source': source # Đánh dấu nó nằm ở thư mục nào (Vector hay Metadata)
                    }
                
                if source == 'vector':
                    merged_dict[unique_key]['rank_vec'] = item['rank']
                else:
                    merged_dict[unique_key]['rank_meta'] = item['rank']

        add_to_rrf(vec_res, 'vector')
        add_to_rrf(meta_res, 'meta')
        
        # Tính điểm RRF
        rrf_candidates = []
        for unique_key, data in merged_dict.items(): # Đổi tên biến fname thành unique_key cho dễ hiểu
            score = 0
            if data['rank_vec'] != float('inf'): score += 1.0 / (rrf_k + data['rank_vec'])
            if data['rank_meta'] != float('inf'): score += 1.0 / (rrf_k + data['rank_meta'])
            
            rrf_candidates.append({
                'frame_name': data['frame_name'], 
                'video_id': data['video_id'],
                'pure_frame_id': data['pure_frame_id'],
                'source': data['source'],
                'rrf_score': score
            })
            
        rrf_candidates = sorted(rrf_candidates, key=lambda x: x['rrf_score'], reverse=True)[:100] # Lấy Top 100 RRF
        t_rrf = time.time() - t1
        print(f"⏱️ [2/3] Thời gian Gộp và tính RRF: {t_rrf:.4f}s")

        # --- CHẶNG 3: HYDRATION & CROSS-ENCODER ---
        t2 = time.time()
        # Bước 3: Hydration & Cross-Encoder
        print(" -> Đang chạy Cross-Encoder Rerank...")
        cross_inputs = []
        valid_candidates = []
        
        for cand in rrf_candidates:
            # Tìm caption tương ứng từ dictionary. Nếu ảnh từ nhánh Vector không có trong JSON thì bỏ qua hoặc dùng text rỗng.
            unique_key = f"{cand['video_id']}_{cand['pure_frame_id']}"
            caption = self.frame_to_caption.get(unique_key, "")
            if caption:
                cross_inputs.append([query, caption])
                valid_candidates.append(cand)
                
        if not cross_inputs:
            print("⚠️ Không có frame nào map được với Caption để chạy Cross-Encoder.")
            return []
            
        cross_scores = self.cross_model.predict(cross_inputs)
        
        # Gộp điểm phá băng (Tie-breaking: Cross + 10% RRF)
        for i, cand in enumerate(valid_candidates):
            cand['cross_score'] = float(cross_scores[i])
            cand['final_score'] = cand['cross_score'] + (0.1 * cand['rrf_score'])
            
        # Bước 4: Chốt Top cuối cùng để giao cho Qwen
        final_candidates = sorted(valid_candidates, key=lambda x: x['final_score'], reverse=True)
        t_cross = time.time() - t2
        print(f"⏱️ [3/3] Thời gian Cross-Encoder chấm điểm: {t_cross:.4f}s")
        return final_candidates[:cross_top_n]

def create_image_grid(image_paths, grid_size=None):
    # Tự động tính số cột/hàng (VD: 10 ảnh -> 4x3 lưới)
    num_images = len(image_paths)
    cols = grid_size if grid_size else math.ceil(math.sqrt(num_images))
    rows = math.ceil(num_images / cols)
    
    # Kích thước mỗi ảnh nhỏ (resize để tiết kiệm VRAM)
    thumb_size = (256, 256)
    grid_img = Image.new('RGB', (cols * thumb_size[0], rows * thumb_size[1]))
    
    for i, path in enumerate(image_paths):
        img = Image.open(path).resize(thumb_size)
        grid_img.paste(img, ((i % cols) * thumb_size[0], (i // cols) * thumb_size[1]))
        
    grid_path = "temp_grid.jpg"
    grid_img.save(grid_path, quality=85)
    return grid_path
# ==========================================
# CÁCH CHẠY THỬ NGHIỆM
# ==========================================
if __name__ == "__main__":
    # TODO: Cập nhật đường dẫn thật trên máy bạn
    VECTOR_NPZ = "vector_embeddings/L01_embeddings.npz"
    VECTOR_JSONL = "vector_embeddings/L01_metadata.jsonl"
    METADATA_DIR = "metadata/L01" # Thư mục chứa các file .json

    VECTOR_IMAGE_DIR = "test_keyframe_vector_momo\\L01_V001_vector_frames"     # Thư mục chứa ảnh từ luồng vector
    META_IMAGE_DIR = "test_keyframes_momo"     # Thư mục chứa ảnh từ luồng metadata
    
    searcher = DualStreamSearcher(VECTOR_NPZ, VECTOR_JSONL, METADATA_DIR)
    
    query = "A man standing next to a white horse in a green field"
    
    # 2. CHẠY TÌM KIẾM VÀ RERANK (Lọc xuống Top 10)
    top_frames = searcher.run_pipeline(query, cross_top_n=10)
    
    print(f"\n🎯 KẾT QUẢ TOP {len(top_frames)} ỨNG VIÊN TỪ CROSS-ENCODER:")
    for i, frame in enumerate(top_frames):
        print(f"Top {i+1}: [{frame['source'].upper()}] {frame['frame_name']} | Final Score: {frame['final_score']:.4f}")

    # ==========================================
    # 3. CHỐT HẠ BẰNG QWEN2.5-VL (QUA OLLAMA)
    # ==========================================
    TOP_K_VLM = 5
    
    print(f"\n🚀 CHUYỂN GIAO TOP {TOP_K_VLM} ỨNG VIÊN CHO QWEN2.5-VL (OLLAMA)...")
    
    image_paths = []
    valid_frames = []
    
    # Gom đường dẫn ảnh thật
    for frame in top_frames[:TOP_K_VLM]:
        # 💡 CHIA NHÁNH ĐỌC FILE DỰA VÀO NGUỒN GỐC
        if frame['source'] == 'vector':
            img_path = os.path.join(VECTOR_IMAGE_DIR, frame['frame_name'])
        else:
            img_path = os.path.join(META_IMAGE_DIR, frame['frame_name'])
            
        if os.path.exists(img_path):
            image_paths.append(img_path)
            valid_frames.append(frame)
        else:
            print(f"⚠️ Bỏ qua: Không tìm thấy ảnh vật lý tại {img_path}")
    grid_img_path = create_image_grid(image_paths)

    instruction = (
        f"I have provided an image grid containing {TOP_K_VLM} candidate frames in path {grid_img_path} "
        f"The frames are arranged in a grid from top-left to bottom-right, numbered 1 to {TOP_K_VLM}. "
        f"User query: '{query}'\n"
        "Identify which frame number (1-%d) best matches the query. "
        "Return only the number."
    )
        
    messages = [{
        'role': 'user',
        'content': instruction,
        'images': image_paths
    }]
    
    vlm_start_time = time.time()
    try:
        print("🧠 Đang suy luận...")
        response = ollama.chat(
            model='qwen2.5vl:3b',
            messages=messages
        )
        output_text = response['message']['content'].strip()
        
        print("\n" + "="*50)
        print("🏆 KẾT QUẢ CHUNG CUỘC TỪ QWEN2.5-VL")
        print("="*50)
        print(f"Câu trả lời thô: {output_text}")
        
        # Trích xuất số ID thực tế từ câu trả lời của VLM
        try:
            chosen_id = int(''.join(filter(str.isdigit, output_text)))
            if 1 <= chosen_id <= len(valid_frames):
                chosen_frame = valid_frames[chosen_id - 1]
                print(f"-> QWEN ĐÃ CHỌN ẢNH SỐ {chosen_id}: {chosen_frame['frame_name']}")
            else:
                print(f"-> Qwen trả về số {chosen_id}, nằm ngoài phạm vi 1-{len(valid_frames)}.")
        except ValueError:
            print("-> Không trích xuất được số từ câu trả lời của Qwen.")
            
        print(f"⏱️ Thời gian Qwen suy luận: {time.time() - vlm_start_time:.3f} giây")
        print("="*50 + "\n")
        
    except Exception as e:
        print(f"❌ Lỗi khi gọi Ollama API: {e}")
