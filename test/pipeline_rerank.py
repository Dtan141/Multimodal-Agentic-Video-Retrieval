import os
import json
import numpy as np
# from sentence_transformers import CrossEncoder
# cross_model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

class RerankPipeline:
    def __init__(self, metadata_dir, fps=25.0):
        self.metadata_dir = metadata_dir
        self.fps = fps
        self.RRF_K = 60

    # ---------------------------------------------------------
    # BƯỚC 1 & 2: GỘP, XÓA TRÙNG LÂN CẬN VÀ TÍNH ĐIỂM RRF
    # ---------------------------------------------------------
    def merge_and_rrf(self, vector_results, metadata_results, frame_margin=3):
        merged_dict = {}

        def add_to_dict(results, source):
            for item in results:
                vid = item['video_id']
                fid = item['frame_id']
                rank = item['rank']
                
                # --- Xóa trùng lân cận (Temporal Deduplication) ---
                # Tìm xem trong khoảng +-3 frame đã có đại diện nào chưa
                found_duplicate = False
                for existing_fid in list(merged_dict.get(vid, {}).keys()):
                    if abs(existing_fid - fid) <= frame_margin:
                        # Nếu tìm thấy, lấy Rank tốt hơn (số nhỏ hơn)
                        if source == 'vector':
                            merged_dict[vid][existing_fid]['rank_vec'] = min(merged_dict[vid][existing_fid].get('rank_vec', float('inf')), rank)
                        else:
                            merged_dict[vid][existing_fid]['rank_meta'] = min(merged_dict[vid][existing_fid].get('rank_meta', float('inf')), rank)
                        found_duplicate = True
                        break
                
                # Nếu chưa có, tạo mới
                if not found_duplicate:
                    if vid not in merged_dict:
                        merged_dict[vid] = {}
                    if fid not in merged_dict[vid]:
                        merged_dict[vid][fid] = {'rank_vec': float('inf'), 'rank_meta': float('inf')}
                    
                    if source == 'vector':
                        merged_dict[vid][fid]['rank_vec'] = rank
                    else:
                        merged_dict[vid][fid]['rank_meta'] = rank

        add_to_dict(vector_results, 'vector')
        add_to_dict(metadata_results, 'meta')

        # --- Tính điểm RRF ---
        final_candidates = []
        for vid, frames in merged_dict.items():
            for fid, ranks in frames.items():
                rrf_score = 0
                if ranks['rank_vec'] != float('inf'):
                    rrf_score += 1.0 / (self.RRF_K + ranks['rank_vec'])
                if ranks['rank_meta'] != float('inf'):
                    rrf_score += 1.0 / (self.RRF_K + ranks['rank_meta'])
                
                final_candidates.append({
                    'video_id': vid,
                    'frame_id': fid,
                    'rrf_score': rrf_score
                })
        
        # Sắp xếp từ cao xuống thấp theo RRF
        final_candidates = sorted(final_candidates, key=lambda x: x['rrf_score'], reverse=True)
        return final_candidates

    # ---------------------------------------------------------
    # BƯỚC 3: ÁNH XẠ TIMESTEP & KHÔI PHỤC NGỮ CẢNH (HYDRATION)
    # ---------------------------------------------------------
    def hydrate_context(self, candidates, top_n=100):
        hydrated = []
        # Chỉ lấy Top N ứng viên tốt nhất để nạp Text
        for cand in candidates[:top_n]:
            vid = cand['video_id']
            fid = cand['frame_id']
            timestamp = fid / self.fps

            # Mở file JSON tương ứng
            json_path = os.path.join(self.metadata_dir, f"{vid}.json")
            if not os.path.exists(json_path):
                continue
                
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Quét tìm segment chứa timestamp
            target_segment = None
            for seg in data.get('segments', []):
                if seg['start_time'] <= timestamp <= seg['end_time']:
                    target_segment = seg
                    break
            
            if target_segment:
                cand['segment_id'] = target_segment['segment_id']
                cand['segment_caption'] = target_segment.get('segment_caption', '')
                # Có thể lấy thêm OCR, Object ở đây nếu cần
                hydrated.append(cand)
                
        return hydrated

    # ---------------------------------------------------------
    # BƯỚC 4: LỌC THÔ BẰNG CROSS-ENCODER (PRE-RERANK)
    # ---------------------------------------------------------
    def cross_encoder_rerank(self, hydrated_candidates, query, top_segments=5):
        # 1. Chấm điểm Cross-Encoder
        # model_inputs = [[query, cand['segment_caption']] for cand in hydrated_candidates]
        # cross_scores = cross_model.predict(model_inputs)
        
        # Mô phỏng điểm số (Xóa phần này khi ráp model thật)
        cross_scores = [0.9] * len(hydrated_candidates) 

        for i, cand in enumerate(hydrated_candidates):
            cand['cross_score'] = cross_scores[i]
            # Tie-breaking: Cộng thêm 10% điểm RRF để phá băng
            cand['final_score'] = cand['cross_score'] + (0.1 * cand['rrf_score'])

        # 2. Gộp nhóm theo Segment để nộp cho Qwen
        segment_groups = {}
        for cand in hydrated_candidates:
            seg_key = f"{cand['video_id']}_{cand['segment_id']}"
            if seg_key not in segment_groups:
                segment_groups[seg_key] = {
                    'video_id': cand['video_id'],
                    'segment_id': cand['segment_id'],
                    'segment_caption': cand['segment_caption'],
                    'highest_final_score': cand['final_score'],
                    'frames': [] # Danh sách các frame thuộc segment này
                }
            # Cập nhật điểm cao nhất của segment
            if cand['final_score'] > segment_groups[seg_key]['highest_final_score']:
                segment_groups[seg_key]['highest_final_score'] = cand['final_score']
            
            segment_groups[seg_key]['frames'].append(cand['frame_id'])

        # 3. Lấy Top 5 Segments xuất sắc nhất
        sorted_segments = sorted(list(segment_groups.values()), key=lambda x: x['highest_final_score'], reverse=True)
        return sorted_segments[:top_segments]

    # ---------------------------------------------------------
    # BƯỚC 5: CHỐT HẠ BẰNG VLM (QWEN)
    # ---------------------------------------------------------
    def qwen_final_rerank(self, top_segments, query):
        """
        Input: Top 5 segments (chứa danh sách frame_id)
        Logic: 
        1. Tải ảnh vật lý của các frame_id.
        2. Ghép ảnh + Query nạp vào Qwen2.5-VL.
        3. Parse kết quả trả về ID chính xác nhất.
        """
        print(f"🔥 Đưa {len(top_segments)} segments vào Qwen để chốt hạ...")
        # TODO: Cài đặt code Qwen2.5-VL-3B tại đây
        
        best_frame_id = None 
        return best_frame_id

# ==========================================
# CÁCH SỬ DỤNG PIPELINE
# ==========================================
if __name__ == "__main__":
    # Dữ liệu mô phỏng từ 2 luồng Search
    vec_res = [{'video_id': 'L28_V001', 'frame_id': 150, 'rank': 1}]
    meta_res = [{'video_id': 'L28_V001', 'frame_id': 152, 'rank': 2}]
    user_query = "Người đàn ông cưỡi ngựa màu nâu"

    pipeline = RerankPipeline(metadata_dir="metadata_folder/")
    
    # Bước 1 & 2
    merged_cands = pipeline.merge_and_rrf(vec_res, meta_res)
    
    # Bước 3
    hydrated_cands = pipeline.hydrate_context(merged_cands, top_n=100)
    
    # Bước 4
    top_5_segments = pipeline.cross_encoder_rerank(hydrated_cands, user_query)
    
    # Bước 5
    final_answer = pipeline.qwen_final_rerank(top_5_segments, user_query)