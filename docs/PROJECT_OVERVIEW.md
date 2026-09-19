# AIC 2026 — Mô tả dự án chi tiết (ôn tập phỏng vấn)

> Tài liệu này mô tả toàn bộ kiến trúc, từng module, và luồng hoạt động của hệ thống
> truy xuất multimedia. Dùng để ôn tập và trả lời phỏng vấn. Viết theo hướng "cái gì —
> vì sao — đánh đổi", kèm gợi ý câu hỏi phỏng vấn ở cuối mỗi phần.

---

## 1. Bài toán & tổng quan

**Bài toán:** cho một kho video truyền hình lớn (~873 video, ~409K keyframe), người dùng
nhập truy vấn bằng ngôn ngữ tự nhiên (thường tiếng Việt) để tìm **đúng keyframe/khoảnh khắc**
trả lời truy vấn, rồi xuất file nộp theo thể thức thi (KIS / Q&A / TRAKE). Thể thức tương tự
Lifelog Search Challenge (LSC) / Video Browser Showdown (VBS).

**Ý tưởng cốt lõi:** kết hợp 2 hướng tìm kiếm bổ trợ nhau:
- **Semantic search** — hiểu *hình ảnh* (SigLIP2 nhúng text & ảnh vào cùng không gian → tìm bằng cosine).
- **Metadata search** — hiểu *nội dung phi hình ảnh trên khung hình*: chữ trên màn hình (OCR),
  lời nói (ASR), vật thể (object detection), mô tả cảnh (caption) — tìm bằng BM25.

Hai luồng được **hợp nhất (RRF)**, và có **agent LLM** tự phân tích truy vấn để quyết định
dùng những công cụ (tool) nào.

**Kiến trúc tổng thể:** tách **backend FastAPI** (Python package `aic_retrieval`) + **frontend Next.js**.
Backend nạp toàn bộ index **một lần** lúc khởi động (FAISS + BM25 + embedder), phục vụ qua REST API;
frontend gọi API qua proxy `/api/*`.

**Điểm phỏng vấn:** *"Vì sao hybrid?"* → semantic mạnh về ngữ nghĩa hình ảnh nhưng mù chữ/âm thanh;
metadata bắt được tên riêng, số, chữ chạy, lời thoại mà ảnh không mã hoá tốt. Bổ trợ nhau → recall cao hơn.

---

## 2. Chuẩn bị dữ liệu (offline pipeline)

Dữ liệu keyframe/metadata được sinh sẵn ngoài hệ thống này; backend chỉ **ingest + index**.
Có **HAI tập keyframe khác nhau** sinh từ 2 phương pháp lọc keyframe khác nhau **trên cùng video gốc**:
- **Tập vector (semantic):** dùng để nhúng SigLIP2 → FAISS. Ảnh ở HF `keyframe_vector/…`.
- **Tập metadata:** dùng để gắn OCR/ASR/object/caption. Ảnh ở HF `keyframe_metadata/…`.
→ Frame_id của 2 tập **khác nhau**; khi fuse dùng `(video_id, frame_id)` + khử trùng lân cận thời gian.

Các bước CLI (`aic …`):
| Lệnh | Việc |
|---|---|
| `download-metadata` | Tải JSON metadata/video từ HF (`Chillguy2026/AIC_2026_metadata`) |
| `analyze-metadata` | Thống kê tần suất theo folder → gợi ý stoplist OCR (lọc watermark) |
| `clean-metadata` | Làm sạch: bỏ speech rác (`keep=false`), lọc token OCR rác (htv/online/json…) |
| `ingest-metadata` | Sinh **4 index BM25** (ocr/asr/object/caption) + bảng hydration (segments, fps) |
| `build-index` | Build **FAISS** + `frames.parquet` từ các `.npz` embedding |

**Điểm phỏng vấn:** *"Xử lý nhiễu OCR thế nào?"* → phát hiện token xuất hiện trên ~80% keyframe của
một folder (watermark kênh) bằng **tỷ lệ keyframe** (không phải chỉ document-frequency, để không xoá nhầm
từ nội dung tiếng Việt phổ biến), cộng danh sách curated (htv, online, tuoitretv, và artifact model `json/bbox_2d`).

---

## 3. Chi tiết từng module (backend `aic_retrieval`)

### 3.1 `config.py` — cấu hình tập trung
- Dùng **pydantic-settings**. Thứ tự ưu tiên: init kwargs > env (`AIC_*`, và secret `GROQ_API_KEY`) > `.env` > `configs/default.yaml` > default.
- Thay toàn bộ hằng số hardcode + đường dẫn Windows của code cũ. Secret không nằm trong repo.
- **Vì sao:** một nguồn cấu hình duy nhất, đổi môi trường không sửa code; test override dễ.

### 3.2 `logging_conf.py`
- `setup_logging()` idempotent + `get_logger`. Thay `print` rải rác.

### 3.3 `embedding/siglip2.py` — `Siglip2Embedder`
- Model `google/siglip2-base-patch16-224`; `get_text_features`/`get_image_features`, fp16 trên CUDA.
- **L2-normalize** đầu ra → cosine = dot product. Text được lowercase, pad `max_length`.
- **Vì sao SigLIP2:** huấn luyện contrastive kiểu sigmoid, nhúng ảnh–text cùng không gian, hỗ trợ đa ngữ tốt hơn CLIP.
- **Điểm phỏng vấn:** *"Cosine vs dot?"* → sau L2-norm hai đại lượng bằng nhau, nên FAISS `IndexFlatIP` (inner product) = cosine.

### 3.4 `index/faiss_index.py` — chỉ mục vector
- `IndexFlatIP` (brute-force **chính xác**). Vì vector đã chuẩn hoá → IP = cosine → **trùng khớp** kết quả brute-force numpy của code cũ, nhưng nạp 1 lần & bền.
- Hỗ trợ **`IDSelector`** để giới hạn tìm trong tập row được phép (dùng cho lọc phạm vi video).
- **Đánh đổi:** ~409K vector thì FlatIP đủ nhanh (semantic query ~0.1s). Corpus lớn hơn → chuyển HNSW/IVF-PQ (đã chừa chỗ).

### 3.5 `index/metadata_store.py` — bảng metadata frame
- `frames.parquet` căn hàng **1-1** với FAISS (row i ↔ vector i): `video_id, frame_id, filename, source_folder, folder_id, global_frame_id`.
- Nạp nhanh, ít RAM; cột `video_id`/`folder_id` phục vụ lọc phạm vi.

### 3.6 `index/build.py` — build index
- Đọc từng `.npz` (`vectors`+`frame_ids`) + `.jsonl` metadata, **khử trùng** theo `global_frame_id`, add vào FAISS **theo từng shard** (tiết kiệm RAM), lưu FAISS + parquet.

### 3.7 `text/ingest_metadata.py` + `text/bm25_index.py` — tìm kiếm từ khoá
- Ingest JSON → **4 corpus BM25 độc lập**: `caption`, `asr` (mức segment → map ra mọi keyframe của segment), `ocr`, `object` (mức keyframe).
- `ModuleBM25`: tokenizer unicode giữ dấu tiếng Việt; `search(query, top_k, keep=…)` xếp hạng document rồi **trải phẳng ra frame**, khử trùng, có predicate `keep(video_id)` để lọc phạm vi ngay khi trải phẳng.
- Ghi thêm `segments.parquet` + `fps.json` cho hydration.
- **Vì sao BM25:** nhẹ, không cần GPU, mạnh cho khớp từ khoá/tên riêng; IDF tự hạ trọng số từ phổ biến.

### 3.8 `preprocess/analyze.py` + `preprocess/clean.py` — làm sạch dữ liệu
- `analyze`: theo từng folder, xếp hạng token OCR theo **tỷ lệ keyframe** để phát hiện watermark; xuất report + stoplist gợi ý.
- `clean`: bỏ trường nhiễu trong speech + speech `keep=false`; lọc token OCR theo stoplist (curated ∪ auto); ghi ra `metadata_clean/`.
- **Kết quả:** OCR docs 193K → 116K (loại ~77K keyframe chỉ toàn watermark), ASR 45K → 42K.

### 3.9 `search/` — lõi tìm kiếm
- `semantic.py` `SemanticSearcher`: query → SigLIP2 → FAISS top-k (nhận `allowed_ids` cho lọc phạm vi).
- `lexical.py` `LexicalSearcher`: bọc 4 `ModuleBM25`, `search(module, query, keep)`.
- `fusion.py` `rrf_fuse`: **Reciprocal Rank Fusion** trên nhiều nguồn + **khử trùng lân cận thời gian** (±N frame) → gộp frame gần nhau từ các nhánh.
- `hydration.py` `Hydrator`: `frame_id / fps → timestamp → segment` để gắn `segment_id` + `caption` (dùng `segments.parquet`+`fps.json`, tra cứu bằng bisect).
- `rerank.py` `CrossEncoderReranker`: cross-encoder `ms-marco-MiniLM-L-6-v2` chấm `[query, caption]`, `final = cross + 0.1·rrf`. **Mặc định TẮT** (opt-in) — hệ cũ chạy tốt không cần; để dành nâng cấp cùng Qwen-VLM (Phase 2).
- `video_filter.py` `VideoScope`: khớp spec `L21_V018` (video) hoặc `L21` (folder, tự nhận diện theo `_V`); `allowed_row_ids` (mask vector hoá cho semantic) + `keep_fn` (predicate cho lexical). Mô hình **allowlist (include) + denylist (exclude)**.
- `service.py` `SearchService`: **điểm điều phối duy nhất** (API & agent dùng chung):
  - `semantic_search`, `module_search`, `hybrid_search`.
  - `hybrid_search`: semantic **tùy chọn** (điền gì tìm nấy) → gộp các nguồn bằng RRF → hydrate → (tùy chọn) rerank; áp lọc phạm vi cho cả 2 nhánh.

### 3.10 `agent/` — trợ lý phân tích truy vấn
- `llm_client.py`: client **OpenAI-compatible** trỏ **Groq** (`qwen/qwen3.8-27b`), JSON mode + retry/backoff cho rate limit free tier.
- `router.py`: **prompt-based JSON routing** (không phụ thuộc native tool-calling): LLM nhận query → trả JSON `{semantic_query, modules:{ocr,asr,object,caption}, reasoning}`. Dịch phần **thị giác sang tiếng Anh** cho SigLIP2/caption/object; **giữ tiếng Việt** cho ocr/asr. Có fallback về semantic-only nếu LLM lỗi.
- `orchestrator.py` `AgentSearcher`: route → gọi `hybrid_search` với plan → trả kết quả + **reasoning trace**. **Semantic luôn chạy** (bắt buộc); các module do agent chọn.
- **Điểm phỏng vấn:** *"Vì sao prompt-based routing thay vì function-calling?"* → model free trên Groq không đảm bảo tool-calling; JSON có cấu trúc + parse phòng thủ chạy bền với mọi model.

### 3.11 `hf_assets.py` — ảnh & video
- **Ảnh keyframe:** `get_frame_bytes` thử root theo nguồn hit (`semantic→keyframe_vector`, `meta→keyframe_metadata`, qua `?set=`), rồi root kia, cuối cùng **trích đúng frame từ video gốc bằng OpenCV** (fallback cho frame không có ảnh). **Cache 2 tầng:** `hf_hub_download` (đĩa, bền) + `lru_cache` thumbnail 384px (RAM). Thumbnail ~14KB thay 94KB.
- **Video:** `get_video_path` (tải HF, cache) + `get_video_info` (fps/frame_count qua cv2). Phục vụ byte-range để tua frame-accurate.
- **Vì sao:** trước đây mỗi thumbnail = 1 HTTPS tới HF (~2s, không cache). Sau tối ưu: ảnh đã cache ~1ms.

### 3.12 `submission.py` — xuất file nộp
- `build_submission_csv(query_type, rows, event_count)` cho KIS/Q&A/TRAKE; UTF-8, phẩy, không header, ≤100 dòng; Q&A ≤100 ký tự; TRAKE cảnh báo frame không tăng dần.

### 3.13 `api/` — REST (FastAPI)
- `main.py`: **lifespan** nạp `SearchService` (+ agent nếu có `GROQ_API_KEY`) **một lần**; bật CORS cho Next.js.
- `routes.py`: `/health`, `/search`, `/search/hybrid`, `/agent/search`, `/image` (+`?full`,`?set`), `/video/{id}` (+`/info`), `/submission/validate`, `/prefetch`, `/modules`. `_enrich` gắn `image_path` (kèm hint root) cho mỗi kết quả.
- `schemas.py`: pydantic request/response.

### 3.14 `cli.py` — `aic …`
- `info, download-metadata, analyze-metadata, clean-metadata, ingest-metadata, build-index, serve, agent`.

---

## 4. Frontend (Next.js + TypeScript + Tailwind)
- Bố cục **2 cột**: trái (sticky) = truy vấn + lọc phạm vi + tạo CSV; phải = lưới kết quả.
- 2 chế độ: **Agent** (chỉ nhập query, hiện reasoning) và **Thủ công** (chọn module + từ khoá; semantic tùy chọn).
- `components/`: `ResultGrid` (lưới + checkbox chọn), `VideoInspector` (player frame-accurate + bảng CSV sống + ±5s), `CsvBuilder` (KIS/Q&A/TRAKE, sinh theo dải, answer chung, tên file).
- `lib/api.ts`: client gọi backend; `prefetch` tải trước thumbnail top-k.

---

## 5. Luồng xử lý một truy vấn (rất hay bị hỏi)

**Chế độ thủ công (hybrid):**
1. Frontend gửi `POST /search/hybrid {query?, modules{...}, include/exclude}`.
2. `SearchService.hybrid_search`: nếu có query → semantic (FAISS, có IDSelector nếu lọc); mỗi module có từ khoá → BM25 (có `keep`).
3. `rrf_fuse` gộp các nguồn + khử trùng lân cận thời gian.
4. `Hydrator` gắn caption/segment/timestamp.
5. `_enrich` gắn `image_path` → trả kết quả; frontend `prefetch` warm ảnh; hiện lưới.

**Chế độ agent:** thêm bước đầu — `router` (Groq) phân tích query → plan (semantic_query tiếng Anh + module + từ khoá) → phần còn lại như hybrid; trả kèm reasoning trace.

**Xem & nộp:** bấm keyframe → `VideoInspector` phát video (byte-range) đúng frame → tick "dùng frame" → bảng CSV → `POST /submission/validate` → tải CSV.

---

## 6. Hiệu năng & kỹ thuật đáng nói khi phỏng vấn
- Semantic query ~**0.1s** trên 409K vector (FAISS FlatIP, nạp 1 lần vs code cũ nạp lại mỗi phiên).
- Ảnh: cache đĩa+RAM + thumbnail 384px + prefetch → ~**1ms** khi đã ấm (trước ~2s/ảnh, mọi lần).
- Lọc phạm vi semantic **chính xác** bằng FAISS `IDSelector` (không over-fetch).
- Chất lượng kỹ thuật: config layer, logging, **44 test (pytest)**, ruff, CI (GitHub Actions), Docker/compose.

## 7. Hạn chế & hướng phát triển (Phase 2)
- Rerank cross-encoder đang tắt mặc định; kế hoạch: **Qwen2.5-VL** rerank nhìn ảnh thật.
- Truy vấn **chuỗi/tuần tự** nhiều cảnh (temporal) cho TRAKE.
- Khử near-duplicate keyframe (pHash+DBSCAN); SigLIP2 biến thể lớn hơn.
- Hai tập keyframe khác nhau → fusion chéo nhánh ít trùng trong ±N frame (có thể tăng margin/fusion theo segment).

---

## 8. Bộ câu hỏi tự kiểm tra
- Vì sao tách backend/frontend? Lifespan nạp index 1 lần giải quyết vấn đề gì của code cũ?
- FAISS FlatIP khác brute-force numpy chỗ nào? Khi nào phải đổi ANN?
- RRF là gì, vì sao chọn thay vì cộng điểm thô? Khử trùng lân cận thời gian để làm gì?
- Agent quyết định module bằng cách nào mà không cần function-calling? Xử lý query tiếng Việt ra sao?
- Vì sao có 2 tập keyframe, ảnh phục vụ từ đâu, fallback trích video khi nào?
- Lọc phạm vi video hoạt động thế nào ở nhánh semantic (IDSelector) vs lexical (predicate)?
- Làm sạch OCR: vì sao dùng tỷ lệ keyframe thay vì document-frequency?
