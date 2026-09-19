# Nâng cấp hệ thống Truy xuất Multimedia AIC 2026 — Milestone Nền tảng (v2)

> Bản v2 cập nhật theo yêu cầu: **frontend Next.js** (bỏ Streamlit) + **tầng Agent** điều phối search.

## Context (Vì sao làm việc này)

Hệ thống hiện tại là **codebase nghiên cứu/thi đấu** chạy được nhưng chưa chuẩn "sản phẩm thật" và còn kìm hãm thành tích:

- **Search brute-force**: `normalized_vectors @ query_vector` trên toàn bộ vector nạp RAM mỗi phiên (`AIC2026_retrieval_pipeline.py:1393-1397`) — không index, không scale.
- **Hybrid search + rerank chỉ là prototype rời** (`test/rerank.py`, `test/pipeline_rerank.py` — có stub `0.9`, import gãy `siglip2_embedder_v1`), chưa nối vào app.
- **Metadata search trong app chỉ khớp từ khóa chính xác** (`test/new_pipeline.py`), chưa dùng OCR/ASR/Object.
- **Nợ kỹ thuật**: không API/backend, không config layer (hardcode + path Windows `C:\`/`G:\`), không test/Docker/CI/logging, 3 bản app Streamlit trùng lặp.

### Quyết định đã chốt
- Mục tiêu: **cân bằng** chất lượng kỹ thuật (CV) + thành tích thi đấu.
- Kiến trúc: **tách backend (FastAPI) + frontend (Next.js + TypeScript + Tailwind + shadcn/ui)**.
- **Agent điều phối search** (mới): 2 chế độ — thủ công 2 nhánh & agent tự phân tích query.
- Agent LLM: **Groq free tier** (OpenAI-compatible, `base_url=https://api.groq.com/openai/v1`, key qua env `GROQ_API_KEY`), **routing bằng JSON có cấu trúc** (không phụ thuộc native tool-calling). Lưu ý free tier có giới hạn RPM/TPM → cần retry/backoff.
- Metadata: **mỗi module = 1 tool riêng** (`ocr_search`, `asr_search`, `object_search`, `caption_search`); **semantic search luôn bắt buộc**.
- OCR/ASR/Object đã chạy sẵn bên ngoài → ta chỉ **ingest + index**, không code lại module trích xuất.
- Phạm vi: **milestone nền tảng trước** (temporal search, query tiếng Việt, Qwen-VLM rerank để Phase 2).

### Dữ liệu (thực tế trên máy)
- **Embeddings**: `database/vector_embeddings/` — `.npz` (`vectors`+`frame_ids`+meta) + `.jsonl` (frame metadata).
  - ✅ **Cả 10 tập L21–L30 đều hợp lệ, tổng ≈ 409.715 vector (768 chiều)**. Các `.npz` là **symlink** trỏ tới `~/working/prj/AIC2026/data/hf_vectors/` (đọc xuyên symlink bình thường). Số vector từng tập: L21=30.052, L22=34.364, L23=8.123, L24=20.289, L25=116.469, L26=132.087, L27=7.874, L28=22.767, L29=20.545, L30=17.145.
  - → Đường dẫn embeddings để **cấu hình được** (trỏ thư mục symlink hoặc thư mục thật đều được).
- **Metadata OCR/ASR/Object/Caption**: tải từ HF **private** `Chillguy2026/AIC_2026_metadata` (token HF đã có sẵn trên máy).
- ⚠️ Chưa có venv cho project (chỉ có `~/venvs/ppg-spo2` không liên quan) → WS1 tạo venv mới.
- Path trong `.jsonl` vẫn là Windows `C:\MyD\...` → chuẩn hóa khi ingest.

---

## Kiến trúc đích

```
repo/
├── backend/  (package Python: aic_retrieval)
│   ├── pyproject.toml, Dockerfile
│   ├── configs/default.yaml           # paths, model id, thresholds, HF repo, LLM (Groq base_url + model)
│   └── src/aic_retrieval/
│       ├── config.py                  # pydantic-settings (env + yaml), BỎ hardcode path
│       ├── logging_conf.py
│       ├── embedding/siglip2.py       # nguồn duy nhất Siglip2Embedder
│       ├── index/
│       │   ├── faiss_index.py         # IndexFlatIP (exact); ngỏ HNSW/IVF-PQ
│       │   └── metadata_store.py      # frame table: video_id, frame_id, filename, timestamp, segment_id
│       ├── text/
│       │   ├── ingest_metadata.py     # tải HF + parse JSON → docs theo TỪNG module
│       │   └── bm25_index.py          # index riêng: ocr / asr / object / caption
│       ├── search/
│       │   ├── semantic.py            # FAISS vector search (luôn chạy)
│       │   ├── lexical.py             # BM25 theo module
│       │   ├── fusion.py              # RRF (+ temporal dedup)
│       │   └── rerank.py              # cross-encoder (Qwen-VLM = hook Phase 2)
│       ├── agent/                     # ⭐ MỚI
│       │   ├── llm_client.py          # OpenAI-compatible (Groq base_url, model, GROQ_API_KEY)
│       │   ├── router.py              # prompt → JSON: chọn module + trích từ khóa mỗi module
│       │   ├── tools.py               # registry: semantic_search + 4 metadata tool riêng
│       │   └── orchestrator.py        # semantic(bắt buộc) + tool được chọn → fuse → rerank → trace
│       ├── submission.py              # KIS/Q&A/TRAKE (chuyển từ app)
│       ├── hf_assets.py               # fetch ảnh / tải video từ HF
│       ├── api/
│       │   ├── main.py                # FastAPI + lifespan nạp index 1 lần + CORS
│       │   ├── schemas.py
│       │   └── routes.py              # /health /search /search/hybrid /agent/search /image /video /submission
│       └── cli.py                     # typer: download-metadata, build-index, ingest-metadata, serve
├── frontend/  (Next.js + TS + Tailwind + shadcn/ui)   # ⭐ MỚI
│   ├── app/  (search thủ công · agent chat · video inspector · CSV builder)
│   ├── components/  (ResultGrid, VideoInspector, CsvBuilder, AgentPanel, ModeToggle)
│   └── lib/api.ts
├── docker-compose.yml   # service: api + frontend
└── tests/
```

Giữ nguyên code cũ, xây song song rồi dời dần. Sau khi API ổn, bỏ 3 bản app Streamlit.

---

## Milestone nền tảng — workstream

### WS1 — Khung backend + config + chất lượng cơ bản
- Tạo venv mới; `pyproject.toml` (giữ pin torch cu118/transformers 4.44.2; thêm faiss, fastapi, uvicorn, pydantic-settings, typer; dev: pytest, ruff).
- `config.py` (pydantic-settings): paths, `MODEL_ID`, `TOP_K`, thresholds, HF repo, **LLM Groq base_url/model + `GROQ_API_KEY` (env, không commit)**. Xóa path Windows ở `upload_hf_vector.py:12`, `vector_embeddings/embed_vector.py:18`, `test/convert.py|clean.py|fix.py`, `test/retroactive_filter.py:8`.
- `logging_conf.py`; sửa import gãy `siglip2_embedder_v1` → `aic_retrieval.embedding.siglip2`; bỏ `import gc` trùng.

### WS2 — FAISS index + metadata store (thay brute-force)
- `faiss_index.py`: đọc `.npz` (`vectors`+`frame_ids`) → **`IndexFlatIP`** (vector đã L2-norm → dot = cosine, **exact**), persist đĩa; `search(qvec, top_k)`.
- `metadata_store.py`: bảng frame (parquet/SQLite): `row_id → {video_id, frame_id, filename, source_folder, global_frame_id, timestamp}`; chuẩn hóa path. Tái dùng dedup `_frame_key` ở `AIC2026_retrieval_pipeline.py:154-281`.
- ✅ Đủ 10 tập (~410K vector) để build index toàn bộ ngay.

### WS3 — Tải + ingest metadata HF theo TỪNG module + BM25
- `cli download-metadata`: dùng `huggingface_hub` (token sẵn có) tải `Chillguy2026/AIC_2026_metadata` về local.
- `ingest_metadata.py`: parse JSON mỗi video → tách văn bản **theo module**: `caption`, `ocr`, `asr(speech)`, `object`; gắn `video_id/frame_id/segment_id/start_time/end_time`. **Xác minh schema thật từ 1 file tải về** (đối chiếu `README.md`).
- `bm25_index.py`: **4 index BM25 độc lập** (ocr/asr/object/caption), persist; tái dùng build + trải phẳng frame ở `rerank.py:50-131`.

### WS4 — Search core
- `semantic.py`: query → SigLIP2 text embed → FAISS top-k.
- `lexical.py`: `search_module(module, keywords, top_k)` chạy BM25 của đúng module → trải phẳng ra frame.
- `fusion.py`: **RRF** từ `rerank.py:170-217` + **temporal dedup ±N** từ `pipeline_rerank.py:16-71` + hydrate frame→timestamp→segment `pipeline_rerank.py:76-105`.
- `rerank.py`: cross-encoder `cross-encoder/ms-marco-MiniLM-L-6-v2` (bỏ stub), `final = cross + 0.1*rrf` (`rerank.py:238-243`); hook `qwen_final_rerank` (Phase 2).

### WS5 — ⭐ Tầng Agent (điều phối)
- `llm_client.py`: client **OpenAI-compatible** trỏ **Groq** (`https://api.groq.com/openai/v1/chat/completions`), model qua config (vd `llama-3.3-70b-versatile`), key `GROQ_API_KEY`; timeout + retry/backoff cho giới hạn free tier.
- `router.py`: **prompt-based JSON routing** — LLM nhận query, trả JSON kiểu:
  ```json
  {"semantic_query": "...", "modules": {"ocr": ["từ khóa"], "object": ["person","horse"], "asr": [], "caption": ["..."]}}
  ```
  Có validate/parse chống JSON hỏng; nếu lỗi → fallback chỉ semantic.
- `tools.py`: registry 5 tool — `semantic_search` + `ocr_search`/`asr_search`/`object_search`/`caption_search` (mỗi tool gọi `lexical.search_module`).
- `orchestrator.py`: **luôn chạy semantic** + các module agent chọn → `fusion.RRF` → `rerank` → trả kết quả kèm **reasoning trace** (đã chọn module gì, từ khóa gì) để hiển thị & log.

### WS6 — FastAPI backend
- `api/main.py`: **lifespan** nạp FAISS + metadata store + 4 BM25 + embedder + cross-encoder **một lần**; bật **CORS** cho Next.js.
- Endpoints:
  - `GET /health`
  - `POST /search` (semantic thuần)
  - `POST /search/hybrid` (chế độ **thủ công**: client tự chọn module + từ khóa → RRF + rerank)
  - `POST /agent/search` (chế độ **agent**: chỉ query → router → orchestrator → kết quả + trace)
  - `GET /image` (proxy ảnh HF, từ `fetch_private_hf_image:103`)
  - `GET /video` (byte-range, từ `_VideoAssetHandler`/`download_video_from_hf:369`)
  - `POST /submission/validate` (dùng `build_submission_csv:1253`)

### WS7 — ⭐ Frontend Next.js (thay Streamlit)
- Next.js + TS + Tailwind + shadcn/ui; `lib/api.ts` gọi backend.
- **ModeToggle**: (a) Thủ công 2 nhánh — chọn module + nhập từ khóa; (b) Agent — chỉ nhập query, hiển thị **reasoning trace**.
- **ResultGrid**: lưới ảnh top-k (ảnh qua `/image`), score, rank.
- **VideoInspector**: port player HTML5 frame-accurate (`render_custom_video_browser:639`) sang React component (giữ byte-range seek, overlay Frame ID, nhảy frame).
- **CsvBuilder**: trang tạo KIS/Q&A/TRAKE gọi `/submission/validate`, tải CSV.

### WS8 — Test + CI + Docker + README
- `tests/`: unit `fusion` (RRF/temporal dedup), `router` (parse JSON + fallback), `submission`, `config`; smoke test API (`TestClient`, FAISS nhỏ giả lập).
- `.github/workflows/ci.yml`: ruff + pytest (backend) + `next build`/lint (frontend).
- `Dockerfile` (backend) + Next.js build; `docker-compose.yml` (api + frontend).
- Viết lại `README.md`: kiến trúc, setup, run, sơ đồ 2 chế độ + agent.

---

## Phase 2 (KHÔNG làm lần này)
- **Qwen2.5-VL rerank** qua Ollama (hook có sẵn; `rerank.py:314-356`).
- **Temporal/sequential search** (query chuỗi cảnh A→B).
- **Query tiếng Việt** (dịch VN→EN hoặc đa ngữ SigLIP2) — có thể tận dụng luôn agent LLM để dịch/chuẩn hóa.
- **Khử near-duplicate keyframe** (pHash+DBSCAN, `test/pipeline_extract_kf_vector_final.py`).
- Nâng SigLIP2 lên biến thể lớn hơn/so400m.

---

## Verification (đầu-cuối)
1. `pip install -e ".[dev]"` trong venv mới.
2. `python -m aic_retrieval.cli download-metadata` → tải JSON từ HF; log số video.
3. `python -m aic_retrieval.cli build-index` → FAISS + metadata store; số vector khớp tổng `.npz` (hiện = L26).
4. `python -m aic_retrieval.cli ingest-metadata` → 4 BM25 index; log số doc mỗi module.
5. `uvicorn ...api.main:app` → `/health`; `POST /search` top-k giảm dần; `POST /search/hybrid` (chọn module) & `POST /agent/search` (chỉ query) trả kết quả + trace.
6. So top-10 `/search` với brute-force cũ trên L26 → FAISS IndexFlatIP **trùng khớp**.
7. `npm run dev` (frontend) → chuyển 2 chế độ, xem grid, mở VideoInspector, tạo & tải CSV.
8. `pytest` xanh; `ruff` sạch; `docker compose up` chạy cả api + frontend.

## Cần bạn xác nhận / cung cấp
- **`GROQ_API_KEY` + model name** (vd `llama-3.3-70b-versatile`) để điền env/config; key không commit vào repo.
- (Tùy chọn) xác nhận repo HF metadata đúng là `Chillguy2026/AIC_2026_metadata` và có quyền đọc bằng token hiện tại.
