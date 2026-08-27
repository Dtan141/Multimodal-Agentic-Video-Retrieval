# AIC2026 Retrieval — optimized layout

## Mục tiêu

Bản này giữ retrieval exact bằng SigLIP2 nhưng chuyển các việc nặng ra khỏi vòng rerun của Streamlit.

## Cấu trúc

- `app.py`: UI + orchestration, không chứa logic low-level.
- `retrieval_engine.py`: load/build index, normalize một lần, exact Top-K.
- `build_search_index.py`: build compiled index offline từ các NPZ hiện có.
- `hf_assets.py`: auth + tải keyframe private song song + memory cache.
- `video_player.py`: video range server + browser player tối ưu seek/hover.
- `submission.py`: KIS / Q&A / TRAKE + CSV validation/export.
- `siglip2_embedder.py`: SigLIP2 wrapper; app retrieval chỉ load text side preprocessing.
- `config.py`: toàn bộ path/config.

## 1. Đặt file

Cách đơn giản nhất: copy các file trong thư mục này vào thư mục project hiện tại, nơi đang có `workspace/`.

Nếu `workspace` nằm nơi khác, set biến môi trường `AIC_WORKSPACE` trỏ tới folder đó.

Windows PowerShell:

```powershell
$env:AIC_WORKSPACE="D:\\path\\to\\workspace"
```

## 2. Cài dependency

```bash
pip install -r requirements_optimized.txt
hf auth login
```

## 3. Build search index một lần

```bash
python build_search_index.py
```

Kết quả:

```text
workspace/
├─ vector_embeddings/       # dữ liệu cũ, giữ nguyên
└─ search_index/
   ├─ vectors.npy           # float32, L2-normalized sẵn
   ├─ metadata.jsonl
   └─ manifest.json
```

Không cần build lại sau mỗi lần chạy app. Chỉ build lại khi thêm/sửa embedding.

## 4. Chạy app

```bash
streamlit run app.py
```

## Vì sao nhanh hơn

1. Không normalize toàn bộ document embeddings ở mỗi Search.
2. Compiled index chỉ đọc một matrix contiguous thay vì quét/ghép nhiều NPZ + JSONL ở lần đầu.
3. `@st.cache_resource` giữ nguyên model và search engine trong process, tránh serialize/copy matrix lớn như `cache_data`.
4. Keyframe private HF tải song song (mặc định 8 worker), chỉ tải phần đang hiển thị, có LRU byte cache.
5. Kéo seek bar không seek video chính liên tục; seek chỉ xảy ra khi thả chuột.
6. Hover thumbnail dùng một browser video phụ, bỏ OpenCV decode từng frame qua HTTP `/thumb`.
7. CSV editor là Streamlit fragment riêng, nên sửa bảng CSV không làm player video bị dựng lại/reset.
8. TRAKE đúng rule: chỉ chọn số events rồi nhập frame thủ công.

## Khi nào mới cần FAISS?

Với vài chục nghìn keyframe, exact matrix-vector multiplication thường đủ nhanh và cho kết quả chính xác 100% theo cosine. Chỉ cân nhắc FAISS khi index tăng lên hàng trăm nghìn/hàng triệu vector hoặc phép search thực sự trở thành bottleneck sau khi đo timing trong UI.
