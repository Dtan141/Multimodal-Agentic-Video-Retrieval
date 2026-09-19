# Cài đặt & chạy trên Windows (tải dữ liệu vào ổ D:)

> Dành cho Windows + **Git Bash** (MINGW64). Mọi thứ tải về (model, ảnh, video) được
> đưa vào **ổ D:** để không đầy ổ C:. Máy CPU-only vẫn chạy được.

## 0) Yêu cầu
- **Python 3.10–3.11**, **Node ≥ 18**, **Git Bash**.
- Có sẵn thư mục dự án và `database/index/` (FAISS + BM25 đã build). Kiểm tra:
  ```bash
  ls database/index/vectors.faiss database/index/frames.parquet database/index/text
  ```
  Nếu thiếu → xem mục [Build lại index](#build-lại-index-nếu-thiếu-databaseindex).
- Khuyến nghị: **RAM trống ≥ 4GB** hoặc tăng pagefile (mục 7). Index ~1.2GB + model ~1.5GB.

## 1) Đưa cache HuggingFace sang ổ D: (làm 1 lần, vĩnh viễn)
Mở **PowerShell**:
```powershell
mkdir D:\hf_cache
setx HF_HUB_CACHE "D:\hf_cache"
```
> `setx` chỉ có hiệu lực ở **terminal mở MỚI**. Token đăng nhập vẫn ở ổ C: (rất nhỏ) nên không cần login lại.
> Nếu muốn cả token sang D:: `setx HF_HOME "D:\hf_cache"` rồi chạy `hf auth login` lại.

## 2) Tạo & kích hoạt venv (tránh lẫn với conda)
Mở **Git Bash** MỚI, tại thư mục gốc dự án:
```bash
conda deactivate 2>/dev/null   # tắt env conda (base) nếu đang bật, để khỏi tranh PATH
python -m venv ~/venvs/aic2026
source ~/venvs/aic2026/Scripts/activate      # Windows: Scripts (KHÔNG phải bin)

# XÁC NHẬN đang dùng python của venv (rất quan trọng):
python -c "import sys; print(sys.executable)"
# PHẢI in ra: C:\Users\<ban>\venvs\aic2026\Scripts\python.exe
```
> Nếu dòng trên **không** trỏ vào `venvs\aic2026` (do conda/PATH), hãy gọi thẳng python của venv trong MỌI lệnh: thay `python`/`pip` bằng `~/venvs/aic2026/Scripts/python -m ...`.

## 3) Cài package (chạy từ THƯ MỤC GỐC dự án)
```bash
export HF_HUB_CACHE="D:/hf_cache"                      # đảm bảo trong phiên này cũng trỏ D:
pip install -e "./backend[dev,index,text,api,ml]"      # nhớ có ./ ở đầu
pip install torch --index-url https://download.pytorch.org/whl/cpu   # torch bản CPU
pip install hf_xet                                     # (tùy chọn) tải HF nhanh hơn
```
> Lỗi *"not a valid editable requirement"* = bạn đang ở thư mục con (vd `frontend/`).
> Hãy `cd` về thư mục gốc (nơi có cả `backend/` và `frontend/`) và dùng `./backend[...]`.

## 4) Đăng nhập HuggingFace (để đọc ảnh/video private)
```bash
hf auth login        # dán token, chọn "Y" lưu lại
aic info             # xác nhận config; groq_api_key nên là <set> nếu dùng agent
```
> Agent: đặt key trong `backend/.env` → dòng `GROQ_API_KEY=...` (đừng commit file này).

## 5) Chạy backend
```bash
export HF_HUB_CACHE="D:/hf_cache"
aic serve --port 8000
# (hoặc nếu venv không "ăn": ~/venvs/aic2026/Scripts/python -m aic_retrieval.cli serve --port 8000)
```
Lần đầu sẽ tải model SigLIP2 (~1.5GB) về `D:\hf_cache`. Khi thấy
`Uvicorn running on http://0.0.0.0:8000` là backend đã sẵn sàng.

## 6) Chạy frontend (terminal Git Bash khác)
```bash
cd "/d/COMPETITION/AIC26/Multimodal Agentic Video Retrieval/frontend"   # đường dẫn dự án của bạn
npm install            # lần đầu
npm run dev            # mở http://localhost:3000
```
> `frontend/.env.local` cần `API_BASE=http://localhost:8000` (copy từ `.env.local.example`).

## 7) Nếu gặp lỗi bộ nhớ — tăng pagefile (bộ nhớ ảo)
Triệu chứng: `MemoryError: std::bad_alloc` hoặc `os error 1455: The paging file is too small`.
1. Win → gõ **"Adjust the appearance and performance of Windows"** → mở.
2. Tab **Advanced** → *Virtual memory* → **Change…**
3. Bỏ tick **Automatically manage…**; chọn ổ **D:** → **Custom size**: Initial `8192`, Maximum `16384` (MB) → **Set** → **OK**.
4. **Restart** máy, mở Git Bash mới, `export HF_HUB_CACHE="D:/hf_cache"` rồi chạy lại `aic serve`.
> Ngoài ra: `conda deactivate` + đóng bớt app để giải phóng RAM. Code đã tự **mmap** FAISS
> index khi thiếu RAM (không đổ 1.2GB vào RAM), nên chủ yếu chỉ cần đủ chỗ cho model.

## Build lại index (nếu thiếu database/index)
Chỉ cần khi bạn **không** có sẵn `database/index/`. Cần tải lại embeddings `.npz`
(khi copy từ Linux, các `.npz` là symlink nên có thể bị rỗng 0MB):
```bash
export HF_HUB_CACHE="D:/hf_cache"
~/venvs/aic2026/Scripts/python - <<'PY'
from huggingface_hub import hf_hub_download
import shutil, os
os.makedirs("database/vector_embeddings", exist_ok=True)
for L in ["L21","L22","L23","L24","L25","L26","L27","L28","L29","L30"]:
    f = f"Videos_{L}_embeddings.npz"
    p = hf_hub_download("Chillguy2026/AIC_2026_data", f"vector_embedding/{f}",
                        repo_type="dataset", local_dir="_hfdl")
    shutil.copy(p, os.path.join("database/vector_embeddings", f)); print("OK", f)
PY
aic download-metadata      # rồi:
aic analyze-metadata
aic clean-metadata
aic ingest-metadata
aic build-index
```

## Bảng lỗi thường gặp
| Lỗi | Nguyên nhân | Cách xử lý |
|---|---|---|
| `bin/activate: No such file` | Windows dùng `Scripts/` | `source ~/venvs/aic2026/Scripts/activate` |
| `not a valid editable requirement` | Đang ở thư mục con | `cd` về gốc, dùng `./backend[...]` |
| `sys.executable` trỏ global | conda/PATH lấn | `conda deactivate`, hoặc gọi `~/venvs/aic2026/Scripts/python -m ...` |
| `No space left on device` | Cache ở ổ C: đầy | `setx HF_HUB_CACHE "D:\hf_cache"` |
| `std::bad_alloc` / `paging file too small (1455)` | Thiếu RAM/pagefile | Tăng pagefile (mục 7), đóng app, `conda deactivate` |
| `.npz` 0MB | Symlink hỏng khi copy | Tải lại `.npz` (mục Build lại index) |
