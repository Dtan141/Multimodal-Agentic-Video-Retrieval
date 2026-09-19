# Hướng dẫn dọn dẹp & push lên GitHub cá nhân

> Mục tiêu: đẩy phiên bản mới (backend + frontend + docs) lên **GitHub cá nhân của bạn**,
> loại bỏ file cũ đã bị thay thế và **KHÔNG** đẩy dữ liệu lớn (`database/` ~2.2GB).
>
> ⚠️ Chạy từng bước, đọc kỹ. Lệnh xoá là không thể hoàn tác — nên commit/backup trước.

---

## 0) Chụp ảnh trạng thái trước khi làm

```bash
cd ~/working/prj/AIC_2026_Linh
git status
git remote -v          # hiện đang trỏ tới repo của người khác (PHTLing)
```

---

## 1) BẮT BUỘC: chặn dữ liệu lớn khỏi bị commit

`database/` (2.2GB, gồm symlink `.npz` trỏ ra ngoài repo) và `figures/` không nên lên GitHub.
Thêm vào `.gitignore`:

```bash
cat >> .gitignore <<'EOF'

# Dữ liệu lớn / dẫn xuất — không đẩy lên GitHub
database/
figures/
EOF
```

Kiểm tra đã chặn:
```bash
git check-ignore database database/vector_embeddings   # phải in ra tên (đã ignore)
```

---

## 2) Phân loại file

| Nhóm | File/thư mục | Xử lý |
|---|---|---|
| **GIỮ (hệ thống mới)** | `backend/`, `frontend/`, `docs/`, `.github/`, `docker-compose.yml`, `.dockerignore`, `README.md`, `.gitignore` | Thêm & commit |
| **XOÁ (cũ, đã thay thế)** | `AIC2026_retrieval_pipeline.py` (→ backend+frontend), `siglip2_embedder.py` (→ `backend/.../embedding/siglip2.py`), `upload_hf_vector.py`, `requirements.txt` (→ `backend/pyproject.toml`), `vector_embeddings/`, `test/` (script thử nghiệm, KHÔNG phải test thật — test thật ở `backend/tests/`) | `git rm` |
| **TUỲ CHỌN** | `data_processing/` (pipeline sinh keyframe/embedding offline) | Giữ nếu muốn khoe pipeline dữ liệu; hoặc `git rm` cho gọn |
| **KHÔNG đẩy** | `database/`, `figures/`, `backend/.env`, `node_modules/`, `.next/` | Đã ignore ở bước 1 / sẵn có |

> Lưu ý: `backend/.env` (chứa `GROQ_API_KEY`) đã được ignore — kiểm tra lại:
> `git check-ignore backend/.env` (phải in ra tên).

---

## 3) Xoá file cũ khỏi git

**Cách A — xoá hẳn khỏi repo và đĩa** (khuyến nghị cho repo portfolio gọn):
```bash
git rm -r AIC2026_retrieval_pipeline.py siglip2_embedder.py upload_hf_vector.py \
          requirements.txt vector_embeddings test
# (tuỳ chọn) xoá luôn pipeline offline:
# git rm -r data_processing
```

**Cách B — bỏ khỏi git nhưng GIỮ bản local** (nếu còn muốn tham khảo trên máy):
```bash
git rm -r --cached AIC2026_retrieval_pipeline.py siglip2_embedder.py upload_hf_vector.py \
                   requirements.txt vector_embeddings test
# rồi thêm chúng vào .gitignore để không bị add lại
printf '\n/AIC2026_retrieval_pipeline.py\n/siglip2_embedder.py\n/upload_hf_vector.py\n/requirements.txt\n/vector_embeddings/\n/test/\n' >> .gitignore
```

---

## 4) Thêm hệ thống mới & kiểm tra

```bash
git add backend frontend docs .github docker-compose.yml .dockerignore README.md .gitignore
git status                       # xem lại danh sách sẽ commit

# AN TOÀN: chắc chắn KHÔNG có gì thuộc database/ hay file lớn bị stage
git diff --cached --name-only | grep -E '^database/|\.npz$|\.mp4$|node_modules/|\.next/' \
  && echo "⚠️ CÓ file không nên commit — dừng lại kiểm tra!" || echo "OK, sạch."

# (khuyến nghị) xem file lớn nhất sắp commit
git diff --cached --name-only | xargs -r du -h 2>/dev/null | sort -rh | head -10
```

---

## 5) Commit

```bash
# nếu chưa cấu hình tên/email cho commit của bạn:
# git config user.name "Ten Cua Ban"
# git config user.email "email@cua.ban"

git commit -m "Refactor: FastAPI backend + Next.js frontend for AIC 2026 retrieval

- Semantic (SigLIP2+FAISS) + metadata (BM25) hybrid search, RRF fusion
- LLM agent (Groq) query routing; manual mode (module-only supported)
- Video-scope filtering, thumbnail cache/prefetch, submission builder
- Remove legacy prototype scripts; add tests, CI, Docker, docs"
```

---

## 6) Trỏ về GitHub cá nhân & push

Trước tiên **tạo repo rỗng** trên github.com của bạn (ví dụ `aic-2026-retrieval`), KHÔNG thêm README/gitignore.

**Cách 1 — đổi luôn `origin` sang repo của bạn** (đơn giản nhất):
```bash
git remote set-url origin https://github.com/<TEN_GITHUB_CUA_BAN>/aic-2026-retrieval.git
# hoặc dùng SSH:
# git remote set-url origin git@github.com:<TEN_GITHUB_CUA_BAN>/aic-2026-retrieval.git

git branch -M main
git push -u origin main
```

**Cách 2 — giữ `origin` cũ, thêm remote riêng tên `me`:**
```bash
git remote add me https://github.com/<TEN_GITHUB_CUA_BAN>/aic-2026-retrieval.git
git push -u me main
```

### Xác thực khi push (HTTPS)
GitHub không nhận mật khẩu — khi được hỏi, nhập **username GitHub + Personal Access Token (PAT)**.
Tạo PAT: GitHub → Settings → Developer settings → *Personal access tokens* → *Fine-grained* (quyền `Contents: Read and write` cho repo). Muốn lưu khỏi nhập lại:
```bash
git config --global credential.helper store   # lưu PAT vào ~/.git-credentials (dạng plaintext)
```
Hoặc dùng SSH: `ssh-keygen -t ed25519 -C "email@cua.ban"` rồi thêm `~/.ssh/id_ed25519.pub` vào GitHub → Settings → SSH keys, và dùng remote dạng `git@github.com:...`.

---

## 7) (Tuỳ chọn) Muốn lịch sử SẠCH — bỏ hết commit cũ của prototype

Nếu muốn repo cá nhân chỉ có 1 lịch sử gọn (không kéo theo commit prototype cũ):
```bash
git checkout --orphan clean-main       # nhánh mới không có lịch sử
git add -A
git commit -m "AIC 2026 multimedia retrieval system"
git branch -M clean-main main
git push -u origin main --force        # đẩy lịch sử mới lên repo CỦA BẠN
```
> Chỉ làm việc này trên **repo cá nhân của bạn**, đừng `--force` lên repo chung của người khác.

---

## 8) Kiểm tra sau khi push
- Mở repo trên GitHub: xác nhận có `backend/`, `frontend/`, `docs/`, `README.md` render đẹp (mermaid hiển thị), **không** thấy `database/`, `.env`, `node_modules/`.
- `git ls-files | wc -l` và dung lượng repo hợp lý (vài MB, không phải GB).

## Checklist nhanh
- [ ] `.gitignore` đã chặn `database/`, `figures/`, `backend/.env`
- [ ] Đã `git rm` file cũ (mục 3)
- [ ] `git diff --cached` không có file lớn / database / .env
- [ ] Đã tạo repo cá nhân rỗng trên GitHub
- [ ] `git remote set-url origin <repo của bạn>` rồi `git push -u origin main`
