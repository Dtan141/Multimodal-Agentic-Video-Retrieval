# Backend API image (CPU by default). For GPU, use an nvidia/cuda base and install
# torch with the matching CUDA index-url (see README).
FROM python:3.12-slim

WORKDIR /app

# libgl/libglib for opencv (video info)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/*

COPY backend/pyproject.toml backend/README.md ./backend/
COPY backend/src ./backend/src
COPY backend/configs ./backend/configs

RUN pip install --no-cache-dir -e "./backend[index,text,api,ml]" \
 && pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# The API serves from /app/database/index (mount it) and needs an HF token
# (mount ~/.cache/huggingface) for image/video proxying + the SigLIP model.
EXPOSE 8000
CMD ["uvicorn", "aic_retrieval.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
