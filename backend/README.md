# AIC 2026 Retrieval Backend

Python package (`aic_retrieval`) for multimedia retrieval: SigLIP2 semantic search
+ FAISS index + per-module metadata (OCR/ASR/Object/Caption) hybrid search + a Groq
agent that routes queries — served over FastAPI.

## Layout

```
backend/
├── configs/default.yaml          # config (override via env AIC_* or .env)
├── src/aic_retrieval/
│   ├── config.py                 # pydantic-settings
│   ├── logging_conf.py
│   ├── embedding/siglip2.py      # SigLIP2 embedder (single source of truth)
│   ├── index/                    # FAISS + metadata store   (WS2)
│   ├── text/                     # metadata ingest + BM25    (WS3)
│   ├── search/                   # semantic/lexical/fusion/rerank (WS4)
│   ├── agent/                    # Groq client, router, tools (WS5)
│   ├── api/                      # FastAPI app               (WS6)
│   └── cli.py                    # `aic` command
```

## Setup

```bash
# create the venv (Python 3.12)
python3 -m venv ~/venvs/aic2026
source ~/venvs/aic2026/bin/activate

# core package (config/logging/cli)
pip install -e "backend[dev]"

# ML + API stacks (installed as needed for WS2+)
pip install -e "backend[ml,api]"
# torch (CUDA 11.8) is installed separately:
#   pip install torch==2.7.1+cu118 --index-url https://download.pytorch.org/whl/cu118
```

## Quick check

```bash
aic info          # print resolved settings + embeddings summary
```

## Configuration

Precedence: init kwargs > env (`AIC_*`, plus `GROQ_API_KEY`) > `.env` > `configs/default.yaml` > defaults.
Copy `.env.example` to `.env` for secrets. The Hugging Face token comes from
`hf auth login`, not from config.
