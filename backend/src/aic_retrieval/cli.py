"""Command line entry point (``aic ...``).

WS1 implements ``info``; the other commands are stubs wired up in later
workstreams (build-index=WS2, download/ingest-metadata=WS3, serve=WS6).
"""

from __future__ import annotations

import typer

from aic_retrieval.config import get_settings
from aic_retrieval.logging_conf import get_logger, setup_logging

app = typer.Typer(add_completion=False, help="AIC 2026 retrieval backend CLI")
logger = get_logger(__name__)


@app.callback()
def _main(log_level: str = typer.Option(None, help="Override log level (DEBUG/INFO/...).")) -> None:
    settings = get_settings()
    setup_logging(log_level or settings.log_level)


@app.command()
def info() -> None:
    """Print resolved settings and a summary of the embeddings directory."""
    settings = get_settings()

    typer.echo("=== Settings ===")
    for field in type(settings).model_fields:
        value = getattr(settings, field)
        if field == "groq_api_key":
            value = "<set>" if value else "<unset>"
        typer.echo(f"  {field:22s} = {value}")

    emb_dir = settings.embeddings_dir
    typer.echo(f"\n=== Embeddings dir: {emb_dir} ===")
    if not emb_dir.exists():
        typer.secho("  (directory does not exist yet)", fg=typer.colors.YELLOW)
        return

    npz_files = sorted(emb_dir.glob("*.npz"))
    jsonl_files = sorted(emb_dir.glob("*.jsonl"))
    typer.echo(f"  {len(npz_files)} .npz, {len(jsonl_files)} .jsonl")
    for path in npz_files:
        size_mb = path.stat().st_size / (1024 * 1024)  # stat() follows symlinks
        typer.echo(f"  {path.name:34s} {size_mb:8.1f} MB")


@app.command("build-index")
def build_index_cmd(
    embeddings_dir: str = typer.Option(None, help="Override embeddings dir."),
    index_dir: str = typer.Option(None, help="Override output index dir."),
) -> None:
    """(WS2) Build the FAISS index + frame metadata store from the .npz files."""
    from pathlib import Path

    from aic_retrieval.index.build import build_index

    settings = get_settings()
    emb = Path(embeddings_dir) if embeddings_dir else settings.embeddings_dir
    out = Path(index_dir) if index_dir else settings.index_dir

    typer.echo(f"Building index from {emb} -> {out}")
    result = build_index(emb, out)

    typer.echo("")
    for row in result["files"]:
        typer.echo(f"  {row['file']:34s} {row['status']:20s} kept={row.get('kept', 0)}")
    typer.secho(
        f"\n✅ {result['total_vectors']:,} vectors (dim={result['dim']}) indexed at {out}",
        fg=typer.colors.GREEN,
    )


@app.command("download-metadata")
def download_metadata_cmd(
    dest: str = typer.Option(None, help="Override metadata dir."),
) -> None:
    """(WS3) Download OCR/ASR/Object/Caption metadata JSON from Hugging Face."""
    from pathlib import Path

    from aic_retrieval.text.download import download_metadata

    settings = get_settings()
    dest_dir = Path(dest) if dest else settings.metadata_dir
    local = download_metadata(
        repo_id=settings.hf_metadata_repo_id,
        repo_type=settings.hf_metadata_repo_type,
        dest_dir=dest_dir,
    )
    typer.secho(f"✅ Metadata downloaded to {local}", fg=typer.colors.GREEN)


@app.command("ingest-metadata")
def ingest_metadata_cmd(
    metadata_dir: str = typer.Option(None, help="Override metadata dir."),
    index_dir: str = typer.Option(None, help="Override index dir."),
) -> None:
    """(WS3) Parse metadata JSON into per-module BM25 indexes."""
    from pathlib import Path

    from aic_retrieval.text.ingest_metadata import ingest_metadata

    settings = get_settings()
    if metadata_dir:
        md = Path(metadata_dir)
    elif settings.clean_metadata_dir.exists():
        md = settings.clean_metadata_dir  # prefer cleaned data when available
    else:
        md = settings.metadata_dir
    out = Path(index_dir) if index_dir else settings.index_dir

    typer.echo(f"Ingesting from {md}")
    result = ingest_metadata(md, out)
    typer.echo(f"videos={result['videos']} segments={result['segments']}")
    for module, n in result["module_docs"].items():
        typer.echo(f"  {module:8s} docs = {n:,}")
    typer.secho(f"\n✅ BM25 indexes written to {out / 'text'}", fg=typer.colors.GREEN)


@app.command("analyze-metadata")
def analyze_metadata_cmd(
    metadata_dir: str = typer.Option(None, help="Override metadata dir."),
    out_dir: str = typer.Option(None, help="Output dir (default <index_dir>/preprocess)."),
    kf_fraction: float = typer.Option(0.30, help="Keyframe-fraction threshold for OCR stoplist."),
    top_n: int = typer.Option(25, help="How many top terms to print per folder."),
) -> None:
    """(WS3.5) Per-folder frequency analysis + suggested OCR stoplist."""
    from pathlib import Path

    from aic_retrieval.preprocess.analyze import analyze

    settings = get_settings()
    md = Path(metadata_dir) if metadata_dir else settings.metadata_dir
    out = Path(out_dir) if out_dir else settings.index_dir / "preprocess"

    report, stoplist = analyze(md, out, kf_fraction=kf_fraction, top_n=top_n)
    for fo, rep in report.items():
        typer.secho(
            f"\n=== {fo}: {rep['videos']} videos, {rep['ocr_keyframes']} ocr-keyframes, "
            f"suggested stoplist={rep['suggested_stop_count']} ===",
            fg=typer.colors.CYAN,
        )
        typer.echo("  top OCR tokens [token | #kf | kf_frac | #vid | vid_cov]:")
        for tok, kf, frac, nv, cov in rep["top_ocr"][:top_n]:
            flag = "  <-- junk?" if frac >= kf_fraction else ""
            typer.echo(f"    {tok:20s} {kf:7d} {frac:6.2f} {nv:5d} {cov:6.2f}{flag}")
    typer.secho(f"\n✅ Report + stoplist written to {out}", fg=typer.colors.GREEN)


@app.command("clean-metadata")
def clean_metadata_cmd(
    metadata_dir: str = typer.Option(None, help="Raw metadata dir."),
    clean_dir: str = typer.Option(None, help="Output clean dir (default clean_metadata_dir)."),
) -> None:
    """(WS3.5) Slim + junk-filter metadata (needs `analyze-metadata` first)."""
    import json
    from pathlib import Path

    from aic_retrieval.preprocess.clean import clean_metadata, merge_stoplist

    settings = get_settings()
    md = Path(metadata_dir) if metadata_dir else settings.metadata_dir
    out = Path(clean_dir) if clean_dir else settings.clean_metadata_dir
    stoplist_path = settings.index_dir / "preprocess" / "ocr_stoplist.json"

    if not stoplist_path.exists():
        typer.secho("Run `aic analyze-metadata` first (ocr_stoplist.json missing).", fg=typer.colors.RED)
        raise typer.Exit(code=1)

    auto = json.loads(stoplist_path.read_text(encoding="utf-8"))
    merged = merge_stoplist({k: v for k, v in auto.items() if k != "_global"})
    stoplist_path.write_text(json.dumps(merged, ensure_ascii=False, indent=2), encoding="utf-8")
    typer.echo(f"Stoplist: {len(merged['_global'])} global + per-folder auto -> {stoplist_path}")

    result = clean_metadata(md, out, merged)
    typer.secho(
        f"\n✅ {result['videos']} videos cleaned -> {out}\n"
        f"   speech kept={result['speech_kept']:,} dropped={result['speech_dropped']:,} | "
        f"ocr tokens removed={result['ocr_tokens_removed']:,}",
        fg=typer.colors.GREEN,
    )


@app.command()
def agent(
    query: str = typer.Argument(..., help="Natural-language query (Vietnamese ok)."),
    top_k: int = typer.Option(10, help="How many results."),
    rerank: bool = typer.Option(False, help="Enable optional cross-encoder rerank."),
) -> None:
    """(WS5) Run the agent: route the query, then hybrid search."""
    from aic_retrieval.agent.orchestrator import AgentSearcher

    ag = AgentSearcher.load(get_settings())
    out = ag.run(query, top_k=top_k, rerank=rerank)
    plan = out["plan"]
    typer.secho(f"\nsemantic_query: {plan['semantic_query']}", fg=typer.colors.CYAN)
    typer.echo(f"modules: {plan['module_queries']}")
    typer.echo(f"reasoning: {plan.get('reasoning', '')}\n")
    for r in out["results"]:
        typer.echo(
            f"  #{r.get('rank')} {r['video_id']} frame={r['frame_id']} "
            f"rrf={r.get('rrf_score', 0):.4f} cross={r.get('cross_score', '-')} src={r.get('sources')}"
        )


@app.command()
def serve(
    host: str = typer.Option(None, help="Bind host."),
    port: int = typer.Option(None, help="Bind port."),
    reload: bool = typer.Option(False, help="Auto-reload (dev)."),
) -> None:
    """(WS6) Run the FastAPI backend with uvicorn."""
    import uvicorn

    settings = get_settings()
    uvicorn.run(
        "aic_retrieval.api.main:app",
        host=host or settings.api_host,
        port=port or settings.api_port,
        reload=reload,
    )


if __name__ == "__main__":
    app()
