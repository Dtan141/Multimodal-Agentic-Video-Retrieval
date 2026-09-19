"""Lexical (BM25) search over the per-module metadata indexes."""

from __future__ import annotations

from aic_retrieval.config import Settings
from aic_retrieval.text.bm25_index import ModuleBM25
from aic_retrieval.text.ingest_metadata import MODULES


class LexicalSearcher:
    def __init__(self, modules: dict[str, ModuleBM25]) -> None:
        self.modules = modules

    @classmethod
    def load(cls, settings: Settings) -> LexicalSearcher:
        text_dir = settings.index_dir / "text"
        mods = {m: ModuleBM25.load(text_dir, m) for m in MODULES}
        return cls(mods)

    @property
    def available(self) -> list[str]:
        return list(self.modules)

    def search(self, module: str, query: str, top_k: int = 100, keep=None) -> list[dict]:
        if module not in self.modules:
            raise KeyError(f"Unknown module {module!r}; available: {self.available}")
        hits = self.modules[module].search(query, top_k=top_k, keep=keep)
        return [{**h, "source": f"meta:{module}"} for h in hits]
