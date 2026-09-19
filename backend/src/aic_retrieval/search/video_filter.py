"""Video-scope filtering (include / exclude) by video name or folder.

A spec containing ``_V`` (case-insensitive) matches a video_id exactly
(``L21_V018``); otherwise it matches a whole folder (``L21`` -> all L21_*).

keep(video_id) = (include empty OR matches an include spec) AND (matches no exclude spec)
"""

from __future__ import annotations

from collections.abc import Callable, Iterable

import numpy as np


def _split_specs(specs: Iterable[str] | None) -> tuple[set[str], set[str]]:
    exact: set[str] = set()
    folder: set[str] = set()
    for s in specs or []:
        s = str(s).strip()
        if not s:
            continue
        if "_v" in s.lower():
            exact.add(s)
        else:
            folder.add(s)
    return exact, folder


class VideoScope:
    def __init__(self, video_ids: np.ndarray, folder_ids: np.ndarray) -> None:
        self.video_ids = video_ids
        self.folder_ids = folder_ids

    @classmethod
    def from_store(cls, store) -> VideoScope:
        return cls(
            store.df["video_id"].to_numpy().astype(str),
            store.df["folder_id"].to_numpy().astype(str),
        )

    def allowed_row_ids(
        self,
        include: Iterable[str] | None,
        exclude: Iterable[str] | None,
    ) -> np.ndarray | None:
        """Row ids allowed by the filter, or None when there is no restriction."""
        ei, fi = _split_specs(include)
        ee, fe = _split_specs(exclude)
        if not (ei or fi or ee or fe):
            return None

        n = len(self.video_ids)
        if ei or fi:
            include_mask = np.zeros(n, dtype=bool)
            if ei:
                include_mask |= np.isin(self.video_ids, list(ei))
            if fi:
                include_mask |= np.isin(self.folder_ids, list(fi))
        else:
            include_mask = np.ones(n, dtype=bool)

        exclude_mask = np.zeros(n, dtype=bool)
        if ee:
            exclude_mask |= np.isin(self.video_ids, list(ee))
        if fe:
            exclude_mask |= np.isin(self.folder_ids, list(fe))

        keep = include_mask & ~exclude_mask
        return np.where(keep)[0].astype(np.int64)

    @staticmethod
    def keep_fn(
        include: Iterable[str] | None,
        exclude: Iterable[str] | None,
    ) -> Callable[[str], bool] | None:
        """Predicate over video_id for the lexical branch, or None if no restriction."""
        ei, fi = _split_specs(include)
        ee, fe = _split_specs(exclude)
        if not (ei or fi or ee or fe):
            return None
        has_include = bool(ei or fi)

        def keep(video_id: str) -> bool:
            folder = video_id.split("_", 1)[0]
            inc = (not has_include) or (video_id in ei) or (folder in fi)
            exc = (video_id in ee) or (folder in fe)
            return inc and not exc

        return keep
