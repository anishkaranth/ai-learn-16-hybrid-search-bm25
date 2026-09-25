"""Hybrid fusion: reciprocal rank fusion (RRF) and weighted score fusion."""
from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np


def rrf(rankings: Sequence[List[str]], k: int = 60) -> List[str]:
    """score(d) = sum over rankers of 1 / (k + rank_d). Uses ranks only, so score scales don't matter."""
    s: Dict[str, float] = {}
    for ranking in rankings:
        for r, d in enumerate(ranking, start=1):
            s[d] = s.get(d, 0.0) + 1.0 / (k + r)
    return sorted(s, key=lambda d: (-s[d], d))


def minmax(x: np.ndarray) -> np.ndarray:
    lo, hi = float(x.min()), float(x.max())
    return np.zeros_like(x) if hi - lo < 1e-12 else (x - lo) / (hi - lo)


def weighted(ids: List[str], sparse: np.ndarray, dense: np.ndarray, alpha: float) -> List[str]:
    """alpha * minmax(dense) + (1 - alpha) * minmax(sparse). alpha=0 -> pure BM25, alpha=1 -> pure dense."""
    s = alpha * minmax(dense) + (1 - alpha) * minmax(sparse)
    return [ids[i] for i in np.argsort(-s, kind="stable")]
