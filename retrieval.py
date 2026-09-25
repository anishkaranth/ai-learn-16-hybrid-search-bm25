"""Build all retrievers and run every method on every query."""
from __future__ import annotations

from typing import Callable, Dict, List

import numpy as np

from bm25 import BM25
from data import BACKGROUND, DOCS, QUERIES
from dense import LSADense
from fusion import rrf, weighted
from metrics import evaluate

DEPTH = 10  # candidates each first-stage ranker hands to RRF


def build(dim: int = 24, k1: float = 1.5, b: float = 0.75):
    return BM25(DOCS, k1=k1, b=b), LSADense(DOCS, BACKGROUND, dim=dim)


def methods(bm: BM25, de: LSADense, alpha: float = 0.5, rrf_k: int = 60) -> Dict[str, Callable[[str], List[str]]]:
    return {
        "bm25": lambda q: bm.search(q, len(bm.ids)),
        "dense": lambda q: de.search(q, len(de.ids)),
        f"rrf_k{rrf_k}": lambda q: rrf([bm.search(q, DEPTH), de.search(q, DEPTH)], k=rrf_k),
        f"weighted_a{alpha}": lambda q: weighted(bm.ids, bm.scores(q), de.scores(q), alpha),
    }


def run(method: Callable[[str], List[str]]) -> Dict:
    rows = []
    for qid, typ, q, rel in QUERIES:
        ranked = method(q)
        rows.append({"id": qid, "type": typ, "top3": ranked[:3], **evaluate(ranked, rel)})
    keys = [k for k in rows[0] if k not in ("id", "type", "top3")]
    agg = {"all": {k: round(float(np.mean([r[k] for r in rows])), 4) for k in keys}}
    for typ in ("keyword", "paraphrase"):
        agg[typ] = {k: round(float(np.mean([r[k] for r in rows if r["type"] == typ])), 4) for k in keys}
    return {"aggregate": agg, "rows": rows}
