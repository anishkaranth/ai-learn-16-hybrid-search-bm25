#!/usr/bin/env python3
"""Hybrid-search smoke: BM25 vs dense LSA vs RRF vs weighted fusion on keyword + paraphrase queries -> results/."""
from __future__ import annotations

import json
import re
import time
from pathlib import Path

import numpy as np

from data import DOCS, QUERIES
from fusion import rrf, weighted
from retrieval import DEPTH, build, methods, run
from smoke_plots import make_plots, write_results_md

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
SEED = 42  # everything is deterministic; kept for the series convention
DIM, K1, B, ALPHA, RRF_K = 24, 1.5, 0.75, 0.5, 60


def _compact(js: str) -> str:
    return re.sub(r"\[\s+([^\[\]{}]*?)\s+\]", lambda m: "[" + re.sub(r"\s+", " ", m.group(1)) + "]", js)


def _ndcg_by_type(res):
    return {g: res["aggregate"][g]["ndcg@5"] for g in ("keyword", "paraphrase", "all")}


def main() -> None:
    np.random.seed(SEED)
    t0 = time.perf_counter()
    bm, de = build(DIM, K1, B)
    runs = {name: run(fn) for name, fn in methods(bm, de, ALPHA, RRF_K).items()}
    alpha_sweep = []
    for a in np.round(np.linspace(0, 1, 11), 1):
        r = run(lambda q, a=float(a): weighted(bm.ids, bm.scores(q), de.scores(q), a))
        alpha_sweep.append({"alpha": float(a), **_ndcg_by_type(r)})
    rrf_sweep = []
    for k in (1, 10, 60):
        r = run(lambda q, k=k: rrf([bm.search(q, DEPTH), de.search(q, DEPTH)], k=k))
        rrf_sweep.append({"k": k, **_ndcg_by_type(r)})
    best = max(alpha_sweep, key=lambda r: (r["all"], -abs(r["alpha"] - 0.5)))
    runtime = time.perf_counter() - t0

    m = {
        "project": "ai-learn-16-hybrid-search-bm25", "seed": SEED, "n_docs": len(DOCS), "n_queries": len(QUERIES),
        "n_by_type": {t: sum(1 for q in QUERIES if q[1] == t) for t in ("keyword", "paraphrase")},
        "config": {"bm25_k1": K1, "bm25_b": B, "dense_dim": DIM, "weighted_alpha": ALPHA, "rrf_k": RRF_K, "rrf_depth": DEPTH},
        "dense": {"vocab_size": len(de.vocab), "explained_var": round(de.explained, 4), "doc_oov_rate": round(de.doc_oov_rate, 4)},
        "metric_keys": list(next(iter(runs.values()))["aggregate"]["all"]),
        "methods": {n: r["aggregate"] for n, r in runs.items()},
        "alpha_sweep": alpha_sweep, "best_alpha": best, "rrf_sweep": rrf_sweep,
        "relevance": [" ".join(f"{d}:{g}" for d, g in q[3].items()) for q in QUERIES],
        "rows": {n: r["rows"] for n, r in runs.items()},
        "runtime_s": round(runtime, 4),
    }
    RESULTS.mkdir(exist_ok=True)
    plots = make_plots(RESULTS, m)
    m["plots"] = plots
    write_results_md(RESULTS, m, plots)
    slim = {k: v for k, v in m.items() if k not in ("rows", "relevance")}
    slim["per_query_ndcg@5"] = {n: {r["id"]: round(r["ndcg@5"], 3) for r in rows} for n, rows in m["rows"].items()}
    (RESULTS / "metrics.json").write_text(_compact(json.dumps(slim, indent=1)), encoding="utf-8")
    hy = f"weighted_a{ALPHA}"
    shot = {"project": m["project"], "seed": SEED, "config": m["config"], "n_queries": len(QUERIES),
            "ndcg@5": {n: {g: a[g]["ndcg@5"] for g in ("keyword", "paraphrase", "all")} for n, a in m["methods"].items()},
            "mrr@10_all": {n: a["all"]["mrr@10"] for n, a in m["methods"].items()},
            "recall@5_all": {n: a["all"]["recall@5"] for n, a in m["methods"].items()},
            "best_alpha": best, "runtime_s": m["runtime_s"],
            "pass": bool(m["methods"][hy]["all"]["ndcg@5"] >= max(m["methods"]["bm25"]["all"]["ndcg@5"], m["methods"]["dense"]["all"]["ndcg@5"]))}
    (RESULTS / "JSON.shot").write_text(json.dumps(shot, indent=2), encoding="utf-8")
    for n, a in m["methods"].items():
        print(f"  {n:14s} " + " ".join(f"{g}={a[g]['ndcg@5']:.3f}" for g in ("keyword", "paraphrase", "all")) + f" mrr={a['all']['mrr@10']:.3f}")
    print(f"best alpha={best['alpha']} all={best['all']:.3f} | pass={shot['pass']} | runtime {runtime:.3f}s | wrote {RESULTS}")


if __name__ == "__main__":
    main()
