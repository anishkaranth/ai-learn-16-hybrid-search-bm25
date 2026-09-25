"""Matplotlib SVG plots + RESULTS.md writer for the hybrid-search smoke run."""
from __future__ import annotations

import io
from pathlib import Path
from typing import Any, Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from svg_utils import minify_svg  # noqa: E402

plt.rcParams["svg.hashsalt"] = "ai-learn-16"
plt.rcParams["svg.fonttype"] = "none"
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
_META = {"Date": None}
_COLORS = ["#e76f51", "#2a9d8f", "#e9c46a", "#264653"]


def _save(fig, path: Path) -> str:
    fig.tight_layout()
    buf = io.StringIO()
    fig.savefig(buf, format="svg", metadata=_META)
    plt.close(fig)
    path.write_text(minify_svg(buf.getvalue()), encoding="utf-8")
    return path.name


def make_plots(out: Path, m: Dict[str, Any]) -> List[str]:
    out.mkdir(exist_ok=True)
    names = []
    meths = list(m["methods"])
    groups = ["keyword", "paraphrase", "all"]
    fig, ax = plt.subplots(figsize=(7, 3.4))
    w = 0.8 / len(meths)
    x = np.arange(len(groups))
    for i, name in enumerate(meths):
        vals = [m["methods"][name][g]["ndcg@5"] for g in groups]
        ax.bar(x + i * w - 0.4 + w / 2, vals, w, label=name, color=_COLORS[i % 4])
    ax.set_xticks(x, groups)
    ax.set_ylim(0, 1.05)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_ylabel("nDCG@5")
    ax.set_title("Sparse vs dense vs hybrid by query type")
    ax.legend(fontsize=8, loc="lower right")
    names.append(_save(fig, out / "ndcg_by_query_type.svg"))

    sw = m["alpha_sweep"]
    fig, ax = plt.subplots(figsize=(6, 3.2))
    a = [r["alpha"] for r in sw]
    for g, c in zip(groups, _COLORS):
        ax.plot(a, [r[g] for r in sw], color=c, label=g)
    ax.set_xlabel("alpha (0 = BM25 only, 1 = dense only)")
    ax.set_ylabel("nDCG@5")
    ax.set_ylim(0, 1.05)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_title("Weighted fusion: alpha sweep")
    ax.legend(fontsize=8, loc="lower left")
    names.append(_save(fig, out / "alpha_sweep.svg"))
    return names


def write_results_md(out: Path, m: Dict[str, Any], plots: List[str]) -> None:
    keys = m["metric_keys"]
    c = m["config"]
    L = ["# Results -- ai-learn-16-hybrid-search-bm25", "",
         f"**Seed:** `{m['seed']}` | docs={m['n_docs']} | queries={m['n_queries']} ({m['n_by_type']['keyword']} keyword, {m['n_by_type']['paraphrase']} paraphrase) | "
         f"BM25 k1={c['bm25_k1']} b={c['bm25_b']} | dense LSA dim={c['dense_dim']} (background explained var {m['dense']['explained_var']:.3f}, doc-token OOV {m['dense']['doc_oov_rate']:.2f}) | "
         f"RRF depth={c['rrf_depth']}", ""]
    for g in ("all", "keyword", "paraphrase"):
        L += [f"## {g} queries (real smoke run)", "", "| method | " + " | ".join(keys) + " |", "|---|" + "---:|" * len(keys)]
        for name, agg in m["methods"].items():
            L.append(f"| `{name}` | " + " | ".join(f"{agg[g][k]:.3f}" for k in keys) + " |")
        L.append("")
    L += ["## Fusion sweeps (nDCG@5)", "", "| weighted alpha | keyword | paraphrase | all |", "|---:|---:|---:|---:|"]
    for r in m["alpha_sweep"]:
        L.append(f"| {r['alpha']:.1f} | {r['keyword']:.3f} | {r['paraphrase']:.3f} | {r['all']:.3f} |")
    L += ["", f"Best alpha on this eval set: **{m['best_alpha']['alpha']}** (all nDCG@5 {m['best_alpha']['all']:.3f}). It was tuned on the same 28 queries, so treat it as optimistic.", "",
          "| RRF k | keyword | paraphrase | all |", "|---:|---:|---:|---:|"]
    for r in m["rrf_sweep"]:
        L.append(f"| {r['k']} | {r['keyword']:.3f} | {r['paraphrase']:.3f} | {r['all']:.3f} |")
    L += ["", "## Per-query top-3 (weighted fusion vs BM25 vs dense)", "", "| id | type | relevant | BM25 top-3 | dense top-3 | weighted top-3 |", "|---|---|---|---|---|---|"]
    hy = list(m["methods"])[-1]
    for r_b, r_d, r_h, rel in zip(m["rows"]["bm25"], m["rows"]["dense"], m["rows"][hy], m["relevance"]):
        L.append(f"| {r_b['id']} | {r_b['type']} | {rel} | {' '.join(r_b['top3'])} | {' '.join(r_d['top3'])} | {' '.join(r_h['top3'])} |")
    L += ["", "## Plots", ""] + [f"![{p}]({p})" for p in plots] + ["", f"Wall time: {m['runtime_s']:.3f}s on CPU.", ""]
    (out / "RESULTS.md").write_text("\n".join(L), encoding="utf-8")
