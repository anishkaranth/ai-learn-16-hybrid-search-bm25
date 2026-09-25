# ai-learn-16-hybrid-search-bm25

**Phase D, day 16** of the AI learning track. RAG retrieval in production is usually **hybrid**: a lexical ranker (BM25) for exact tokens plus a dense ranker for meaning, merged with a fusion rule. This repo builds each piece from scratch: BM25, a dense TF-IDF → SVD (LSA) retriever, **reciprocal rank fusion** and **weighted score fusion**. It then compares recall@k, MRR and nDCG on a labeled toy helpdesk corpus with **keyword-heavy** queries (error codes, product names, flags) and **paraphrased** queries (the same need in different words).

It follows TF-IDF search (`ai-learn-07`), RAG (`ai-learn-08`) and the end-to-end assistant (`ai-learn-14`), whose limitations section named paraphrase failures as the next problem. It is self-contained: NumPy + matplotlib, no network, deterministic, well under a second on CPU.

## What you'll learn

- How BM25 works (IDF, term-frequency saturation `k1`, length normalisation `b`) and why it is unbeatable on rare exact tokens.
- Why a dense model trained on generic text misses out-of-vocabulary strings but handles synonyms. Here the LSA "encoder" is fitted only on a background corpus, and 54% of document tokens are out of its vocabulary.
- How RRF and weighted fusion differ. RRF uses only ranks, so it cannot tell a confident ranker from a clueless one. Normalised score fusion can.
- How to evaluate retrieval by query slice, and why tuning the fusion weight on your eval set is optimistic.

## Architecture

```mermaid
flowchart LR
  Q[query] --> T[tokenize + stem]
  T --> S[BM25 over docs]
  T --> V[TF-IDF with background vocab]
  BG[(background corpus)] -. fit vocab, IDF, SVD .-> V
  V --> P[project to 24-d LSA space, L2 norm]
  P --> C[cosine vs doc embeddings]
  S --> R1[sparse ranking + scores]
  C --> R2[dense ranking + scores]
  R1 --> RRF["RRF: sum 1/(k + rank), top-10 from each"]
  R2 --> RRF
  R1 --> W["weighted: a*minmax(dense) + (1-a)*minmax(bm25)"]
  R2 --> W
  RRF --> E[recall@1/3/5, MRR@10, nDCG@5 per query type]
  W --> E
  R1 --> E
  R2 --> E
  E --> OUT[results/]
```

## Layout

| path | purpose |
|---|---|
| `data.py` | 30 helpdesk docs, 35 background lines (dense-model "pre-training" text), 28 graded queries (12 keyword, 16 paraphrase) |
| `text.py` | tokenizer, stopwords, crude suffix stemmer |
| `bm25.py` | Okapi BM25 (`k1`, `b`, non-negative IDF) |
| `dense.py` | `LSADense`: TF-IDF → truncated SVD → cosine search, with an OOV-rate diagnostic |
| `fusion.py` | `rrf`, `minmax`, `weighted` |
| `metrics.py` | recall@k, MRR@k, graded nDCG@k |
| `retrieval.py` | builds the retrievers and runs every method over the queries |
| `run_smoke.py` / `smoke_plots.py` | smoke run, alpha and RRF-k sweeps, SVG plots, `RESULTS.md` |
| `notebooks/hybrid_search.ipynb` | walkthrough |
| `results/` | committed `RESULTS.md`, `metrics.json`, `JSON.shot`, `*.svg` |

## Run

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python run_smoke.py
```

## Headline results (from `results/metrics.json`)

| method | keyword nDCG@5 | paraphrase nDCG@5 | all nDCG@5 | all MRR@10 | all recall@5 |
|---|---:|---:|---:|---:|---:|
| BM25 | **1.000** | 0.668 | 0.810 | 0.833 | 0.804 |
| dense (LSA) | 0.553 | **0.939** | 0.773 | 0.756 | 0.839 |
| RRF (k=60) | 0.605 | 0.782 | 0.706 | 0.705 | 0.786 |
| weighted (α=0.5) | **1.000** | 0.853 | **0.916** | **0.917** | **0.946** |

- Weighted fusion keeps BM25's perfect keyword score and recovers most of the dense model's paraphrase gains. An α sweep peaks at 0.947 nDCG@5 for α=0.7, but that α was tuned on the eval set.
- **RRF scored below both single rankers** here. For all-OOV keyword queries the dense ranking is arbitrary, and RRF gives it the same weight as BM25's confident ranking. Smaller k helps (k=1: 0.800 overall), but it still trails weighted fusion. This is a real failure mode worth knowing, not a tuning bug.

## Limitations and next steps

The corpus is small and hand-written. The background corpus was written for this domain, which flatters the dense model on paraphrases. LSA is a weak stand-in for a neural encoder. Next is a learned reranker on top of first-stage retrieval (`ai-learn-17`).
