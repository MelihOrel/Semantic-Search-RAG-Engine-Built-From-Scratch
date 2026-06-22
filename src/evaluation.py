"""Retrieval evaluation harness — the part that proves rigor.

Ground truth
------------
We have no human relevance judgements, so we derive them from the CFPB's own
free labels: a retrieved document is *relevant* to a query complaint if it
shares the same ``issue`` label (and ``product`` too, if configured). The query
itself is a held-out complaint's narrative; the query's own document is excluded
from its candidate pool so it cannot trivially retrieve itself.

Metrics (all computed at document level after de-duplicating chunks back to
their parent ``complaint_id``):

* Recall@k — fraction of relevant docs found in the top-k.
* MRR      — mean reciprocal rank of the first relevant doc.
* nDCG@k   — normalised discounted cumulative gain (binary relevance).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Sequence

import numpy as np
import pandas as pd

from .retrievers import BaseRetriever, SearchResult
from .utils import get_logger


@dataclass
class Query:
    complaint_id: str
    text: str
    labels: Dict[str, Any]


def build_query_set(corpus: pd.DataFrame, config: Dict[str, Any]) -> List[Query]:
    """Sample held-out complaints to use as queries.

    We only keep complaints whose relevance signature (issue [+product]) is
    shared by at least one *other* document, otherwise Recall is undefined
    (no relevant target exists).
    """
    ecfg = config["evaluation"]
    label_cols = list(ecfg["relevance_labels"])
    if ecfg.get("require_product") and "product" in corpus.columns:
        label_cols = list(dict.fromkeys(label_cols + ["product"]))

    sig = corpus[label_cols].astype(str).agg("||".join, axis=1)
    counts = sig.value_counts()
    eligible_mask = sig.map(counts) >= 2  # at least one other doc shares the sig
    eligible = corpus[eligible_mask]

    n = min(ecfg["query_sample_size"], len(eligible))
    sample = eligible.sample(n=n, random_state=ecfg["random_seed"])
    return [
        Query(
            complaint_id=str(r["complaint_id"]),
            text=r["narrative"],
            labels={c: r[c] for c in label_cols},
        )
        for _, r in sample.iterrows()
    ]


def relevant_doc_ids(query: Query, corpus: pd.DataFrame, label_cols: Sequence[str]) -> set[str]:
    """All doc ids sharing the query's label signature, excluding the query itself."""
    mask = np.ones(len(corpus), dtype=bool)
    for c in label_cols:
        mask &= corpus[c].astype(str).values == str(query.labels[c])
    ids = set(corpus.loc[mask, "complaint_id"].astype(str))
    ids.discard(query.complaint_id)
    return ids


def _dedupe_to_docs(results: List[SearchResult], exclude: str) -> List[str]:
    """Collapse ranked chunk hits to a ranked list of unique parent doc ids."""
    seen: set[str] = set()
    ordered: List[str] = []
    for r in results:
        cid = r.complaint_id
        if cid == exclude or cid in seen:
            continue
        seen.add(cid)
        ordered.append(cid)
    return ordered


def recall_at_k(ranked_docs: List[str], relevant: set[str], k: int) -> float:
    """Recall@k with the conventional capped denominator.

    Because relevance here is label-derived, a query's relevant set can be much
    larger than k (e.g. 69 complaints share one issue/product signature). Using
    the raw |relevant| denominator would cap Recall@10 at 10/69 even for a
    perfect ranking. We therefore normalise by ``min(|relevant|, k)`` — the
    maximum number of relevant docs that *could* appear in the top-k — which is
    the standard convention for retrieval with large relevance sets.
    """
    if not relevant:
        return math.nan
    hits = sum(1 for d in ranked_docs[:k] if d in relevant)
    return hits / min(len(relevant), k)


def reciprocal_rank(ranked_docs: List[str], relevant: set[str]) -> float:
    for i, d in enumerate(ranked_docs):
        if d in relevant:
            return 1.0 / (i + 1)
    return 0.0


def ndcg_at_k(ranked_docs: List[str], relevant: set[str], k: int) -> float:
    if not relevant:
        return math.nan
    dcg = 0.0
    for i, d in enumerate(ranked_docs[:k]):
        if d in relevant:
            dcg += 1.0 / math.log2(i + 2)
    ideal_hits = min(len(relevant), k)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_hits))
    return dcg / idcg if idcg > 0 else math.nan


def evaluate_retriever(
    retriever: BaseRetriever,
    queries: List[Query],
    corpus: pd.DataFrame,
    config: Dict[str, Any],
) -> Dict[str, float]:
    """Run one retriever over the query set and return averaged metrics."""
    ecfg = config["evaluation"]
    label_cols = list(ecfg["relevance_labels"])
    if ecfg.get("require_product") and "product" in corpus.columns:
        label_cols = list(dict.fromkeys(label_cols + ["product"]))
    k_values = config["retrieval"]["k_values"]
    primary_k = config["retrieval"]["primary_k"]
    # Over-fetch chunks so that after de-dup we still have enough docs.
    fetch_k = max(k_values) * 8

    per_q: Dict[str, List[float]] = {f"recall@{k}": [] for k in k_values}
    per_q.update({f"ndcg@{k}": [] for k in k_values})
    per_q["mrr"] = []

    for q in queries:
        relevant = relevant_doc_ids(q, corpus, label_cols)
        if not relevant:
            continue
        results = retriever.search(q.text, fetch_k)
        ranked_docs = _dedupe_to_docs(results, exclude=q.complaint_id)
        for k in k_values:
            per_q[f"recall@{k}"].append(recall_at_k(ranked_docs, relevant, k))
            per_q[f"ndcg@{k}"].append(ndcg_at_k(ranked_docs, relevant, k))
        per_q["mrr"].append(reciprocal_rank(ranked_docs, relevant))

    out = {m: float(np.nanmean(v)) if v else math.nan for m, v in per_q.items()}
    out["primary_k"] = primary_k
    out["n_queries"] = len([1 for q in queries if relevant_doc_ids(q, corpus, label_cols)])
    return out


def run_leaderboard(
    combos: List[Dict[str, Any]],
    queries: List[Query],
    corpus: pd.DataFrame,
    config: Dict[str, Any],
) -> pd.DataFrame:
    """Evaluate every (retriever × model × chunking) combo; return tidy table.

    ``combos`` is a list of dicts with keys: retriever, embedding_model,
    chunking, and the built ``instance`` (already indexed).
    """
    log = get_logger("evaluation", config.get("logging", {}).get("level", "INFO"))
    rows = []
    for combo in combos:
        metrics = evaluate_retriever(combo["instance"], queries, corpus, config)
        pk = config["retrieval"]["primary_k"]
        row = {
            "retriever": combo["retriever"],
            "embedding_model": combo["embedding_model"],
            "chunking": combo["chunking"],
            **metrics,
        }
        rows.append(row)
        log.info(
            "%-12s | %-18s | %-15s -> Recall@%d: %.3f | MRR: %.3f | nDCG@%d: %.3f",
            combo["retriever"], combo["embedding_model"], combo["chunking"],
            pk, metrics.get(f"recall@{pk}", float("nan")),
            metrics.get("mrr", float("nan")),
            pk, metrics.get(f"ndcg@{pk}", float("nan")),
        )
    df = pd.DataFrame(rows)
    pk = config["retrieval"]["primary_k"]
    sort_col = f"recall@{pk}"
    if sort_col in df.columns:
        df = df.sort_values(sort_col, ascending=False).reset_index(drop=True)
    return df
