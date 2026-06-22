"""Main entry point — orchestrates the full pipeline end to end.

Load & Clean → Chunk → Index (dense + BM25 [+ hybrid]) → Evaluate every
(retriever × embedding-model × chunking) combo → Rank & Save Leaderboard →
Visualise → Run RAG example queries.
"""
from __future__ import annotations

import argparse
from typing import Any, Dict, List

import numpy as np

from src.chunking import ALL_STRATEGIES, get_chunker
from src.data_processor import load_or_build_corpus
from src.evaluation import build_query_set, run_leaderboard
from src.rag import EXAMPLE_QUESTIONS, answer_question
from src.retrievers.bm25 import BM25Retriever
from src.retrievers.dense import DenseRetriever
from src.retrievers.hybrid import HybridRRFRetriever
from src.utils import get_logger, load_config, resolve
from src.visualize import (
    plot_chunking_ablation,
    plot_embedding_space,
    plot_retrieval_leaderboard,
)


def _banner(log, title: str) -> None:
    log.info("=" * 66)
    log.info(title)
    log.info("=" * 66)


def build_combos(corpus, config: Dict[str, Any], log) -> List[Dict[str, Any]]:
    """Index every retriever for every (model × chunking) combination."""
    combos: List[Dict[str, Any]] = []
    models = config["embeddings"]["models"]
    hybrid_enabled = config["retrieval"]["hybrid"]["enabled"]
    rrf_k = config["retrieval"]["hybrid"]["rrf_k"]

    for strategy in ALL_STRATEGIES:
        chunker = get_chunker(config, strategy)
        chunks = chunker.chunk_corpus(corpus)
        log.info("Chunked %s -> %d chunks (%s)", "corpus", len(chunks), strategy)

        # BM25 baseline (model-agnostic).
        bm25 = BM25Retriever(); bm25.index(chunks)
        log.info("Indexed %d chunks (%s) into BM25", len(chunks), strategy)
        combos.append({
            "retriever": "bm25", "embedding_model": "-",
            "chunking": strategy, "instance": bm25,
        })

        # Dense retrievers, one per embedding model.
        for spec in models:
            dense = DenseRetriever(config, spec); dense.index(chunks)
            log.info("Indexed %d chunks (%s) into Qdrant [%s]", len(chunks), strategy, spec["name"])
            combos.append({
                "retriever": "dense", "embedding_model": spec["name"],
                "chunking": strategy, "instance": dense,
            })

            if hybrid_enabled:
                hyb = HybridRRFRetriever(dense, bm25, rrf_k=rrf_k); hyb.index(chunks)
                combos.append({
                    "retriever": "hybrid", "embedding_model": spec["name"],
                    "chunking": strategy, "instance": hyb,
                })
    return combos


def main() -> None:
    ap = argparse.ArgumentParser(description="CFPB Semantic Search & RAG Engine")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--force-rebuild", action="store_true", help="rebuild the corpus cache")
    ap.add_argument("--skip-rag", action="store_true", help="skip the RAG demo queries")
    args = ap.parse_args()

    config = load_config(args.config)
    log = get_logger("main", config.get("logging", {}).get("level", "INFO"))

    _banner(log, "STEP 1 — Load & Clean Corpus")
    corpus = load_or_build_corpus(config, force=args.force_rebuild)
    log.info("Working corpus: %d documents", len(corpus))

    _banner(log, "STEP 2-4 — Chunk & Index (dense + BM25 + hybrid)")
    combos = build_combos(corpus, config, log)
    log.info("Built %d (retriever × model × chunking) combinations", len(combos))

    _banner(log, "STEP 5 — Evaluation Harness")
    queries = build_query_set(corpus, config)
    log.info("Query set: %d held-out complaints", len(queries))
    leaderboard = run_leaderboard(combos, queries, corpus, config)

    out_csv = resolve(config["reports"]["leaderboard_csv"])
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    leaderboard.to_csv(out_csv, index=False)
    log.info("Saved leaderboard -> %s", out_csv)

    pk = config["retrieval"]["primary_k"]
    top = leaderboard.iloc[0]
    bm25_rows = leaderboard[leaderboard["retriever"] == "bm25"]
    bm25_best = bm25_rows[f"recall@{pk}"].max() if len(bm25_rows) else float("nan")
    log.info(
        "BM25 baseline — Recall@%d: %.3f", pk, bm25_best,
    )
    uplift = ((top[f"recall@{pk}"] - bm25_best) / bm25_best * 100) if bm25_best else float("nan")
    log.info(
        "WINNER: %s + %s + %s — Recall@%d: %.3f, MRR: %.3f, nDCG@%d: %.3f (beats BM25 by %.0f%%)",
        top["retriever"], top["embedding_model"], top["chunking"],
        pk, top[f"recall@{pk}"], top["mrr"], pk, top[f"ndcg@{pk}"], uplift,
    )

    _banner(log, "STEP 6 — Visualisations")
    plot_retrieval_leaderboard(leaderboard, config)
    plot_chunking_ablation(leaderboard, config)
    # Build an embedding matrix for the projection using the first dense model
    # on whole-document chunks (one vector per complaint).
    try:
        spec = config["embeddings"]["models"][0]
        proj = DenseRetriever(config, spec)
        proj_model = proj._ensure_model()  # noqa: SLF001
        vecs = proj_model.encode(
            corpus["narrative"].tolist(),
            normalize_embeddings=config["embeddings"]["normalize"],
            show_progress_bar=False, convert_to_numpy=True,
        ).astype(np.float32)
        plot_embedding_space(corpus, vecs, config, color_by="product")
        log.info("Saved 3 figures -> %s", resolve(config["reports"]["figures_dir"]))
    except Exception as e:  # noqa: BLE001
        log.warning("Embedding projection skipped: %s", e)

    if not args.skip_rag:
        _banner(log, "STEP 7 — RAG Capstone (grounded, cited answers)")
        # Use the winning retriever instance for the demo.
        winner = next(
            c["instance"] for c in combos
            if c["retriever"] == top["retriever"]
            and c["embedding_model"] == top["embedding_model"]
            and c["chunking"] == top["chunking"]
        )
        for q in EXAMPLE_QUESTIONS:
            ans = answer_question(q, winner, config)
            log.info("Q: %s", q)
            log.info("   confidence=%.3f refused=%s", ans.confidence, ans.refused)
            log.info("   A: %s", ans.answer[:500].replace("\n", " "))
            if ans.citations:
                log.info("   cited complaint_ids: %s", ", ".join(ans.citations[:8]))
            log.info("-" * 66)

    _banner(log, "PIPELINE COMPLETE")


if __name__ == "__main__":
    main()
