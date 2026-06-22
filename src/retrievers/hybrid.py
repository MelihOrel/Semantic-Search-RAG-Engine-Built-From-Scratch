"""Hybrid retriever: Reciprocal Rank Fusion of a dense + a sparse retriever.

RRF is rank-based, so it sidesteps the score-scale mismatch between cosine
similarity (dense) and BM25 (sparse). Score for a document d:

    RRF(d) = sum_r  1 / (rrf_k + rank_r(d))

where the sum is over each retriever r in which d appears, and rank is 0-based.
We over-fetch each base retriever (k * pool_factor) before fusing so good docs
ranked low by one retriever still surface.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Dict, List

from . import BaseRetriever, SearchResult
from ..chunking import Chunk


class HybridRRFRetriever(BaseRetriever):
    name = "hybrid_rrf"

    def __init__(
        self,
        dense: BaseRetriever,
        sparse: BaseRetriever,
        rrf_k: int = 60,
        pool_factor: int = 5,
    ):
        self.dense = dense
        self.sparse = sparse
        self.rrf_k = rrf_k
        self.pool_factor = pool_factor
        self.name = f"hybrid_rrf[{dense.name}+{sparse.name}]"
        # Cache chunk text by chunk_id so fused results carry their text.
        self._text: Dict[str, str] = {}

    def index(self, chunks: List[Chunk]) -> None:
        # Assumes the wrapped retrievers are already indexed on the same chunks;
        # we only need the text lookup here. (main.py indexes the bases.)
        self._text = {c.chunk_id: c.text for c in chunks}

    def search(self, query: str, k: int) -> List[SearchResult]:
        pool = max(k * self.pool_factor, k)
        dense_hits = self.dense.search(query, pool)
        sparse_hits = self.sparse.search(query, pool)

        fused: Dict[str, float] = defaultdict(float)
        parent: Dict[str, str] = {}
        for hits in (dense_hits, sparse_hits):
            for h in hits:
                fused[h.chunk_id] += 1.0 / (self.rrf_k + h.rank)
                parent[h.chunk_id] = h.complaint_id
                self._text.setdefault(h.chunk_id, h.text)

        ranked = sorted(fused.items(), key=lambda kv: kv[1], reverse=True)[:k]
        return [
            SearchResult(
                chunk_id=cid,
                complaint_id=parent[cid],
                score=score,
                text=self._text.get(cid, ""),
                rank=rank,
            )
            for rank, (cid, score) in enumerate(ranked)
        ]
