"""Sparse BM25 retriever — the keyword baseline dense retrieval must beat.

Uses ``rank-bm25`` (Okapi BM25). Tokenisation is deliberately simple and
transparent (lowercase, alphanumeric tokens) so the baseline is honest and
reproducible rather than quietly boosted by heavy preprocessing.
"""
from __future__ import annotations

import re
from typing import List

import numpy as np
from rank_bm25 import BM25Okapi

from . import BaseRetriever, SearchResult
from ..chunking import Chunk

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> List[str]:
    return _TOKEN_RE.findall(text.lower())


class BM25Retriever(BaseRetriever):
    name = "bm25"

    def __init__(self) -> None:
        self._chunks: List[Chunk] = []
        self._bm25: BM25Okapi | None = None

    def index(self, chunks: List[Chunk]) -> None:
        self._chunks = chunks
        corpus_tokens = [tokenize(c.text) for c in chunks]
        # Guard against empty docs which would break BM25's averaging.
        corpus_tokens = [toks if toks else ["__empty__"] for toks in corpus_tokens]
        self._bm25 = BM25Okapi(corpus_tokens)

    def search(self, query: str, k: int) -> List[SearchResult]:
        if self._bm25 is None:
            raise RuntimeError("BM25Retriever.search called before index()")
        scores = self._bm25.get_scores(tokenize(query))
        if len(scores) == 0:
            return []
        top = np.argsort(scores)[::-1][:k]
        results: List[SearchResult] = []
        for rank, idx in enumerate(top):
            c = self._chunks[idx]
            results.append(
                SearchResult(
                    chunk_id=c.chunk_id,
                    complaint_id=c.complaint_id,
                    score=float(scores[idx]),
                    text=c.text,
                    rank=rank,
                )
            )
        return results
