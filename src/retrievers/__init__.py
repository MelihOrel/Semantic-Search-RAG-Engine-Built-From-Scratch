"""Retrieval engines behind a uniform interface.

Every retriever implements:

    index(chunks: List[Chunk]) -> None
    search(query: str, k: int) -> List[SearchResult]

so the evaluation harness can swap them transparently.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

from ..chunking import Chunk


@dataclass
class SearchResult:
    """One ranked hit: which chunk, which parent document, and the score."""
    chunk_id: str
    complaint_id: str
    score: float
    text: str
    rank: int


class BaseRetriever:
    name: str = "base"
    # How to interpret search scores for the RAG confidence guardrail:
    #   "cosine"   -> bounded [-1, 1], directly comparable to a threshold
    #   "unbounded"-> BM25/RRF style, only relative ordering is meaningful
    score_kind: str = "unbounded"

    def index(self, chunks: List[Chunk]) -> None:
        raise NotImplementedError

    def search(self, query: str, k: int) -> List[SearchResult]:
        raise NotImplementedError
