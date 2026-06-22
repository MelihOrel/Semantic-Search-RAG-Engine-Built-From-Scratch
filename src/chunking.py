"""Chunking strategies — a first-class, swappable experiment.

Three strategies share one interface so the evaluation harness can ablate
across them on identical settings:

* ``WholeDocumentChunker``  — one chunk per complaint (baseline).
* ``FixedSizeChunker``      — fixed word count with configurable overlap.
* ``SentenceWindowChunker`` — N-sentence windows with sentence overlap.

Every emitted ``Chunk`` keeps a back-reference to its parent ``complaint_id``
so retrieval results can be scored against document-level labels.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List

import pandas as pd

# Lightweight sentence splitter: split on sentence-final punctuation followed
# by whitespace + a capital / quote / digit. Good enough for complaint prose
# and avoids a heavyweight NLP dependency. Newlines also act as boundaries.
_SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[\"'A-Z0-9\[])|\n+")
_WORD_RE = re.compile(r"\S+")


@dataclass
class Chunk:
    """A unit of indexable text plus provenance back to its parent document."""
    chunk_id: str
    complaint_id: str
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)


def split_sentences(text: str) -> List[str]:
    """Split text into sentences, dropping empties."""
    parts = _SENT_SPLIT_RE.split(text.strip())
    return [p.strip() for p in parts if p and p.strip()]


class BaseChunker:
    """Common interface. Subclasses implement ``_chunk_text``."""

    name: str = "base"

    def _chunk_text(self, text: str) -> List[str]:
        raise NotImplementedError

    def chunk_corpus(self, corpus: pd.DataFrame) -> List[Chunk]:
        """Apply the strategy to every document, carrying metadata through."""
        meta_cols = [c for c in corpus.columns if c not in ("narrative",)]
        chunks: List[Chunk] = []
        for _, row in corpus.iterrows():
            cid = str(row["complaint_id"])
            meta = {c: row[c] for c in meta_cols}
            for i, piece in enumerate(self._chunk_text(row["narrative"])):
                chunks.append(
                    Chunk(
                        chunk_id=f"{cid}::{i}",
                        complaint_id=cid,
                        text=piece,
                        metadata=meta,
                    )
                )
        return chunks


class WholeDocumentChunker(BaseChunker):
    name = "whole_document"

    def _chunk_text(self, text: str) -> List[str]:
        return [text] if text.strip() else []


class FixedSizeChunker(BaseChunker):
    name = "fixed_size"

    def __init__(self, chunk_size: int = 180, overlap: int = 40):
        if overlap >= chunk_size:
            raise ValueError("overlap must be smaller than chunk_size")
        self.chunk_size = chunk_size
        self.overlap = overlap

    def _chunk_text(self, text: str) -> List[str]:
        words = _WORD_RE.findall(text)
        if not words:
            return []
        if len(words) <= self.chunk_size:
            return [" ".join(words)]
        step = self.chunk_size - self.overlap
        out: List[str] = []
        for start in range(0, len(words), step):
            window = words[start : start + self.chunk_size]
            if window:
                out.append(" ".join(window))
            if start + self.chunk_size >= len(words):
                break
        return out


class SentenceWindowChunker(BaseChunker):
    name = "sentence_window"

    def __init__(self, window_size: int = 4, overlap: int = 1):
        if overlap >= window_size:
            raise ValueError("overlap must be smaller than window_size")
        self.window_size = window_size
        self.overlap = overlap

    def _chunk_text(self, text: str) -> List[str]:
        sents = split_sentences(text)
        if not sents:
            return []
        if len(sents) <= self.window_size:
            return [" ".join(sents)]
        step = self.window_size - self.overlap
        out: List[str] = []
        for start in range(0, len(sents), step):
            window = sents[start : start + self.window_size]
            if window:
                out.append(" ".join(window))
            if start + self.window_size >= len(sents):
                break
        return out


def get_chunker(config: Dict[str, Any], strategy: str | None = None) -> BaseChunker:
    """Factory: build the chunker named in config (or the override)."""
    ccfg = config["chunking"]
    strategy = strategy or ccfg["strategy"]
    if strategy == "whole_document":
        return WholeDocumentChunker()
    if strategy == "fixed_size":
        p = ccfg["fixed_size"]
        return FixedSizeChunker(p["chunk_size"], p["overlap"])
    if strategy == "sentence_window":
        p = ccfg["sentence_window"]
        return SentenceWindowChunker(p["window_size"], p["overlap"])
    raise ValueError(f"Unknown chunking strategy: {strategy}")


ALL_STRATEGIES = ["whole_document", "fixed_size", "sentence_window"]
