"""Data engineering & document preparation for the CFPB complaint corpus.

Responsibilities
----------------
1. Load the raw CFPB CSV with Pandas.
2. Filter to rows that actually carry a consumer narrative (the rest are
   metadata-only and useless for a text-retrieval index).
3. Clean the text: normalise the ``XXXX`` redaction runs the CFPB inserts for
   PII, and tidy whitespace *without* destroying sentence boundaries (the
   sentence-window chunker depends on them).
4. Emit one tidy document record per complaint carrying its id, cleaned
   narrative and the labelled metadata used later for evaluation.
5. Persist the corpus to Parquet.

Run standalone:  ``python -m src.data_processor``
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict

import pandas as pd

from .utils import get_logger, load_config, resolve

# Matches one or more whitespace-separated runs of 4+ X's (case-insensitive),
# which is how the CFPB redacts names, account numbers, dates, etc.
_REDACTION_RE = re.compile(r"(?:\bX{2,}\b[\s,./-]*)+", re.IGNORECASE)
# Collapse 3+ newlines to a paragraph break; collapse runs of spaces/tabs.
_MULTINEWLINE_RE = re.compile(r"\n{3,}")
_INLINE_WS_RE = re.compile(r"[ \t]{2,}")


def clean_narrative(text: str, *, collapse: bool, strip: bool, token: str) -> str:
    """Clean a single narrative string.

    Parameters
    ----------
    collapse : replace each run of redaction markers with a single ``token``.
    strip    : remove redaction markers entirely (overrides ``collapse``).
    token    : the replacement marker, e.g. ``[REDACTED]``.
    """
    if not isinstance(text, str):
        return ""
    if strip:
        text = _REDACTION_RE.sub(" ", text)
    elif collapse:
        text = _REDACTION_RE.sub(f" {token} ", text)
    # Normalise whitespace but preserve sentence/paragraph structure.
    text = _MULTINEWLINE_RE.sub("\n\n", text)
    text = _INLINE_WS_RE.sub(" ", text)
    # Trim space that now hugs newlines, then strip ends.
    text = re.sub(r"[ \t]*\n[ \t]*", "\n", text)
    return text.strip()


def build_corpus(config: Dict[str, Any]) -> pd.DataFrame:
    """Load, filter, clean and structure the corpus. Returns a tidy DataFrame."""
    log = get_logger("data_processor", config.get("logging", {}).get("level", "INFO"))
    dcfg, ccfg = config["data"], config["cleaning"]

    raw_path = resolve(dcfg["raw_csv"])
    log.info("Loading raw CSV: %s", raw_path)
    df = pd.read_csv(raw_path, dtype={dcfg["id_column"]: str}, low_memory=False)
    total = len(df)

    text_col = dcfg["text_column"]
    # Filter to non-empty narratives.
    has_text = df[text_col].fillna("").astype(str).str.strip().astype(bool)
    kept = df[has_text].copy()
    dropped = total - len(kept)
    log.info(
        "Corpus: %d narratives retained, %d metadata-only rows dropped",
        len(kept), dropped,
    )

    # Clean text.
    kept["narrative"] = kept[text_col].apply(
        lambda t: clean_narrative(
            t,
            collapse=ccfg["collapse_redactions"],
            strip=ccfg["strip_redactions"],
            token=ccfg["redaction_token"],
        )
    )
    # Drop any rows that became empty after cleaning (defensive).
    before = len(kept)
    kept = kept[kept["narrative"].str.len() > 0].copy()
    if before - len(kept):
        log.info("Dropped %d rows that were empty post-cleaning", before - len(kept))

    # Assemble tidy records: id + narrative + requested metadata columns.
    id_col = dcfg["id_column"]
    meta_cols = [c for c in dcfg["metadata_columns"] if c in kept.columns]
    out = kept[[id_col, "narrative", *meta_cols]].rename(columns={id_col: "complaint_id"})
    out["complaint_id"] = out["complaint_id"].astype(str)
    out = out.reset_index(drop=True)

    # Light reporting on label coverage (these power the eval ground truth).
    if "issue" in out.columns:
        log.info("Distinct `issue` labels: %d", out["issue"].nunique())
    if "product" in out.columns:
        log.info("Distinct `product` labels: %d", out["product"].nunique())
    log.info("Mean narrative length: %.0f chars", out["narrative"].str.len().mean())
    return out


def save_corpus(corpus: pd.DataFrame, config: Dict[str, Any]) -> Path:
    """Persist the corpus to Parquet, creating parent dirs as needed."""
    out_path = resolve(config["data"]["processed_corpus"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    corpus.to_parquet(out_path, index=False)
    return out_path


def load_or_build_corpus(config: Dict[str, Any], force: bool = False) -> pd.DataFrame:
    """Return the cached Parquet corpus if present, else build and cache it."""
    out_path = resolve(config["data"]["processed_corpus"])
    if out_path.exists() and not force:
        return pd.read_parquet(out_path)
    corpus = build_corpus(config)
    save_corpus(corpus, config)
    return corpus


if __name__ == "__main__":
    cfg = load_config()
    corpus = build_corpus(cfg)
    path = save_corpus(corpus, cfg)
    print(f"Saved {len(corpus)} documents -> {path}")
