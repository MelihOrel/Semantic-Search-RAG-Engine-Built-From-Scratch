"""Visualisation: 300dpi figures for the README and reports.

Produces:
* retrieval_leaderboard.png — Recall/MRR/nDCG across retrievers & models.
* chunking_ablation.png     — how each metric shifts across chunking strategies.
* embedding_space.png       — 2D projection of the corpus coloured by product.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import matplotlib

matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from .utils import get_logger, resolve

sns.set_theme(style="whitegrid", context="talk")
DPI = 300


def _fig_path(config: Dict[str, Any], name: str) -> Path:
    p = resolve(config["reports"]["figures_dir"]) / name
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def plot_retrieval_leaderboard(leaderboard: pd.DataFrame, config: Dict[str, Any]) -> Path:
    """Grouped bars: each metric per (retriever + embedding model) at primary_k."""
    pk = config["retrieval"]["primary_k"]
    metrics = [f"recall@{pk}", "mrr", f"ndcg@{pk}"]
    df = leaderboard.copy()
    df["system"] = df.apply(
        lambda r: r["retriever"]
        + (f"\n{r['embedding_model']}" if r["embedding_model"] not in ("-", "", None) else ""),
        axis=1,
    )
    # Keep the best chunking per system for a clean headline chart.
    df = df.sort_values(f"recall@{pk}", ascending=False).drop_duplicates("system")

    long = df.melt(id_vars="system", value_vars=metrics, var_name="metric", value_name="score")
    fig, ax = plt.subplots(figsize=(max(8, 1.6 * len(df)), 6))
    sns.barplot(data=long, x="system", y="score", hue="metric", ax=ax)
    ax.set_title(f"Retrieval Leaderboard (k={pk})", weight="bold")
    ax.set_xlabel(""); ax.set_ylabel("Score"); ax.set_ylim(0, 1)
    ax.legend(title="", loc="upper right", frameon=True)
    plt.xticks(rotation=20, ha="right")
    fig.tight_layout()
    out = _fig_path(config, "retrieval_leaderboard.png")
    fig.savefig(out, dpi=DPI, bbox_inches="tight"); plt.close(fig)
    return out


def plot_chunking_ablation(leaderboard: pd.DataFrame, config: Dict[str, Any]) -> Path:
    """Line/point plot: metric vs chunking strategy, averaged across retrievers."""
    pk = config["retrieval"]["primary_k"]
    metrics = [f"recall@{pk}", "mrr", f"ndcg@{pk}"]
    agg = leaderboard.groupby("chunking")[metrics].mean().reset_index()
    order = ["whole_document", "fixed_size", "sentence_window"]
    agg["chunking"] = pd.Categorical(agg["chunking"], categories=order, ordered=True)
    agg = agg.sort_values("chunking")
    long = agg.melt(id_vars="chunking", value_vars=metrics, var_name="metric", value_name="score")

    fig, ax = plt.subplots(figsize=(9, 6))
    sns.pointplot(data=long, x="chunking", y="score", hue="metric", ax=ax, markers="o")
    ax.set_title("Chunking Ablation (averaged across retrievers)", weight="bold")
    ax.set_xlabel("Chunking strategy"); ax.set_ylabel("Score"); ax.set_ylim(0, 1)
    ax.legend(title="", loc="best", frameon=True)
    fig.tight_layout()
    out = _fig_path(config, "chunking_ablation.png")
    fig.savefig(out, dpi=DPI, bbox_inches="tight"); plt.close(fig)
    return out


def plot_embedding_space(
    corpus: pd.DataFrame,
    embeddings: np.ndarray,
    config: Dict[str, Any],
    color_by: str = "product",
) -> Path:
    """2D projection (UMAP if available, else t-SNE) coloured by a label."""
    log = get_logger("visualize", config.get("logging", {}).get("level", "INFO"))
    n = len(embeddings)
    try:
        import umap  # type: ignore

        reducer = umap.UMAP(
            n_neighbors=min(15, max(2, n - 1)), min_dist=0.1,
            metric="cosine", random_state=42,
        )
        coords = reducer.fit_transform(embeddings)
        method = "UMAP"
    except Exception as e:  # noqa: BLE001
        from sklearn.manifold import TSNE

        log.info("UMAP unavailable (%s); falling back to t-SNE", type(e).__name__)
        perp = max(5, min(30, n - 1))
        coords = TSNE(n_components=2, perplexity=perp, random_state=42, init="pca").fit_transform(embeddings)
        method = "t-SNE"

    plot_df = pd.DataFrame({"x": coords[:, 0], "y": coords[:, 1]})
    labels = corpus[color_by].astype(str).fillna("unknown").values if color_by in corpus else ["all"] * n
    # Collapse very long product names for the legend.
    plot_df["label"] = [l[:40] + ("…" if len(l) > 40 else "") for l in labels]

    fig, ax = plt.subplots(figsize=(11, 8))
    sns.scatterplot(data=plot_df, x="x", y="y", hue="label", s=60, alpha=0.8, ax=ax)
    ax.set_title(f"Corpus Embedding Space ({method}, coloured by {color_by})", weight="bold")
    ax.set_xlabel(f"{method}-1"); ax.set_ylabel(f"{method}-2")
    ax.legend(title=color_by, bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=9)
    fig.tight_layout()
    out = _fig_path(config, "embedding_space.png")
    fig.savefig(out, dpi=DPI, bbox_inches="tight"); plt.close(fig)
    return out
