"""Dense retriever: sentence-transformer embeddings stored in Qdrant.

* Multiple embedding models are supported via config; each is benchmarked
  independently by the evaluation harness.
* Vectors are persisted to a real vector DB (Qdrant) using cosine distance.
  Qdrant runs in embedded/local mode by default (no Docker needed); a server
  mode is supported for scaling. ``pgvector`` is documented in the README as
  the production alternative.
"""
from __future__ import annotations

import uuid
from typing import Any, Dict, List

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels
from sentence_transformers import SentenceTransformer

from . import BaseRetriever, SearchResult
from ..chunking import Chunk
from ..utils import resolve


class DenseRetriever(BaseRetriever):
    score_kind = "cosine"  # Qdrant cosine similarity, bounded in [-1, 1]

    def __init__(self, config: Dict[str, Any], model_spec: Dict[str, Any]):
        self.config = config
        self.model_spec = model_spec
        self.model_name = model_spec["name"]
        self.name = f"dense[{self.model_name}]"
        self.query_prefix = model_spec.get("query_prefix", "")
        self.batch_size = config["embeddings"]["batch_size"]
        self.normalize = config["embeddings"]["normalize"]
        # Collection name is unique per model so multiple models coexist.
        self._collection = f"cfpb_{self.model_name}".replace("-", "_").replace(".", "_")
        # Unique per-instance id so embedded Qdrant storage folders never clash
        # when several retrievers (same model, different chunking) run together.
        self._instance_id = uuid.uuid4().hex[:8]
        self._model: SentenceTransformer | None = None
        self._client: QdrantClient | None = None
        self._dim: int | None = None

    # -- lazy resource construction ---------------------------------------
    def _ensure_model(self) -> SentenceTransformer:
        if self._model is None:
            self._model = SentenceTransformer(self.model_spec["hf_id"])
            self._dim = self._model.get_sentence_embedding_dimension()
        return self._model

    def _ensure_client(self) -> QdrantClient:
        if self._client is None:
            vcfg = self.config["vector_db"]
            if vcfg["mode"] == "server":
                self._client = QdrantClient(host=vcfg["host"], port=vcfg["port"])
            else:
                # Embedded Qdrant locks its storage folder, so each retriever
                # gets an isolated sub-directory (per embedding model) to allow
                # many retrievers to coexist in one pipeline run. At scale,
                # switch vector_db.mode to "server" for true concurrent access.
                base = resolve(vcfg["local_path"])
                path = base / f"{self._collection}_{self._instance_id}"
                path.mkdir(parents=True, exist_ok=True)
                self._client = QdrantClient(path=str(path))
        return self._client

    def close(self) -> None:
        """Release the embedded Qdrant storage lock."""
        if self._client is not None:
            try:
                self._client.close()
            except Exception:  # noqa: BLE001
                pass
            self._client = None

    # -- embedding ---------------------------------------------------------
    def _embed(self, texts: List[str], is_query: bool = False) -> np.ndarray:
        model = self._ensure_model()
        if is_query and self.query_prefix:
            texts = [self.query_prefix + t for t in texts]
        vecs = model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=self.normalize,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return vecs.astype(np.float32)

    # -- interface ---------------------------------------------------------
    def index(self, chunks: List[Chunk]) -> None:
        client = self._ensure_client()
        self._ensure_model()
        vecs = self._embed([c.text for c in chunks])

        # Recreate the collection so re-runs are deterministic. Use the
        # non-deprecated delete-then-create pattern.
        if client.collection_exists(self._collection):
            client.delete_collection(self._collection)
        client.create_collection(
            collection_name=self._collection,
            vectors_config=qmodels.VectorParams(
                size=self._dim, distance=qmodels.Distance.COSINE
            ),
        )
        points = [
            qmodels.PointStruct(
                id=uuid.uuid4().hex,
                vector=vecs[i].tolist(),
                payload={
                    "chunk_id": c.chunk_id,
                    "complaint_id": c.complaint_id,
                    "text": c.text,
                },
            )
            for i, c in enumerate(chunks)
        ]
        # Upsert in batches to keep memory bounded at scale.
        B = 256
        for s in range(0, len(points), B):
            client.upsert(collection_name=self._collection, points=points[s : s + B])

    def search(self, query: str, k: int) -> List[SearchResult]:
        client = self._ensure_client()
        qv = self._embed([query], is_query=True)[0]
        # `query_points` is the current Qdrant query API (replaces `search`).
        response = client.query_points(
            collection_name=self._collection,
            query=qv.tolist(),
            limit=k,
            with_payload=True,
        )
        hits = response.points
        results: List[SearchResult] = []
        for rank, h in enumerate(hits):
            p = h.payload or {}
            results.append(
                SearchResult(
                    chunk_id=p.get("chunk_id", ""),
                    complaint_id=p.get("complaint_id", ""),
                    score=float(h.score),
                    text=p.get("text", ""),
                    rank=rank,
                )
            )
        return results
