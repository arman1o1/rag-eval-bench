"""High-level retrieval pipeline and factory.

``Retriever`` chains together:
  1. *Query transformation* (optional) — rewrite / HyDE / multi-query.
  2. *Backend retrieval* — dense, sparse, or hybrid.
  3. *Reranking* (optional) — Cohere or local cross-encoder.

``RetrieverFactory.create_retriever`` builds a ``Retriever`` from a
``Settings`` instance, wiring up all dependencies automatically.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Protocol

from src.config import RetrievalMethod, Settings
from src.retrieval.dense import DenseRetriever, RetrievedChunk
from src.retrieval.hybrid import HybridRetriever
from src.retrieval.query_transform import QueryTransformer
from src.retrieval.reranker import Reranker
from src.retrieval.sparse import SparseRetriever

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Protocols for type-safety without coupling
# ---------------------------------------------------------------------------

class _RetrievalBackend(Protocol):
    """Anything that has a ``retrieve(query, top_k)`` method."""

    def retrieve(self, query: str, top_k: int = 10) -> list[RetrievedChunk]: ...


# ---------------------------------------------------------------------------
# Retriever pipeline
# ---------------------------------------------------------------------------

class Retriever:
    """End-to-end retrieval pipeline: transform → retrieve → rerank.

    Parameters
    ----------
    backend:
        The core retrieval engine (dense / sparse / hybrid).
    reranker:
        Optional ``Reranker``; applied after retrieval if provided.
    query_transformer:
        Optional ``QueryTransformer``; applied before retrieval if provided.
    reranker_top_k:
        Number of results to keep after reranking.
    """

    def __init__(
        self,
        backend: _RetrievalBackend,
        reranker: Reranker | None = None,
        query_transformer: QueryTransformer | None = None,
        reranker_top_k: int = 5,
    ) -> None:
        self._backend = backend
        self._reranker = reranker
        self._query_transformer = query_transformer
        self._reranker_top_k = reranker_top_k

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def retrieve(self, query: str, top_k: int = 10) -> list[RetrievedChunk]:
        """Run the full retrieval pipeline for *query*.

        Parameters
        ----------
        query:
            Natural-language query string.
        top_k:
            Maximum number of final results to return (before reranking
            trims further if enabled).

        Returns
        -------
        list[RetrievedChunk]
            Ranked and (optionally) reranked chunks.
        """
        # Step 1: Query transformation
        queries = self._transform_query(query)

        # Step 2: Retrieval (possibly multi-query)
        if isinstance(queries, list):
            chunks = self._multi_retrieve(queries, top_k)
        else:
            chunks = self._backend.retrieve(queries, top_k=top_k)

        # Step 3: Reranking
        if self._reranker is not None and chunks:
            chunks = self._reranker.rerank(
                query,  # Always rerank against the *original* query.
                chunks,
                top_k=self._reranker_top_k,
            )

        return chunks

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _transform_query(self, query: str) -> str | list[str]:
        """Apply query transformation if configured."""
        if self._query_transformer is None:
            return query
        try:
            return self._query_transformer.transform(query)
        except Exception:
            logger.exception("Query transformation failed — using original query.")
            return query

    def _multi_retrieve(
        self,
        queries: list[str],
        top_k: int,
    ) -> list[RetrievedChunk]:
        """Retrieve for each sub-query and merge with dedup."""
        seen_texts: set[str] = set()
        merged: list[RetrievedChunk] = []

        for q in queries:
            results = self._backend.retrieve(q, top_k=top_k)
            for chunk in results:
                if chunk.text not in seen_texts:
                    seen_texts.add(chunk.text)
                    merged.append(chunk)

        # Sort by score descending and cap at top_k.
        merged.sort(key=lambda c: c.score, reverse=True)
        return merged[:top_k]


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

class RetrieverFactory:
    """Assembles a fully-wired ``Retriever`` from application ``Settings``."""

    @staticmethod
    def create(
        retrieval_cfg: Any,
        qdrant_cfg: Any,
        embedding_cfg: Any,
        cohere_api_key: str | None = None,
        results_dir: str = "results",
    ) -> Retriever:
        """Create a Retriever from individual configs (useful for testing)."""
        from src.config import Settings
        settings = Settings(
            retrieval=retrieval_cfg,
            qdrant=qdrant_cfg,
            embedding=embedding_cfg,
            cohere_api_key=cohere_api_key,
            results_dir=results_dir,
        )
        return RetrieverFactory.create_retriever(settings)

    @staticmethod
    def create_retriever(settings: Settings) -> Retriever:
        """Build a ``Retriever`` from *settings*.

        Wires up the embedding service, retrieval backend, and optional
        reranker based on the current configuration.

        Parameters
        ----------
        settings:
            Top-level application settings.

        Returns
        -------
        Retriever
            A ready-to-use retrieval pipeline.
        """
        from src.ingestion.embedder import EmbeddingService  # noqa: WPS433

        retrieval_cfg = settings.retrieval

        # ---- Embedding service (needed by dense & hybrid) ----
        embedding_service = EmbeddingService(settings.embedding)

        # ---- Build backend ----
        backend: _RetrievalBackend

        if retrieval_cfg.method == RetrievalMethod.DENSE:
            backend = DenseRetriever(settings.qdrant, embedding_service)

        elif retrieval_cfg.method == RetrievalMethod.SPARSE:
            backend = _build_sparse(settings)

        elif retrieval_cfg.method == RetrievalMethod.HYBRID:
            dense = DenseRetriever(settings.qdrant, embedding_service)
            sparse = _build_sparse(settings)
            backend = HybridRetriever(dense, sparse, retrieval_cfg)

        else:
            raise ValueError(f"Unknown retrieval method: {retrieval_cfg.method}")

        # ---- Optional reranker ----
        reranker: Reranker | None = None
        if retrieval_cfg.use_reranker:
            reranker = Reranker(
                config=retrieval_cfg,
                cohere_api_key=settings.cohere_api_key,
            )

        logger.info(
            "RetrieverFactory built pipeline — method=%s, reranker=%s",
            retrieval_cfg.method.value,
            "enabled" if reranker else "disabled",
        )

        return Retriever(
            backend=backend,
            reranker=reranker,
            reranker_top_k=retrieval_cfg.reranker_top_k,
        )


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

def _build_sparse(settings: Settings) -> SparseRetriever:
    """Locate and load the BM25 index from the data directory based on chunking strategy."""
    strategy_val = settings.chunking.strategy.value
    index_path = Path("data") / f"bm25_{strategy_val}.pkl"
    return SparseRetriever(index_path=index_path)
