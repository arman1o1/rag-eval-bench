"""Hybrid retrieval — dense + sparse fused via Reciprocal Rank Fusion (RRF).

RRF formula per document *d*:
    score(d) = alpha * (1 / (k + rank_dense(d)))
             + (1 - alpha) * (1 / (k + rank_sparse(d)))

``alpha`` controls the dense-vs-sparse balance (1.0 = fully dense).
``k`` is the RRF smoothing constant (default 60).

Results are deduplicated by chunk text content.
"""

from __future__ import annotations

import logging
from collections import defaultdict

from src.config import RetrievalConfig
from src.retrieval.dense import DenseRetriever, RetrievedChunk
from src.retrieval.sparse import SparseRetriever

logger = logging.getLogger(__name__)


class HybridRetriever:
    """Merges dense and sparse retrieval results using Reciprocal Rank Fusion.

    Parameters
    ----------
    dense_retriever:
        A ``DenseRetriever`` instance.
    sparse_retriever:
        A ``SparseRetriever`` instance.
    config:
        ``RetrievalConfig`` supplying ``rrf_alpha``, ``rrf_k``, and ``top_k``.
    """

    def __init__(
        self,
        dense_retriever: DenseRetriever,
        sparse_retriever: SparseRetriever,
        config: RetrievalConfig,
    ) -> None:
        self._dense = dense_retriever
        self._sparse = sparse_retriever
        self._alpha = config.rrf_alpha
        self._k = config.rrf_k

        logger.info(
            "HybridRetriever initialised — alpha=%.2f, k=%d",
            self._alpha,
            self._k,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def retrieve(self, query: str, top_k: int = 10) -> list[RetrievedChunk]:
        """Retrieve and fuse results from dense + sparse backends.

        Each backend is asked for ``top_k * 2`` candidates so that
        after deduplication and RRF scoring we still have enough results.

        Parameters
        ----------
        query:
            Natural-language query string.
        top_k:
            Maximum number of fused results to return.

        Returns
        -------
        list[RetrievedChunk]
            Deduplicated chunks sorted by descending RRF score.
        """
        fetch_k = top_k * 2

        dense_results = self._dense.retrieve(query, top_k=fetch_k)
        sparse_results = self._sparse.retrieve(query, top_k=fetch_k)

        fused = self._reciprocal_rank_fusion(dense_results, sparse_results)

        logger.debug(
            "Hybrid retrieval fused %d dense + %d sparse → %d results",
            len(dense_results),
            len(sparse_results),
            min(len(fused), top_k),
        )

        return fused[:top_k]

    # ------------------------------------------------------------------
    # RRF implementation
    # ------------------------------------------------------------------

    def _reciprocal_rank_fusion(
        self,
        dense_results: list[RetrievedChunk],
        sparse_results: list[RetrievedChunk],
    ) -> list[RetrievedChunk]:
        """Compute RRF scores and return a single ranked, deduplicated list."""
        # Accumulate scores keyed by chunk text (for deduplication).
        rrf_scores: dict[str, float] = defaultdict(float)
        chunk_map: dict[str, RetrievedChunk] = {}

        for rank, chunk in enumerate(dense_results, start=1):
            rrf_scores[chunk.text] += self._alpha * (1.0 / (self._k + rank))
            # Keep the first occurrence (highest ranked) metadata.
            if chunk.text not in chunk_map:
                chunk_map[chunk.text] = chunk

        for rank, chunk in enumerate(sparse_results, start=1):
            rrf_scores[chunk.text] += (1.0 - self._alpha) * (1.0 / (self._k + rank))
            if chunk.text not in chunk_map:
                chunk_map[chunk.text] = chunk

        # Build output sorted by fused score descending.
        sorted_texts = sorted(rrf_scores, key=rrf_scores.__getitem__, reverse=True)

        return [
            RetrievedChunk(
                text=text,
                score=rrf_scores[text],
                metadata=chunk_map[text].metadata,
            )
            for text in sorted_texts
        ]
