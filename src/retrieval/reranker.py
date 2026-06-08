"""Reranking module — Cohere Rerank API or local cross-encoder fallback.

When a Cohere API key is available, the Cohere reranker is preferred for
quality.  Otherwise, a local ``sentence-transformers`` cross-encoder is
loaded as a zero-cost alternative.
"""

from __future__ import annotations

import logging

from src.config import RetrievalConfig
from src.retrieval.dense import RetrievedChunk

logger = logging.getLogger(__name__)


class Reranker:
    """Re-scores and re-orders retrieved chunks for a given query.

    Parameters
    ----------
    config:
        ``RetrievalConfig`` supplying model names and ``reranker_top_k``.
    cohere_api_key:
        If provided (and non-empty), enables the Cohere reranker backend.
    """

    def __init__(
        self,
        config: RetrievalConfig,
        cohere_api_key: str | None = None,
    ) -> None:
        self._config = config
        self._cohere_client = None
        self._cross_encoder = None

        if cohere_api_key:
            self._init_cohere(cohere_api_key)
        else:
            self._init_local()

    # ------------------------------------------------------------------
    # Backend initialisation
    # ------------------------------------------------------------------

    def _init_cohere(self, api_key: str) -> None:
        """Initialise the Cohere reranker client."""
        try:
            import cohere  # noqa: WPS433

            self._cohere_client = cohere.ClientV2(api_key=api_key)
            logger.info(
                "Reranker using Cohere API — model=%s",
                self._config.reranker_model,
            )
        except ImportError:
            logger.warning(
                "cohere package not installed — falling back to local cross-encoder."
            )
            self._init_local()
        except Exception:
            logger.exception("Failed to initialise Cohere client — falling back to local.")
            self._init_local()

    def _init_local(self) -> None:
        """Load a local cross-encoder from sentence-transformers."""
        try:
            from sentence_transformers import CrossEncoder  # noqa: WPS433

            self._cross_encoder = CrossEncoder(self._config.local_reranker_model)
            logger.info(
                "Reranker using local CrossEncoder — model=%s",
                self._config.local_reranker_model,
            )
        except ImportError:
            logger.error(
                "sentence-transformers not installed — reranking will be unavailable."
            )
        except Exception:
            logger.exception("Failed to load local CrossEncoder.")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def rerank(
        self,
        query: str,
        chunks: list[RetrievedChunk],
        top_k: int | None = None,
    ) -> list[RetrievedChunk]:
        """Re-score *chunks* against *query* and return the top *top_k*.

        Delegates to the Cohere API when available, otherwise uses the local
        cross-encoder.

        Parameters
        ----------
        query:
            The user query.
        chunks:
            Candidate chunks from retrieval.
        top_k:
            Number of top results to keep after reranking. If not provided,
            defaults to config.reranker_top_k.

        Returns
        -------
        list[RetrievedChunk]
            Re-ordered and truncated list of chunks.
        """
        if top_k is None:
            top_k = self._config.reranker_top_k

        if not chunks:
            return []

        if self._cohere_client is not None:
            return self._rerank_cohere(query, chunks, top_k)
        if self._cross_encoder is not None:
            return self._rerank_local(query, chunks, top_k)

        logger.warning("No reranker backend available — returning chunks unchanged.")
        return chunks[:top_k]

    # ------------------------------------------------------------------
    # Cohere backend
    # ------------------------------------------------------------------

    def _rerank_cohere(
        self,
        query: str,
        chunks: list[RetrievedChunk],
        top_k: int,
    ) -> list[RetrievedChunk]:
        """Rerank using the Cohere Rerank v2 API."""
        documents = [chunk.text for chunk in chunks]

        try:
            response = self._cohere_client.rerank(
                model=self._config.reranker_model,
                query=query,
                documents=documents,
                top_n=top_k,
            )
        except Exception:
            logger.exception("Cohere rerank API call failed — returning chunks unchanged.")
            return chunks[:top_k]

        reranked: list[RetrievedChunk] = []
        for result in response.results:
            original = chunks[result.index]
            reranked.append(
                RetrievedChunk(
                    text=original.text,
                    score=result.relevance_score,
                    metadata=original.metadata,
                )
            )

        logger.debug("Cohere reranker returned %d chunks.", len(reranked))
        return reranked

    # ------------------------------------------------------------------
    # Local cross-encoder backend
    # ------------------------------------------------------------------

    def _rerank_local(
        self,
        query: str,
        chunks: list[RetrievedChunk],
        top_k: int,
    ) -> list[RetrievedChunk]:
        """Rerank using a local cross-encoder model."""
        pairs = [(query, chunk.text) for chunk in chunks]

        try:
            scores = self._cross_encoder.predict(pairs)  # type: ignore[union-attr]
        except Exception:
            logger.exception("Local cross-encoder scoring failed — returning chunks unchanged.")
            return chunks[:top_k]

        scored = list(zip(chunks, scores))
        scored.sort(key=lambda x: float(x[1]), reverse=True)

        reranked = [
            RetrievedChunk(
                text=chunk.text,
                score=float(score),
                metadata=chunk.metadata,
            )
            for chunk, score in scored[:top_k]
        ]

        logger.debug("Local cross-encoder reranker returned %d chunks.", len(reranked))
        return reranked
