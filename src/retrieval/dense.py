"""Dense retrieval via Qdrant vector similarity search.

Defines the canonical ``RetrievedChunk`` dataclass used across the entire
retrieval layer.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from qdrant_client import QdrantClient
from qdrant_client.http.models import FieldCondition, Filter, MatchValue

from src.config import QdrantConfig
from src.ingestion.embedder import EmbeddingService

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Shared data structure
# ---------------------------------------------------------------------------

@dataclass
class RetrievedChunk:
    """A single chunk returned by any retriever."""

    text: str
    score: float
    metadata: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Dense retriever
# ---------------------------------------------------------------------------

class DenseRetriever:
    """Retrieves chunks from Qdrant using dense (vector) similarity search.

    Parameters
    ----------
    qdrant_config:
        Connection and collection settings for Qdrant.
    embedding_service:
        An ``EmbeddingService`` instance used to embed the query string.
    """

    def __init__(
        self,
        qdrant_config: QdrantConfig,
        embedding_service: EmbeddingService,
    ) -> None:
        self._config = qdrant_config
        self._embedder = embedding_service

        if qdrant_config.in_memory:
            self._client = QdrantClient(location=":memory:")
        else:
            self._client = QdrantClient(url=qdrant_config.url)

        logger.info(
            "DenseRetriever initialised — collection=%s",
            qdrant_config.collection_name,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def retrieve(
        self,
        query: str,
        top_k: int = 10,
        filters: dict[str, str] | None = None,
    ) -> list[RetrievedChunk]:
        """Embed *query* and return the *top_k* most similar chunks.

        Parameters
        ----------
        query:
            Natural-language query string.
        top_k:
            Maximum number of results to return.
        filters:
            Optional ``{field: value}`` dict translated to Qdrant exact-match
            filters (e.g. ``{"source": "paper.pdf"}``).

        Returns
        -------
        list[RetrievedChunk]
            Chunks sorted by descending cosine similarity score.
        """
        query_vector = self._embed_query(query)

        qdrant_filter = self._build_filter(filters) if filters else None

        try:
            results = self._client.query_points(
                collection_name=self._config.collection_name,
                query=query_vector,
                limit=top_k,
                query_filter=qdrant_filter,
            ).points
        except Exception:
            logger.exception("Qdrant query failed for collection=%s", self._config.collection_name)
            return []

        chunks: list[RetrievedChunk] = []
        for point in results:
            payload = point.payload or {}
            chunks.append(
                RetrievedChunk(
                    text=payload.get("text", ""),
                    score=point.score if point.score is not None else 0.0,
                    metadata={k: v for k, v in payload.items() if k != "text"},
                )
            )

        logger.debug("Dense retrieval returned %d chunks for query: %.80s…", len(chunks), query)
        return chunks

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _embed_query(self, query: str) -> list[float]:
        """Embed a single query string via the embedding service."""
        vectors = self._embedder.embed_texts([query])
        return vectors[0]

    @staticmethod
    def _build_filter(filters: dict[str, str]) -> Filter:
        """Translate a flat dict into a Qdrant ``Filter``."""
        conditions = [
            FieldCondition(key=key, match=MatchValue(value=value))
            for key, value in filters.items()
        ]
        return Filter(must=conditions)
