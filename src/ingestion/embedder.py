"""Embedding service supporting OpenAI and local sentence-transformers backends."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

from src.config import EmbeddingConfig, EmbeddingProvider

logger = logging.getLogger(__name__)


class EmbeddingService:
    """Unified embedding interface for OpenAI and local sentence-transformers models.

    Selects backend based on ``EmbeddingConfig.provider`` and handles batched
    embedding transparently.
    """

    def __init__(self, config: EmbeddingConfig | None = None) -> None:
        self.config = config or EmbeddingConfig()
        self._local_model: SentenceTransformer | None = None
        self._openai_client: object | None = None
        logger.info(
            "EmbeddingService initialised — provider=%s", self.config.provider.value
        )

    # ------------------------------------------------------------------
    # Lazy initialisers
    # ------------------------------------------------------------------

    def _get_openai_client(self):
        """Lazily initialise the OpenAI client."""
        if self._openai_client is None:
            from openai import OpenAI  # noqa: WPS433

            self._openai_client = OpenAI()
            logger.info("OpenAI client initialised (model=%s)", self.config.openai_model)
        return self._openai_client

    def _get_local_model(self) -> SentenceTransformer:
        """Lazily load the local sentence-transformers model."""
        if self._local_model is None:
            from sentence_transformers import SentenceTransformer

            self._local_model = SentenceTransformer(self.config.local_model)
            logger.info("Loaded local model: %s", self.config.local_model)
        return self._local_model

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of texts, dispatching to the configured backend.

        Texts are processed in batches of ``config.batch_size`` to stay within
        API / memory limits.

        Args:
            texts: Strings to embed.

        Returns:
            List of embedding vectors (one per input text).
        """
        if not texts:
            return []

        if self.config.provider == EmbeddingProvider.OPENAI:
            return self._embed_openai(texts)
        return self._embed_local(texts)

    def embed_query(self, query: str) -> list[float]:
        """Embed a single query string.

        Args:
            query: The query text.

        Returns:
            Embedding vector for the query.
        """
        return self.embed_texts([query])[0]

    @property
    def dimensions(self) -> int:
        """Return the dimensionality of the configured embedding model."""
        return self.config.dimensions

    # ------------------------------------------------------------------
    # Backend implementations
    # ------------------------------------------------------------------

    def _embed_openai(self, texts: list[str]) -> list[list[float]]:
        """Embed texts via the OpenAI API in batches."""
        client = self._get_openai_client()
        all_embeddings: list[list[float]] = []
        batch_size = self.config.batch_size

        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            try:
                response = client.embeddings.create(
                    input=batch,
                    model=self.config.openai_model,
                    dimensions=self.config.dimensions,
                )
                batch_embeddings = [item.embedding for item in response.data]
                all_embeddings.extend(batch_embeddings)
                logger.debug(
                    "OpenAI batch %d–%d embedded (%d vectors)",
                    start,
                    start + len(batch) - 1,
                    len(batch_embeddings),
                )
            except Exception:
                logger.exception(
                    "OpenAI embedding failed for batch starting at index %d", start
                )
                raise

        return all_embeddings

    def _embed_local(self, texts: list[str]) -> list[list[float]]:
        """Embed texts using a local sentence-transformers model in batches."""
        model = self._get_local_model()
        all_embeddings: list[list[float]] = []
        batch_size = self.config.batch_size

        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            try:
                vectors: np.ndarray = model.encode(
                    batch,
                    show_progress_bar=False,
                    normalize_embeddings=True,
                )
                all_embeddings.extend(vectors.tolist())
                logger.debug(
                    "Local batch %d–%d embedded (%d vectors)",
                    start,
                    start + len(batch) - 1,
                    len(batch),
                )
            except Exception:
                logger.exception(
                    "Local embedding failed for batch starting at index %d", start
                )
                raise

        return all_embeddings
