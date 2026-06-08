"""Qdrant vector indexer with companion BM25 sparse index."""

from __future__ import annotations

import logging
import pickle
import re
import uuid
from pathlib import Path

from langchain_core.documents import Document
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    PointStruct,
    VectorParams,
)
from rank_bm25 import BM25Okapi

from src.config import QdrantConfig

logger = logging.getLogger(__name__)

# Simple whitespace + punctuation tokeniser for BM25.
_TOKEN_RE = re.compile(r"\w+", re.UNICODE)


class QdrantIndexer:
    """Manage a Qdrant collection and a companion BM25 sparse index.

    Typical usage::

        indexer = QdrantIndexer(config, vector_dim=3072)
        indexer.create_collection()
        indexer.index(chunks, embeddings)
    """

    def __init__(
        self,
        config: QdrantConfig | None = None,
        *,
        vector_dim: int = 3072,
        bm25_path: Path | None = None,
    ) -> None:
        self.config = config or QdrantConfig()
        self.vector_dim = vector_dim
        self.bm25_path = bm25_path or Path("data/bm25_index.pkl")

        if self.config.in_memory:
            self.client = QdrantClient(location=":memory:")
        else:
            self.client = QdrantClient(url=self.config.url)

        logger.info(
            "QdrantIndexer initialised — collection=%s, dim=%d",
            self.config.collection_name,
            self.vector_dim,
        )

    # ------------------------------------------------------------------
    # Collection management
    # ------------------------------------------------------------------

    def create_collection(self, *, recreate: bool = True) -> None:
        """Create (or recreate) the Qdrant collection.

        Args:
            recreate: If *True*, delete the existing collection first.
        """
        name = self.config.collection_name

        if recreate and self.client.collection_exists(name):
            self.client.delete_collection(name)
            logger.info("Deleted existing collection '%s'", name)

        if not self.client.collection_exists(name):
            self.client.create_collection(
                collection_name=name,
                vectors_config=VectorParams(
                    size=self.vector_dim,
                    distance=Distance.COSINE,
                ),
            )
            logger.info(
                "Created collection '%s' (dim=%d, cosine)", name, self.vector_dim
            )

    # ------------------------------------------------------------------
    # Indexing
    # ------------------------------------------------------------------

    def index(
        self,
        chunks: list[Document],
        embeddings: list[list[float]],
        *,
        batch_size: int = 100,
    ) -> None:
        """Upsert chunks + embeddings into Qdrant and build a BM25 index.

        Args:
            chunks: LangChain documents (one per chunk).
            embeddings: Corresponding embedding vectors.
            batch_size: Number of points per upsert call.
        """
        if len(chunks) != len(embeddings):
            raise ValueError(
                f"Mismatch: {len(chunks)} chunks vs {len(embeddings)} embeddings"
            )

        self._upsert_vectors(chunks, embeddings, batch_size)
        self._build_bm25(chunks)

    # ------------------------------------------------------------------
    # Qdrant upsert
    # ------------------------------------------------------------------

    def _upsert_vectors(
        self,
        chunks: list[Document],
        embeddings: list[list[float]],
        batch_size: int,
    ) -> None:
        """Upsert embedding vectors with metadata payloads into Qdrant."""
        collection = self.config.collection_name
        total = len(chunks)

        for start in range(0, total, batch_size):
            end = min(start + batch_size, total)
            points: list[PointStruct] = []

            for i in range(start, end):
                chunk = chunks[i]
                payload = {
                    "text": chunk.page_content,
                    **{k: v for k, v in chunk.metadata.items()},
                }
                points.append(
                    PointStruct(
                        id=str(uuid.uuid4()),
                        vector=embeddings[i],
                        payload=payload,
                    )
                )

            try:
                self.client.upsert(collection_name=collection, points=points)
                logger.debug(
                    "Upserted points %d–%d / %d", start, end - 1, total
                )
            except Exception:
                logger.exception(
                    "Qdrant upsert failed for batch %d–%d", start, end - 1
                )
                raise

        logger.info(
            "Indexed %d vectors into collection '%s'", total, collection
        )

    # ------------------------------------------------------------------
    # BM25 sparse index
    # ------------------------------------------------------------------

    def _build_bm25(self, chunks: list[Document]) -> None:
        """Build a BM25Okapi index and persist it to disk."""
        corpus_tokens: list[list[str]] = []
        doc_ids: list[int] = []

        for idx, chunk in enumerate(chunks):
            tokens = _TOKEN_RE.findall(chunk.page_content.lower())
            corpus_tokens.append(tokens)
            doc_ids.append(idx)

        bm25 = BM25Okapi(corpus_tokens)

        # Store everything needed for retrieval later.
        bm25_data = {
            "bm25": bm25,
            "corpus_tokens": corpus_tokens,
            "doc_ids": doc_ids,
            "texts": [c.page_content for c in chunks],
            "metadatas": [c.metadata for c in chunks],
        }

        self.bm25_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.bm25_path, "wb") as f:
            pickle.dump(bm25_data, f, protocol=pickle.HIGHEST_PROTOCOL)

        logger.info(
            "BM25 index built (%d docs) and saved to %s",
            len(corpus_tokens),
            self.bm25_path,
        )
