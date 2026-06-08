"""Chunking strategies for splitting documents into retrieval-friendly chunks."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod

import numpy as np
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import ChunkingConfig, ChunkingStrategy
from src.ingestion.embedder import EmbeddingService

logger = logging.getLogger(__name__)


# ======================================================================
# Abstract base
# ======================================================================


class BaseChunker(ABC):
    """Interface that every chunking strategy must implement."""

    def __init__(self, config: ChunkingConfig) -> None:
        self.config = config

    @abstractmethod
    def chunk(self, docs: list[Document]) -> list[Document]:
        """Split *docs* into smaller chunks, preserving metadata.

        Args:
            docs: Source documents (typically one per page).

        Returns:
            Chunked documents with inherited + added metadata.
        """

    # ------------------------------------------------------------------
    # Helpers shared across strategies
    # ------------------------------------------------------------------

    @staticmethod
    def _carry_metadata(
        source_doc: Document,
        chunk_text: str,
        chunk_index: int,
    ) -> Document:
        """Create a new ``Document`` that inherits the source's metadata."""
        meta = {**source_doc.metadata, "chunk_index": chunk_index}
        return Document(page_content=chunk_text, metadata=meta)


# ======================================================================
# Concrete strategies
# ======================================================================


class FixedChunker(BaseChunker):
    """Split by fixed character count with overlap."""

    def chunk(self, docs: list[Document]) -> list[Document]:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap,
            separators=["\n\n", "\n", " ", ""],
        )
        return self._split(docs, splitter)

    def _split(
        self,
        docs: list[Document],
        splitter: RecursiveCharacterTextSplitter,
    ) -> list[Document]:
        chunks: list[Document] = []
        idx = 0
        for doc in docs:
            parts = splitter.split_text(doc.page_content)
            for part in parts:
                chunks.append(self._carry_metadata(doc, part, idx))
                idx += 1
        logger.info("FixedChunker produced %d chunks", len(chunks))
        return chunks


class RecursiveChunker(BaseChunker):
    """Recursive splitting with smart, ordered separators."""

    def chunk(self, docs: list[Document]) -> list[Document]:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
        chunks: list[Document] = []
        idx = 0
        for doc in docs:
            parts = splitter.split_text(doc.page_content)
            for part in parts:
                chunks.append(self._carry_metadata(doc, part, idx))
                idx += 1
        logger.info("RecursiveChunker produced %d chunks", len(chunks))
        return chunks


class SemanticChunker(BaseChunker):
    """Embed sentences and split where cosine similarity drops below a threshold.

    Requires an :class:`EmbeddingService` to compute per-sentence embeddings.
    """

    def __init__(
        self,
        config: ChunkingConfig,
        embedding_service: EmbeddingService | None = None,
    ) -> None:
        super().__init__(config)
        self.embedding_service = embedding_service or EmbeddingService()

    def chunk(self, docs: list[Document]) -> list[Document]:
        chunks: list[Document] = []
        global_idx = 0

        for doc in docs:
            sentences = self._split_into_sentences(doc.page_content)
            if len(sentences) <= 1:
                chunks.append(self._carry_metadata(doc, doc.page_content, global_idx))
                global_idx += 1
                continue

            # Embed all sentences for this document-page.
            embeddings = self.embedding_service.embed_texts(sentences)
            emb_array = np.array(embeddings)

            # Compute cosine similarity between consecutive sentences.
            boundaries = self._find_boundaries(emb_array)

            # Build chunks from boundary indices.
            start = 0
            for bnd in boundaries:
                chunk_text = " ".join(sentences[start:bnd]).strip()
                if chunk_text:
                    chunks.append(self._carry_metadata(doc, chunk_text, global_idx))
                    global_idx += 1
                start = bnd

            # Remaining tail.
            tail = " ".join(sentences[start:]).strip()
            if tail:
                chunks.append(self._carry_metadata(doc, tail, global_idx))
                global_idx += 1

        logger.info("SemanticChunker produced %d chunks", len(chunks))
        return chunks

    def _find_boundaries(self, embeddings: np.ndarray) -> list[int]:
        """Return sentence indices where a new chunk should start."""
        threshold = self.config.similarity_threshold
        boundaries: list[int] = []

        for i in range(1, len(embeddings)):
            sim = self._cosine_sim(embeddings[i - 1], embeddings[i])
            if sim < threshold:
                boundaries.append(i)
        return boundaries

    @staticmethod
    def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
        denom = (np.linalg.norm(a) * np.linalg.norm(b))
        if denom == 0:
            return 0.0
        return float(np.dot(a, b) / denom)

    @staticmethod
    def _split_into_sentences(text: str) -> list[str]:
        """Naively split on sentence-ending punctuation."""
        import re

        parts = re.split(r"(?<=[.!?])\s+", text.strip())
        return [s for s in parts if s.strip()]


class SentenceWindowChunker(BaseChunker):
    """Create sentence-level chunks with surrounding-window metadata.

    Each chunk holds a single sentence as ``page_content`` and stores the
    surrounding ±``window_size`` sentences in ``metadata["window"]`` for
    later expansion during retrieval.
    """

    def chunk(self, docs: list[Document]) -> list[Document]:
        window = self.config.window_size
        chunks: list[Document] = []
        global_idx = 0

        for doc in docs:
            sentences = self._split_into_sentences(doc.page_content)
            for i, sentence in enumerate(sentences):
                start = max(0, i - window)
                end = min(len(sentences), i + window + 1)
                window_text = " ".join(sentences[start:end])

                meta = {
                    **doc.metadata,
                    "chunk_index": global_idx,
                    "window": window_text,
                    "sentence_index": i,
                    "window_size": window,
                }
                chunks.append(Document(page_content=sentence, metadata=meta))
                global_idx += 1

        logger.info("SentenceWindowChunker produced %d chunks", len(chunks))
        return chunks

    @staticmethod
    def _split_into_sentences(text: str) -> list[str]:
        import re

        parts = re.split(r"(?<=[.!?])\s+", text.strip())
        return [s for s in parts if s.strip()]


# ======================================================================
# Factory
# ======================================================================


_STRATEGY_MAP: dict[ChunkingStrategy, type[BaseChunker]] = {
    ChunkingStrategy.FIXED: FixedChunker,
    ChunkingStrategy.RECURSIVE: RecursiveChunker,
    ChunkingStrategy.SEMANTIC: SemanticChunker,
    ChunkingStrategy.SENTENCE_WINDOW: SentenceWindowChunker,
}


def get_chunker(
    config: ChunkingConfig,
    *,
    embedding_service: EmbeddingService | None = None,
) -> BaseChunker:
    """Instantiate the appropriate chunker for the given config.

    Args:
        config: Chunking configuration (selects strategy & parameters).
        embedding_service: Required only for the *semantic* strategy.

    Returns:
        A :class:`BaseChunker` instance ready to use.
    """
    cls = _STRATEGY_MAP.get(config.strategy)
    if cls is None:
        raise ValueError(f"Unknown chunking strategy: {config.strategy}")

    if cls is SemanticChunker:
        return cls(config, embedding_service=embedding_service)
    return cls(config)


class Chunker:
    """Wrapper class for testing and client backward compatibility."""

    def __init__(
        self,
        config: ChunkingConfig,
        embedding_service: EmbeddingService | None = None,
    ) -> None:
        self.chunker = get_chunker(config, embedding_service=embedding_service)

    def chunk(self, docs: list[Document]) -> list[Document]:
        return self.chunker.chunk(docs)

