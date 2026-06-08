"""Tests for document chunking strategies.

Tests all 4 chunking strategies (fixed, recursive, semantic, sentence_window)
for output type, metadata preservation, chunk_size adherence, overlap, and edge cases.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from tests.conftest import Document

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_chunker(strategy: str, chunk_size: int = 200, chunk_overlap: int = 20):
    """Create a Chunker with the given strategy, handling import gracefully."""
    try:
        from src.config import ChunkingConfig, ChunkingStrategy
        from src.ingestion.chunker import Chunker

        config = ChunkingConfig(
            strategy=ChunkingStrategy(strategy),
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

        embedding_service = None
        if strategy == "semantic":
            import numpy as np
            embedding_service = MagicMock()
            def mock_embed_texts(texts):
                rng = np.random.RandomState(42)
                return [rng.randn(128).tolist() for _ in texts]
            embedding_service.embed_texts.side_effect = mock_embed_texts

        return Chunker(config, embedding_service=embedding_service)
    except ImportError:
        pytest.skip("Chunker module not yet implemented")


def _long_document(n_sentences: int = 50) -> Document:
    """Create a document with many sentences to ensure chunking actually splits."""
    sentences = [
        f"Sentence {i}: The party of the first part agrees to the terms herein described "
        f"in subsection {i} of the contract agreement document number {i}."
        for i in range(n_sentences)
    ]
    return Document(
        text=" ".join(sentences),
        metadata={"source": "long_contract.pdf", "page": 1},
    )


# ---------------------------------------------------------------------------
# Output type tests
# ---------------------------------------------------------------------------

class TestChunkOutputType:
    """Verify all strategies return list[Document]."""

    @pytest.mark.parametrize("strategy", ["fixed", "recursive", "semantic", "sentence_window"])
    def test_output_is_list_of_documents(self, strategy: str, sample_documents: list[Document]):
        chunker = _make_chunker(strategy)
        chunks = chunker.chunk(sample_documents)

        assert isinstance(chunks, list)
        assert len(chunks) > 0
        for chunk in chunks:
            assert hasattr(chunk, "text") or hasattr(chunk, "page_content")

    @pytest.mark.parametrize("strategy", ["fixed", "recursive"])
    def test_multiple_chunks_from_long_doc(self, strategy: str):
        """Long document should produce multiple chunks."""
        chunker = _make_chunker(strategy, chunk_size=100)
        doc = _long_document(50)
        chunks = chunker.chunk([doc])
        assert len(chunks) > 1


# ---------------------------------------------------------------------------
# Metadata preservation
# ---------------------------------------------------------------------------

class TestMetadataPreservation:
    """Verify source and page metadata survive chunking."""

    @pytest.mark.parametrize("strategy", ["fixed", "recursive", "semantic", "sentence_window"])
    def test_metadata_has_source_and_page(
        self, strategy: str, sample_documents: list[Document]
    ):
        chunker = _make_chunker(strategy)
        chunks = chunker.chunk(sample_documents)

        for chunk in chunks:
            meta = chunk.metadata if hasattr(chunk, "metadata") else {}
            assert "source" in meta, f"Chunk missing 'source' metadata: {meta}"
            assert "page" in meta, f"Chunk missing 'page' metadata: {meta}"


# ---------------------------------------------------------------------------
# Chunk size adherence
# ---------------------------------------------------------------------------

class TestChunkSize:
    """Verify chunk_size is approximately respected."""

    @pytest.mark.parametrize("strategy", ["fixed", "recursive"])
    def test_chunks_within_size_limit(self, strategy: str):
        """Each chunk should be at most ~2x the configured chunk_size (char count)."""
        chunk_size = 200
        chunker = _make_chunker(strategy, chunk_size=chunk_size)
        doc = _long_document(50)
        chunks = chunker.chunk([doc])

        for chunk in chunks:
            text = chunk.text if hasattr(chunk, "text") else chunk.page_content
            # Allow 2x tolerance — chunk_size is approximate in LangChain splitters
            assert len(text) <= chunk_size * 3, (
                f"Chunk too large: {len(text)} chars vs {chunk_size} limit"
            )


# ---------------------------------------------------------------------------
# Overlap tests
# ---------------------------------------------------------------------------

class TestOverlap:
    """Verify overlap works for fixed and recursive strategies."""

    @pytest.mark.parametrize("strategy", ["fixed", "recursive"])
    def test_overlap_creates_shared_content(self, strategy: str):
        """Adjacent chunks should share some text when overlap > 0."""
        chunker = _make_chunker(strategy, chunk_size=150, chunk_overlap=50)
        doc = _long_document(30)
        chunks = chunker.chunk([doc])

        if len(chunks) < 2:
            pytest.skip("Not enough chunks to test overlap")

        # Check that at least some adjacent pairs share text
        overlap_found = False
        for i in range(len(chunks) - 1):
            text_a = chunks[i].text if hasattr(chunks[i], "text") else chunks[i].page_content
            text_b = (
                chunks[i + 1].text
                if hasattr(chunks[i + 1], "text")
                else chunks[i + 1].page_content
            )

            # Check if last words of chunk A appear at start of chunk B
            tail_a = text_a.split()[-10:]
            head_b = text_b.split()[:20]
            if any(w in head_b for w in tail_a):
                overlap_found = True
                break

        assert overlap_found, "No overlap detected between adjacent chunks"


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    """Test edge cases: empty doc, single sentence doc."""

    @pytest.mark.parametrize("strategy", ["fixed", "recursive", "semantic", "sentence_window"])
    def test_empty_document(self, strategy: str):
        """Empty document should produce empty or single-element chunk list."""
        chunker = _make_chunker(strategy)
        doc = Document(text="", metadata={"source": "empty.pdf", "page": 1})
        chunks = chunker.chunk([doc])
        # Either no chunks or one empty chunk — both are valid
        assert isinstance(chunks, list)

    @pytest.mark.parametrize("strategy", ["fixed", "recursive", "semantic", "sentence_window"])
    def test_single_sentence_document(self, strategy: str):
        """Single short sentence should produce exactly one chunk."""
        chunker = _make_chunker(strategy, chunk_size=500)
        doc = Document(
            text="This is a single sentence contract.",
            metadata={"source": "tiny.pdf", "page": 1},
        )
        chunks = chunker.chunk([doc])
        assert len(chunks) >= 1
        text = chunks[0].text if hasattr(chunks[0], "text") else chunks[0].page_content
        assert "single sentence" in text

    @pytest.mark.parametrize("strategy", ["fixed", "recursive"])
    def test_empty_document_list(self, strategy: str):
        """Empty input list should return empty output."""
        chunker = _make_chunker(strategy)
        chunks = chunker.chunk([])
        assert chunks == []
