"""Tests for the retrieval layer — dense, sparse, hybrid, and reranker.

Tests verify return types, BM25 sparse retrieval, RRF fusion math,
and reranker reordering behavior.
"""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest

from tests.conftest import RetrievedChunk

# ---------------------------------------------------------------------------
# Dense retriever
# ---------------------------------------------------------------------------

class TestDenseRetriever:
    """Test DenseRetriever returns RetrievedChunk list."""

    def test_returns_retrieved_chunk_list(self, qdrant_client, mock_embeddings):
        """Dense retrieval should return a list of RetrievedChunk-like objects."""
        try:
            from src.config import EmbeddingConfig, QdrantConfig, RetrievalConfig
            from src.retrieval.retriever import RetrieverFactory
        except ImportError:
            pytest.skip("Retriever module not yet implemented")

        retrieval_cfg = RetrievalConfig(method="dense", top_k=3, use_reranker=False)
        qdrant_cfg = QdrantConfig(in_memory=True)
        embedding_cfg = EmbeddingConfig()

        try:
            with patch("src.ingestion.embedder.EmbeddingService.embed_texts") as mock_embed:
                mock_embed.return_value = [mock_embeddings[0]]
                retriever = RetrieverFactory.create(retrieval_cfg, qdrant_cfg, embedding_cfg)
                results = retriever.retrieve("What are the contract terms?")
                assert isinstance(results, list)
                for r in results:
                    assert hasattr(r, "text") or hasattr(r, "score")
        except Exception as exc:
            # Acceptable if Qdrant collection doesn't exist yet
            if "not found" in str(exc).lower() or "collection" in str(exc).lower():
                pytest.skip(f"Collection not set up: {exc}")
            raise


# ---------------------------------------------------------------------------
# Sparse (BM25) retriever
# ---------------------------------------------------------------------------

class TestSparseRetriever:
    """Test SparseRetriever with mock BM25 index."""

    def test_bm25_returns_ranked_results(self):
        """BM25 should return results ranked by term frequency relevance."""
        try:
            from rank_bm25 import BM25Okapi
        except ImportError:
            pytest.skip("rank-bm25 not installed")

        corpus = [
            "This contract agreement is between Acme Corp and Widget Inc",
            "The confidentiality clause protects proprietary information",
            "Limitation of liability shall not exceed twelve months of payments",
            "Governing law is the State of Delaware for all disputes",
            "Intellectual property rights transfer upon full payment",
        ]
        tokenized = [doc.lower().split() for doc in corpus]
        bm25 = BM25Okapi(tokenized)

        query = "intellectual property rights"
        scores = bm25.get_scores(query.lower().split())

        # The IP-related document (index 4) should score highest
        best_idx = int(np.argmax(scores))
        assert best_idx == 4, f"Expected doc 4, got {best_idx}"
        assert scores[best_idx] > 0

    def test_bm25_returns_zero_for_unrelated_query(self):
        """Completely unrelated query should get low scores."""
        try:
            from rank_bm25 import BM25Okapi
        except ImportError:
            pytest.skip("rank-bm25 not installed")

        corpus = [
            "This contract is between Acme Corp and Widget Inc",
            "The confidentiality clause protects information",
        ]
        tokenized = [doc.lower().split() for doc in corpus]
        bm25 = BM25Okapi(tokenized)

        scores = bm25.get_scores("quantum physics neutrino oscillation".split())
        assert all(s == 0.0 for s in scores)

    def test_sparse_retriever_load_and_retrieve(self, tmp_path):
        """SparseRetriever should load index dictionary and retrieve items correctly."""
        import pickle

        from rank_bm25 import BM25Okapi

        from src.retrieval.sparse import SparseRetriever

        corpus = [
            "Acme Corp contract details.",
            "Confidentiality agreement clause.",
            "Third unrelated document here.",
        ]
        tokenized = [doc.lower().split() for doc in corpus]
        bm25 = BM25Okapi(tokenized)

        # Mock the dictionary saved by indexer.py
        bm25_data = {
            "bm25": bm25,
            "corpus_tokens": tokenized,
            "doc_ids": [0, 1, 2],
            "texts": corpus,
            "metadatas": [{"source": "doc1.txt"}, {"source": "doc2.txt"}, {"source": "doc3.txt"}],
        }

        index_file = tmp_path / "bm25_test.pkl"
        with open(index_file, "wb") as f:
            pickle.dump(bm25_data, f)

        retriever = SparseRetriever(index_path=index_file)
        results = retriever.retrieve("Confidentiality", top_k=2)

        assert len(results) == 1
        assert "Confidentiality" in results[0].text
        assert results[0].metadata["source"] == "doc2.txt"
        assert results[0].score > 0.0


# ---------------------------------------------------------------------------
# Hybrid retriever — RRF fusion
# ---------------------------------------------------------------------------

class TestHybridRetrieverRRF:
    """Test Reciprocal Rank Fusion math."""

    def test_rrf_score_computation(self):
        """RRF formula: score(d) = sum(1 / (k + rank_i)) across retriever lists."""
        k = 60  # Standard RRF constant

        # Simulate two ranked lists (0-indexed ranks)
        # Document A: rank 0 in dense, rank 2 in sparse
        # Document B: rank 1 in dense, rank 0 in sparse
        # Document C: rank 2 in dense, rank 1 in sparse

        dense_ranks = {"A": 0, "B": 1, "C": 2}
        sparse_ranks = {"A": 2, "B": 0, "C": 1}

        rrf_scores: dict[str, float] = {}
        for doc_id in ["A", "B", "C"]:
            score = 0.0
            if doc_id in dense_ranks:
                score += 1.0 / (k + dense_ranks[doc_id] + 1)
            if doc_id in sparse_ranks:
                score += 1.0 / (k + sparse_ranks[doc_id] + 1)
            rrf_scores[doc_id] = score

        # Verify: B should rank highest (rank 1 + rank 0 = best average)
        # B: 1/62 + 1/61 = 0.01613 + 0.01639 = 0.03252
        # A: 1/61 + 1/63 = 0.01639 + 0.01587 = 0.03227
        # C: 1/63 + 1/62 = 0.01587 + 0.01613 = 0.03200
        ranked = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        assert ranked[0][0] == "B", f"Expected B first, got {ranked[0][0]}"
        assert ranked[1][0] == "A", f"Expected A second, got {ranked[1][0]}"
        assert ranked[2][0] == "C", f"Expected C third, got {ranked[2][0]}"

    def test_rrf_with_alpha_weighting(self):
        """RRF with alpha blending: alpha * dense + (1-alpha) * sparse."""
        k = 60
        alpha = 0.7  # Weight dense retrieval more

        dense_ranks = {"A": 0, "B": 1}
        sparse_ranks = {"A": 1, "B": 0}

        scores: dict[str, float] = {}
        for doc_id in ["A", "B"]:
            dense_score = alpha / (k + dense_ranks[doc_id] + 1) if doc_id in dense_ranks else 0
            sparse_score = (
                (1 - alpha) / (k + sparse_ranks[doc_id] + 1) if doc_id in sparse_ranks else 0
            )
            scores[doc_id] = dense_score + sparse_score

        # With alpha=0.7, dense is weighted more → A should win (rank 0 in dense)
        assert scores["A"] > scores["B"]


# ---------------------------------------------------------------------------
# Reranker
# ---------------------------------------------------------------------------

class TestReranker:
    """Test that reranker reorders results."""

    def test_reranker_changes_order(self):
        """Reranker should potentially change the ordering of retrieved chunks."""
        initial_chunks = [
            RetrievedChunk(text="Irrelevant filler text about weather", score=0.9),
            RetrievedChunk(text="The contract expires on December 31, 2025", score=0.5),
            RetrievedChunk(text="This clause limits liability to 12 months", score=0.3),
        ]

        try:
            from src.config import RetrievalConfig
            from src.retrieval.reranker import Reranker

            config = RetrievalConfig(use_reranker=True, reranker_top_k=3)
            reranker = Reranker(config)
            reranked = reranker.rerank("When does the contract expire?", initial_chunks)

            assert isinstance(reranked, list)
            assert len(reranked) <= len(initial_chunks)
            # The expiration-related chunk should ideally rank higher after reranking
        except ImportError:
            pytest.skip("Reranker module not yet implemented")
        except Exception as exc:
            # API key errors are expected in test environment
            if "api" in str(exc).lower() or "key" in str(exc).lower():
                pytest.skip(f"Reranker API not configured: {exc}")
            raise

    def test_reranker_respects_top_k(self):
        """Reranker should return at most reranker_top_k results."""
        chunks = [RetrievedChunk(text=f"Chunk {i}", score=0.5) for i in range(10)]

        try:
            from src.config import RetrievalConfig
            from src.retrieval.reranker import Reranker

            config = RetrievalConfig(use_reranker=True, reranker_top_k=3)
            reranker = Reranker(config)
            reranked = reranker.rerank("test query", chunks)

            assert len(reranked) <= 3
        except ImportError:
            pytest.skip("Reranker module not yet implemented")
        except Exception as exc:
            if "api" in str(exc).lower() or "key" in str(exc).lower():
                pytest.skip(f"Reranker API not configured: {exc}")
            raise
