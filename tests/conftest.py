"""Shared pytest fixtures for RAG Eval Bench tests.

Provides reusable fake documents, chunks, embeddings, Qdrant client,
and evaluation items for deterministic unit testing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest

from src.evaluation.dataset import EvalItem

# ---------------------------------------------------------------------------
# Document / Chunk models used by the pipeline
# ---------------------------------------------------------------------------
# These mirror what the ingestion layer produces. Defined here so tests
# don't depend on ingestion module existing yet.

@dataclass
class Document:
    """Lightweight document representation for tests."""

    text: str
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def page_content(self) -> str:
        """LangChain-compatible alias."""
        return self.text


@dataclass
class RetrievedChunk:
    """Lightweight retrieved chunk representation for tests."""

    text: str
    score: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Fixtures: sample documents
# ---------------------------------------------------------------------------

LEGAL_TEXTS = [
    (
        "This Agreement is entered into as of January 1, 2024, by and between "
        "Acme Corporation ('Company') and Widget Industries Inc. ('Vendor'). "
        "The Company agrees to purchase software services from the Vendor for "
        "a period of 24 months, subject to the terms and conditions set forth herein."
    ),
    (
        "Section 5: Confidentiality. Each party agrees to hold in confidence all "
        "proprietary and confidential information received from the other party. "
        "This obligation shall survive termination of this Agreement for a period "
        "of three (3) years. Confidential Information does not include information "
        "that is publicly available through no fault of the receiving party."
    ),
    (
        "Section 8: Limitation of Liability. In no event shall either party be "
        "liable for any indirect, incidental, special, consequential, or punitive "
        "damages, regardless of the cause of action or the theory of liability. "
        "The total aggregate liability of the Vendor shall not exceed the amounts "
        "paid by the Company in the twelve (12) months preceding the claim."
    ),
    (
        "Section 12: Governing Law and Dispute Resolution. This Agreement shall "
        "be governed by and construed in accordance with the laws of the State of "
        "Delaware, without regard to its conflict of laws principles. Any dispute "
        "arising under this Agreement shall be resolved through binding arbitration "
        "in Wilmington, Delaware."
    ),
    (
        "Section 15: Intellectual Property. All intellectual property rights in "
        "any deliverables created by the Vendor under this Agreement shall be "
        "assigned to the Company upon full payment. The Vendor retains rights to "
        "pre-existing tools and methodologies used in creating the deliverables, "
        "and grants the Company a perpetual, royalty-free license thereto."
    ),
]


@pytest.fixture
def sample_documents() -> list[Document]:
    """3-5 fake Document objects with realistic legal text and metadata."""
    return [
        Document(
            text=text,
            metadata={
                "source": f"contract_{i + 1:03d}.pdf",
                "page": i + 1,
                "section": f"Section {i + 1}",
            },
        )
        for i, text in enumerate(LEGAL_TEXTS)
    ]


@pytest.fixture
def sample_chunks(sample_documents: list[Document]) -> list[Document]:
    """Pre-chunked documents — each document split roughly in half."""
    chunks: list[Document] = []
    for doc in sample_documents:
        words = doc.text.split()
        mid = len(words) // 2
        for j, slice_ in enumerate([words[:mid], words[mid:]]):
            chunks.append(
                Document(
                    text=" ".join(slice_),
                    metadata={
                        **doc.metadata,
                        "chunk_index": j,
                    },
                )
            )
    return chunks


# ---------------------------------------------------------------------------
# Fixtures: mock embeddings
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_embeddings() -> list[list[float]]:
    """Deterministic fake embeddings (seeded random, 128-dim for speed)."""
    rng = np.random.RandomState(42)
    return [rng.randn(128).tolist() for _ in range(10)]


@pytest.fixture
def embedding_fn(mock_embeddings: list[list[float]]):
    """A mock embedding function returning deterministic vectors."""
    counter = {"idx": 0}

    def _embed(texts: list[str]) -> list[list[float]]:
        results = []
        for _ in texts:
            results.append(mock_embeddings[counter["idx"] % len(mock_embeddings)])
            counter["idx"] += 1
        return results

    return _embed


# ---------------------------------------------------------------------------
# Fixtures: Qdrant client (in-memory)
# ---------------------------------------------------------------------------

@pytest.fixture
def qdrant_client():
    """Qdrant client in in-memory mode for testing.

    Falls back to a MagicMock if qdrant-client isn't installed.
    """
    try:
        from qdrant_client import QdrantClient

        client = QdrantClient(location=":memory:")
        yield client
        # Cleanup: drop any collections created during tests
        for collection in client.get_collections().collections:
            client.delete_collection(collection.name)
    except ImportError:
        yield MagicMock()


# ---------------------------------------------------------------------------
# Fixtures: evaluation items
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_eval_items() -> list[EvalItem]:
    """Fake EvalItem objects for evaluation tests."""
    return [
        EvalItem(
            question="Who are the parties in this agreement?",
            ground_truth_answer="Acme Corporation and Widget Industries Inc.",
            source_doc="contract_001.pdf",
            source_page=1,
            source_section="Parties",
            context_texts=[LEGAL_TEXTS[0]],
        ),
        EvalItem(
            question="How long does the confidentiality obligation last?",
            ground_truth_answer="Three years after termination.",
            source_doc="contract_001.pdf",
            source_page=2,
            source_section="Confidentiality",
            context_texts=[LEGAL_TEXTS[1]],
        ),
        EvalItem(
            question="What is the liability cap for the Vendor?",
            ground_truth_answer="The amounts paid by the Company in the preceding twelve months.",
            source_doc="contract_001.pdf",
            source_page=3,
            source_section="Limitation of Liability",
            context_texts=[LEGAL_TEXTS[2]],
        ),
        EvalItem(
            question="Which state's law governs this agreement?",
            ground_truth_answer="The laws of the State of Delaware.",
            source_doc="contract_001.pdf",
            source_page=4,
            source_section="Governing Law",
            context_texts=[LEGAL_TEXTS[3]],
        ),
        EvalItem(
            question="Who owns the IP in deliverables?",
            ground_truth_answer="The Company owns the IP upon full payment.",
            source_doc="contract_001.pdf",
            source_page=5,
            source_section="Intellectual Property",
            context_texts=[LEGAL_TEXTS[4]],
        ),
    ]
