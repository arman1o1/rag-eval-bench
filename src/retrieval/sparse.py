"""Sparse retrieval using a serialised BM25 index.

The indexer (in the ingestion pipeline) persists two pickle artefacts:
  1. ``bm25_index.pkl``  — a fitted ``rank_bm25.BM25Okapi`` instance.
  2. ``bm25_chunks.pkl`` — a ``list[dict]`` with keys ``text`` and ``metadata``
     aligned positionally with the BM25 corpus.

This module loads both and performs keyword retrieval at query time.
"""

from __future__ import annotations

import logging
import pickle
import re
from pathlib import Path

from rank_bm25 import BM25Okapi

from src.retrieval.dense import RetrievedChunk

logger = logging.getLogger(__name__)


def _tokenize(text: str) -> list[str]:
    """Lowercase whitespace+punctuation tokenizer for BM25 queries."""
    return re.findall(r"\w+", text.lower())


class SparseRetriever:
    """Keyword-based retriever backed by a BM25Okapi index on disk.

    Parameters
    ----------
    index_path:
        Path to the pickled dictionary containing the BM25Okapi object
        and chunk texts/metadata.
    """

    def __init__(
        self,
        index_path: str | Path,
    ) -> None:
        self._index_path = Path(index_path)

        self._bm25: BM25Okapi | None = None
        self._chunks: list[dict] = []

        self._load()

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def _load(self) -> None:
        """Deserialise the BM25 index and its companion chunk store."""
        if not self._index_path.exists():
            raise FileNotFoundError(
                f"BM25 index not found at {self._index_path}. "
                "Run the ingestion pipeline first."
            )

        with open(self._index_path, "rb") as f:
            data = pickle.load(f)  # noqa: S301

        self._bm25 = data.get("bm25")

        # Reconstruct chunks from serialized texts and metadatas
        self._chunks = []
        texts = data.get("texts", [])
        metadatas = data.get("metadatas", [])
        for text, meta in zip(texts, metadatas):
            self._chunks.append({"text": text, "metadata": meta})

        logger.info(
            "SparseRetriever loaded — %d documents from %s",
            len(self._chunks),
            self._index_path,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def retrieve(self, query: str, top_k: int = 10) -> list[RetrievedChunk]:
        """Return the *top_k* BM25-scored chunks for *query*.

        Parameters
        ----------
        query:
            Natural-language query string.
        top_k:
            Maximum number of results to return.

        Returns
        -------
        list[RetrievedChunk]
            Chunks sorted by descending BM25 score.
        """
        if self._bm25 is None or not self._chunks:
            logger.warning("SparseRetriever has no loaded index — returning empty.")
            return []

        tokenized_query = _tokenize(query)
        scores = self._bm25.get_scores(tokenized_query)

        # Get top-k indices sorted by descending score.
        top_indices = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True,
        )[:top_k]

        results: list[RetrievedChunk] = []
        for idx in top_indices:
            score = float(scores[idx])
            if score <= 0.0:
                break  # BM25 scores below 0 are irrelevant
            chunk = self._chunks[idx]
            results.append(
                RetrievedChunk(
                    text=chunk["text"],
                    score=score,
                    metadata=chunk.get("metadata", {}),
                )
            )

        logger.debug("Sparse retrieval returned %d chunks for query: %.80s…", len(results), query)
        return results
