"""Context builder — assembles retrieved chunks into a formatted context string.

Handles relevance filtering, deduplication, sentence-window expansion,
token budget management, and citation-indexed formatting.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from src.config import GenerationConfig

logger = logging.getLogger(__name__)


@dataclass
class RetrievedChunk:
    """Represents a chunk returned by the retrieval layer.

    This is a forward-compatible definition. Once ``src.retrieval.dense``
    is implemented, the canonical class should live there and be imported
    here instead.
    """

    text: str
    score: float = 0.0
    doc_id: str = ""
    chunk_index: int = 0
    metadata: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_CHARS_PER_TOKEN = 4  # rough approximation


def _estimate_tokens(text: str) -> int:
    """Return an approximate token count (4 chars ≈ 1 token)."""
    return max(1, len(text) // _CHARS_PER_TOKEN)


def _strip_text(text: str) -> str:
    """Normalize whitespace for deduplication comparison."""
    return " ".join(text.split())


# ---------------------------------------------------------------------------
# ContextBuilder
# ---------------------------------------------------------------------------


class ContextBuilder:
    """Builds a citation-indexed context string from retrieved chunks.

    Pipeline:
        1. Sentence-window expansion (if metadata available)
        2. Relevance filtering (``min_relevance_score``)
        3. Deduplication (exact match after whitespace normalization)
        4. Token budget truncation (``max_context_tokens``)
        5. Citation-indexed formatting (``[1] …\\n[2] …``)
    """

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def build(
        self,
        chunks: list[RetrievedChunk],
        config: GenerationConfig,
    ) -> str:
        """Assemble a formatted context string from *chunks*.

        Args:
            chunks: Retrieved chunks (already ranked by relevance).
            config: Generation configuration with budget / threshold params.

        Returns:
            A citation-indexed context string ready to inject into a prompt.
        """
        if not chunks:
            logger.warning("build() called with an empty chunk list.")
            return ""

        # Step 1 — sentence-window expansion
        expanded = self._expand_sentence_windows(chunks)

        # Step 2 — relevance filtering
        filtered = self._filter_by_relevance(expanded, config.min_relevance_score)
        if not filtered:
            logger.warning(
                "All chunks dropped by relevance filter (threshold=%.2f).",
                config.min_relevance_score,
            )
            return ""

        # Step 3 — deduplication
        deduped = self._deduplicate(filtered)

        # Step 4 — token budget management
        budget_chunks = self._apply_token_budget(deduped, config.max_context_tokens)
        if not budget_chunks:
            logger.warning(
                "No chunks fit within the token budget (%d tokens).",
                config.max_context_tokens,
            )
            return ""

        # Step 5 — citation-indexed formatting
        return self._format(budget_chunks)

    # ------------------------------------------------------------------ #
    # Internal steps
    # ------------------------------------------------------------------ #

    @staticmethod
    def _expand_sentence_windows(chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
        """Expand retrieval chunks using stored sentence-window metadata.

        If a chunk's metadata contains ``window_text`` (populated by the
        sentence-window chunking strategy), replace the chunk text with
        the full surrounding window while preserving the original score.
        """
        expanded: list[RetrievedChunk] = []
        for chunk in chunks:
            window_text: str | None = chunk.metadata.get("window_text")
            if window_text:
                logger.debug(
                    "Expanding chunk %s/%d from %d → %d chars via sentence-window.",
                    chunk.doc_id,
                    chunk.chunk_index,
                    len(chunk.text),
                    len(window_text),
                )
                expanded.append(
                    RetrievedChunk(
                        text=window_text,
                        score=chunk.score,
                        doc_id=chunk.doc_id,
                        chunk_index=chunk.chunk_index,
                        metadata=chunk.metadata,
                    )
                )
            else:
                expanded.append(chunk)
        return expanded

    @staticmethod
    def _filter_by_relevance(
        chunks: list[RetrievedChunk],
        min_score: float,
    ) -> list[RetrievedChunk]:
        """Drop chunks whose relevance score is below *min_score*."""
        before = len(chunks)
        kept = [c for c in chunks if c.score >= min_score]
        dropped = before - len(kept)
        if dropped:
            logger.info(
                "Relevance filter: kept %d / %d chunks (min_score=%.2f).",
                len(kept),
                before,
                min_score,
            )
        return kept

    @staticmethod
    def _deduplicate(chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
        """Remove near-duplicate chunks (exact text match after stripping)."""
        seen: set[str] = set()
        unique: list[RetrievedChunk] = []
        for chunk in chunks:
            normalized = _strip_text(chunk.text)
            if normalized not in seen:
                seen.add(normalized)
                unique.append(chunk)
            else:
                logger.debug(
                    "Dropped duplicate chunk (doc=%s, idx=%d).",
                    chunk.doc_id,
                    chunk.chunk_index,
                )
        if len(unique) < len(chunks):
            logger.info("Deduplication: %d → %d chunks.", len(chunks), len(unique))
        return unique

    @staticmethod
    def _apply_token_budget(
        chunks: list[RetrievedChunk],
        max_tokens: int,
    ) -> list[RetrievedChunk]:
        """Keep as many top-ranked chunks as fit within *max_tokens*.

        Chunks are assumed to arrive in relevance-descending order, so the
        most important ones are kept first.  If a single chunk exceeds the
        remaining budget it is truncated (at a character boundary) to fit.
        """
        kept: list[RetrievedChunk] = []
        used_tokens = 0

        for chunk in chunks:
            chunk_tokens = _estimate_tokens(chunk.text)
            if used_tokens + chunk_tokens <= max_tokens:
                kept.append(chunk)
                used_tokens += chunk_tokens
            else:
                # Try to fit a truncated version of this chunk
                remaining_tokens = max_tokens - used_tokens
                if remaining_tokens > 20:  # only bother if meaningful space left
                    max_chars = remaining_tokens * _CHARS_PER_TOKEN
                    truncated_text = chunk.text[:max_chars].rsplit(" ", 1)[0] + " …"
                    kept.append(
                        RetrievedChunk(
                            text=truncated_text,
                            score=chunk.score,
                            doc_id=chunk.doc_id,
                            chunk_index=chunk.chunk_index,
                            metadata=chunk.metadata,
                        )
                    )
                    logger.debug(
                        "Truncated chunk %s/%d to fit token budget.",
                        chunk.doc_id,
                        chunk.chunk_index,
                    )
                break  # budget exhausted

        logger.info(
            "Token budget: kept %d chunks (~%d tokens / %d max).",
            len(kept),
            used_tokens,
            max_tokens,
        )
        return kept

    @staticmethod
    def _format(chunks: list[RetrievedChunk]) -> str:
        """Format chunks as a citation-indexed context string.

        Example output::

            [1] First chunk text here…
            [2] Second chunk text here…
        """
        lines: list[str] = []
        for idx, chunk in enumerate(chunks, start=1):
            lines.append(f"[{idx}] {chunk.text}")
        return "\n".join(lines)
