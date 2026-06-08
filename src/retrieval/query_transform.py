"""Query transformation strategies for improved retrieval.

Three concrete strategies are provided:

* **QueryRewriter** — rewrites an ambiguous query into a clearer form.
* **HyDE** — generates a *hypothetical* answer and returns it for embedding.
* **MultiQueryGenerator** — generates N sub-questions, enabling multi-angle
  retrieval with downstream dedup.

All strategies subclass ``QueryTransformer`` and accept an
``openai.OpenAI`` client at construction time.
"""

from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod

from openai import OpenAI

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------

class QueryTransformer(ABC):
    """Abstract base for query transformation strategies."""

    def __init__(self, client: OpenAI, model: str = "gpt-4o") -> None:
        self._client = client
        self._model = model

    @abstractmethod
    def transform(self, query: str) -> str | list[str]:
        """Return one or more transformed queries.

        A single string means "replace the original query".
        A list means "retrieve for each and merge".
        """

    def _chat(self, system: str, user: str, temperature: float = 0.3) -> str:
        """Helper: single-turn chat completion returning assistant text."""
        try:
            response = self._client.chat.completions.create(
                model=self._model,
                temperature=temperature,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
            )
            return response.choices[0].message.content or ""
        except Exception:
            logger.exception("LLM call failed during query transformation.")
            return user  # Graceful fallback: return original query


# ---------------------------------------------------------------------------
# Concrete strategies
# ---------------------------------------------------------------------------

class QueryRewriter(QueryTransformer):
    """Rewrites an ambiguous or poorly-formed query into a clearer one."""

    _SYSTEM_PROMPT = (
        "You are a query rewriting assistant for a RAG system. "
        "Given a user query, rewrite it to be clearer, more specific, and "
        "better suited for semantic search. Return ONLY the rewritten query, "
        "nothing else."
    )

    def transform(self, query: str) -> str:
        """Return a single rewritten query."""
        rewritten = self._chat(self._SYSTEM_PROMPT, query)
        logger.debug("QueryRewriter: '%s' → '%s'", query, rewritten)
        return rewritten.strip()


class HyDE(QueryTransformer):
    """Hypothetical Document Embeddings — generates a plausible answer.

    The hypothetical answer is returned for embedding instead of the raw
    query, exploiting the observation that answers live closer to relevant
    documents in embedding space than questions do.
    """

    _SYSTEM_PROMPT = (
        "You are a knowledgeable assistant. Given a question, write a short, "
        "factual paragraph that would directly answer the question. Do NOT "
        "preface with 'Here is …' — just write the answer paragraph."
    )

    def transform(self, query: str) -> str:
        """Return a hypothetical answer string for embedding."""
        hypothetical = self._chat(self._SYSTEM_PROMPT, query, temperature=0.0)
        logger.debug("HyDE generated %d-char hypothetical for: %.80s…", len(hypothetical), query)
        return hypothetical.strip()


class MultiQueryGenerator(QueryTransformer):
    """Generates N diverse sub-questions from a single query.

    Retrieval is performed for each sub-question and results are merged
    with deduplication downstream.
    """

    _SYSTEM_PROMPT = (
        "You are a search-query expansion assistant. Given a user question, "
        "generate {n} diverse search queries that would help retrieve all the "
        "information needed to answer the original question.\n\n"
        "Return a JSON array of strings. Example:\n"
        '[\"query one\", \"query two\", \"query three\"]'
    )

    def __init__(
        self,
        client: OpenAI,
        model: str = "gpt-4o",
        n_queries: int = 3,
    ) -> None:
        super().__init__(client, model)
        self._n_queries = n_queries

    def transform(self, query: str) -> list[str]:
        """Return a list of sub-queries (always includes the original)."""
        system = self._SYSTEM_PROMPT.format(n=self._n_queries)
        raw = self._chat(system, query)

        sub_queries = self._parse_queries(raw, query)
        logger.debug("MultiQueryGenerator produced %d sub-queries.", len(sub_queries))
        return sub_queries

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_queries(raw: str, original: str) -> list[str]:
        """Best-effort parse of the JSON array the LLM returns."""
        # Strip markdown fences if present.
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            lines = cleaned.splitlines()
            cleaned = "\n".join(lines[1:-1])

        try:
            queries = json.loads(cleaned)
            if isinstance(queries, list) and all(isinstance(q, str) for q in queries):
                # Always include the original query.
                return [original] + [q for q in queries if q != original]
        except (json.JSONDecodeError, TypeError):
            logger.warning("Failed to parse MultiQuery LLM output — using original query only.")

        return [original]
