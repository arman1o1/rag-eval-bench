"""RAG generator — LLM-powered answer generation with citation support.

Uses OpenAI's chat completion API to generate answers grounded in
retrieved context, with both synchronous and async streaming interfaces.
"""

from __future__ import annotations

import logging
import re
from collections.abc import AsyncIterator
from dataclasses import dataclass, field

from openai import AsyncOpenAI, OpenAI

from src.config import GenerationConfig, get_settings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Prompt template
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
You are a precise, helpful assistant. Answer the user's question using ONLY \
the provided context. Follow these rules strictly:

1. Cite your sources using bracket notation like [1], [2], etc., corresponding \
to the numbered context chunks.
2. If the context does not contain enough information to answer, say so clearly. \
Do NOT make up information.
3. Combine information from multiple sources when relevant, citing each.
4. Keep your answer concise and well-structured."""

_USER_PROMPT_TEMPLATE = """\
Context:
{context}

Question: {query}

Provide a thorough answer citing the relevant sources."""


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class RAGResponse:
    """Structured response from the RAG generator."""

    answer: str
    citations: list[int] = field(default_factory=list)
    model: str = ""
    usage: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Citation parsing
# ---------------------------------------------------------------------------

_CITATION_RE = re.compile(r"\[(\d+)]")


def parse_citations(text: str) -> list[int]:
    """Extract unique, sorted citation indices from *text*."""
    return sorted({int(m) for m in _CITATION_RE.findall(text)})


# ---------------------------------------------------------------------------
# RAGGenerator
# ---------------------------------------------------------------------------


class RAGGenerator:
    """Generate answers with citation support using OpenAI chat models.

    Args:
        config: Optional generation configuration. Falls back to
            ``get_settings().generation`` when not provided.
        api_key: Optional explicit API key; otherwise read from settings.
    """

    def __init__(
        self,
        config: GenerationConfig | None = None,
        api_key: str | None = None,
    ) -> None:
        settings = get_settings()
        self.config = config or settings.generation
        resolved_key = api_key or settings.openai_api_key

        self._client = OpenAI(api_key=resolved_key)
        self._async_client = AsyncOpenAI(api_key=resolved_key)

        logger.info(
            "RAGGenerator initialised (model=%s, temperature=%.2f).",
            self.config.model,
            self.config.temperature,
        )

    # ------------------------------------------------------------------ #
    # Synchronous generation
    # ------------------------------------------------------------------ #

    def generate(self, query: str, context: str) -> RAGResponse:
        """Generate an answer for *query* grounded in *context*.

        Returns:
            A :class:`RAGResponse` with answer text, parsed citations,
            model name, and token usage.
        """
        messages = self._build_messages(query, context)

        try:
            response = self._client.chat.completions.create(
                model=self.config.model,
                messages=messages,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
            )
        except Exception:
            logger.exception("OpenAI chat completion failed.")
            raise

        choice = response.choices[0]
        answer = choice.message.content or ""
        citations = parse_citations(answer)

        usage = {}
        if response.usage:
            usage = {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            }

        logger.info(
            "Generated answer (%d chars, %d citations, %s).",
            len(answer),
            len(citations),
            usage.get("total_tokens", "?"),
        )

        return RAGResponse(
            answer=answer,
            citations=citations,
            model=response.model,
            usage=usage,
        )

    # ------------------------------------------------------------------ #
    # Async streaming generation
    # ------------------------------------------------------------------ #

    async def generate_stream(self, query: str, context: str) -> AsyncIterator[str]:
        """Stream answer tokens for *query* grounded in *context*.

        Yields:
            Incremental text chunks as they arrive from the API.
        """
        messages = self._build_messages(query, context)

        try:
            stream = await self._async_client.chat.completions.create(
                model=self.config.model,
                messages=messages,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                stream=True,
            )
        except Exception:
            logger.exception("OpenAI streaming chat completion failed.")
            raise

        async for chunk in stream:
            delta = chunk.choices[0].delta
            if delta.content:
                yield delta.content

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #

    @staticmethod
    def _build_messages(query: str, context: str) -> list[dict[str, str]]:
        """Construct the chat messages list."""
        return [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {
                "role": "user",
                "content": _USER_PROMPT_TEMPLATE.format(context=context, query=query),
            },
        ]
