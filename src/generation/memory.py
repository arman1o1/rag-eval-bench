"""Conversation memory — turn tracking and query reformulation.

Provides simple in-memory storage for conversation turns and an
LLM-powered query reformulator that resolves anaphora (pronouns,
implicit references) using conversation history.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from openai import OpenAI

from src.config import get_settings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Prompt for query reformulation
# ---------------------------------------------------------------------------

_REFORMULATION_PROMPT = """\
You are a query reformulation assistant. Given a conversation history and a \
follow-up question, rewrite the follow-up question as a fully self-contained \
question. Resolve all pronouns and implicit references using the conversation \
history.

Rules:
1. The rewritten question MUST be understandable without any prior context.
2. Do NOT answer the question — only rewrite it.
3. If the follow-up is already self-contained, return it unchanged.
4. Return ONLY the rewritten question, nothing else.

Conversation history:
{history}

Follow-up question: {query}

Rewritten question:"""


# ---------------------------------------------------------------------------
# ConversationMemory
# ---------------------------------------------------------------------------


@dataclass
class Turn:
    """A single conversation turn."""

    query: str
    answer: str


class ConversationMemory:
    """In-memory conversation history with LLM-based query reformulation.

    Args:
        api_key: Optional OpenAI API key (falls back to settings).
        model: Chat model used for reformulation (default: ``gpt-4o-mini``).
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "gpt-4o-mini",
    ) -> None:
        settings = get_settings()
        resolved_key = api_key or settings.openai_api_key

        self._client = OpenAI(api_key=resolved_key)
        self._model = model
        self._turns: list[Turn] = []

        logger.info("ConversationMemory initialised (reformulation model=%s).", model)

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def add_turn(self, query: str, answer: str) -> None:
        """Record a conversation turn.

        Args:
            query: The user's question.
            answer: The assistant's answer.
        """
        self._turns.append(Turn(query=query, answer=answer))
        logger.debug("Added turn %d to conversation memory.", len(self._turns))

    def get_history(self, max_turns: int = 5) -> list[dict[str, str]]:
        """Return recent turns as ``{role, content}`` message dicts.

        Args:
            max_turns: Maximum number of recent turns to return.

        Returns:
            A list of dicts with ``role`` (``'user'`` / ``'assistant'``)
            and ``content`` keys, ordered oldest-first.
        """
        recent = self._turns[-max_turns:] if max_turns > 0 else []
        messages: list[dict[str, str]] = []
        for turn in recent:
            messages.append({"role": "user", "content": turn.query})
            messages.append({"role": "assistant", "content": turn.answer})
        return messages

    def reformulate_query(self, query: str) -> str:
        """Rewrite *query* to be self-contained using conversation history.

        If there is no prior history, the query is returned as-is. Uses
        the OpenAI chat API to resolve pronouns and references.

        Args:
            query: The user's current (potentially context-dependent) query.

        Returns:
            A self-contained version of the query.
        """
        if not self._turns:
            logger.debug("No history — returning query unchanged.")
            return query

        history_str = self._format_history_for_prompt()
        prompt = _REFORMULATION_PROMPT.format(history=history_str, query=query)

        try:
            response = self._client.chat.completions.create(
                model=self._model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=256,
            )
        except Exception:
            logger.exception("Query reformulation LLM call failed — returning original query.")
            return query

        reformulated = (response.choices[0].message.content or "").strip()
        if not reformulated:
            logger.warning("LLM returned empty reformulation — returning original query.")
            return query

        if reformulated != query:
            logger.info("Reformulated query: '%s' → '%s'", query, reformulated)
        return reformulated

    def clear(self) -> None:
        """Clear all stored conversation turns."""
        count = len(self._turns)
        self._turns.clear()
        logger.info("Cleared %d turns from conversation memory.", count)

    @property
    def turn_count(self) -> int:
        """Number of stored turns."""
        return len(self._turns)

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #

    def _format_history_for_prompt(self, max_turns: int = 5) -> str:
        """Format recent history as a readable string for the prompt."""
        recent = self._turns[-max_turns:]
        lines: list[str] = []
        for turn in recent:
            lines.append(f"User: {turn.query}")
            lines.append(f"Assistant: {turn.answer}")
        return "\n".join(lines)
