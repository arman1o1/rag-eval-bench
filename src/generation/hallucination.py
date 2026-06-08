"""Hallucination detector — claim extraction + NLI-based grounding verification.

Pipeline:
    1. Extract atomic claims from the generated answer using an LLM.
    2. Score each claim against source chunks using a cross-encoder NLI model.
    3. Label each claim as grounded / ungrounded / ambiguous and report the
       overall grounded ratio.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Literal

from openai import OpenAI
from sentence_transformers import CrossEncoder

from src.config import get_settings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class Claim:
    """A single atomic claim extracted from an answer."""

    text: str
    label: Literal["grounded", "ungrounded", "ambiguous"] = "ambiguous"
    confidence: float = 0.0
    source_index: int | None = None


@dataclass
class HallucinationReport:
    """Aggregated grounding report for a generated answer."""

    claims: list[Claim] = field(default_factory=list)
    grounded_ratio: float = 0.0
    flagged_claims: list[Claim] = field(default_factory=list)

    @property
    def total_claims(self) -> int:
        """Total number of claims evaluated."""
        return len(self.claims)

    @property
    def hallucinated_claims(self) -> int:
        """Total number of claims flagged as hallucinated."""
        return len(self.flagged_claims)


# ---------------------------------------------------------------------------
# Prompt for claim extraction
# ---------------------------------------------------------------------------

_CLAIM_EXTRACTION_PROMPT = """\
Extract all atomic factual claims from the following answer. Each claim should \
be a single, self-contained statement that can be independently verified.

Answer:
{answer}

Return a JSON array of strings, each being one atomic claim. Example:
["The Eiffel Tower is in Paris.", "It was built in 1889."]

Return ONLY the JSON array, no additional text."""

# ---------------------------------------------------------------------------
# Thresholds
# ---------------------------------------------------------------------------

_ENTAILMENT_THRESHOLD = 0.7  # score above which a claim is "grounded"
_CONTRADICTION_THRESHOLD = 0.3  # score below which a claim is "ungrounded"


# ---------------------------------------------------------------------------
# HallucinationDetector
# ---------------------------------------------------------------------------


class HallucinationDetector:
    """Detect hallucinated claims in a generated answer.

    Uses an LLM to decompose the answer into atomic claims, then checks
    each claim against the source chunks using a cross-encoder NLI model
    (``cross-encoder/nli-deberta-v3-base``).

    Args:
        nli_model_name: HuggingFace model ID for the NLI cross-encoder.
        api_key: Optional OpenAI API key (falls back to settings).
    """

    def __init__(
        self,
        nli_model_name: str = "cross-encoder/nli-deberta-v3-base",
        api_key: str | None = None,
    ) -> None:
        settings = get_settings()
        resolved_key = api_key or settings.openai_api_key

        self._openai = OpenAI(api_key=resolved_key)
        self._nli_model_name = nli_model_name
        self._nli: CrossEncoder | None = None  # lazy-loaded

        logger.info("HallucinationDetector initialised (nli_model=%s).", nli_model_name)

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def detect(
        self,
        answer: str,
        source_chunks: list[str] | None = None,
        context: str | list[str] | None = None,
    ) -> HallucinationReport:
        """Analyse *answer* for hallucinations against *source_chunks* or *context*.

        Args:
            answer: Generated answer text.
            source_chunks: List of source chunk texts used for generation.
            context: Context string or list of strings (alternative to source_chunks).

        Returns:
            A :class:`HallucinationReport` with per-claim labels and an
            overall ``grounded_ratio``.
        """
        if not answer.strip():
            logger.warning("detect() called with empty answer.")
            return HallucinationReport()

        if source_chunks is None:
            if isinstance(context, str):
                source_chunks = [context]
            elif isinstance(context, list):
                source_chunks = context
            else:
                source_chunks = []

        if not source_chunks:
            logger.warning("detect() called with no source chunks — all claims will be ungrounded.")

        # Step 1 — extract claims via LLM
        raw_claims = self.extract_claims(answer)
        if not raw_claims:
            logger.info("No atomic claims extracted from the answer.")
            return HallucinationReport()

        # Step 2 — verify each claim via NLI cross-encoder
        claims = self._verify_claims(raw_claims, source_chunks)

        # Step 3 — compute aggregates
        grounded = [c for c in claims if c.label == "grounded"]
        grounded_ratio = len(grounded) / len(claims) if claims else 0.0
        flagged = [c for c in claims if c.label != "grounded"]

        report = HallucinationReport(
            claims=claims,
            grounded_ratio=grounded_ratio,
            flagged_claims=flagged,
        )

        logger.info(
            "Hallucination report: %d claims, %.0f%% grounded, %d flagged.",
            len(claims),
            grounded_ratio * 100,
            len(flagged),
        )
        return report

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #

    def _get_nli_model(self) -> CrossEncoder:
        """Lazy-load the NLI cross-encoder to avoid startup cost."""
        if self._nli is None:
            logger.info("Loading NLI cross-encoder: %s", self._nli_model_name)
            self._nli = CrossEncoder(self._nli_model_name)
        return self._nli

    def extract_claims(self, answer: str) -> list[str]:
        """Use an LLM to decompose *answer* into atomic claims."""
        prompt = _CLAIM_EXTRACTION_PROMPT.format(answer=answer)

        try:
            response = self._openai.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=1024,
            )
        except Exception:
            logger.exception("Claim extraction LLM call failed.")
            return []

        raw = response.choices[0].message.content or ""

        # Parse the JSON array from the response
        try:
            # Strip potential markdown code fences
            cleaned = raw.strip()
            if cleaned.startswith("```"):
                cleaned = cleaned.split("\n", 1)[1].rsplit("```", 1)[0]
            claims: list[str] = json.loads(cleaned)
            if not isinstance(claims, list):
                raise ValueError("Expected a JSON array.")
            logger.info("Extracted %d atomic claims.", len(claims))
            return [str(c).strip() for c in claims if str(c).strip()]
        except (json.JSONDecodeError, ValueError):
            logger.error("Failed to parse claims JSON from LLM response: %s", raw[:200])
            return []

    def _verify_claims(
        self,
        claim_texts: list[str],
        source_chunks: list[str],
    ) -> list[Claim]:
        """Score each claim against source chunks using the NLI cross-encoder.

        The NLI model outputs logits for [contradiction, neutral, entailment].
        We use the entailment probability as the grounding confidence.
        """
        if not source_chunks:
            return [
                Claim(text=ct, label="ungrounded", confidence=0.0, source_index=None)
                for ct in claim_texts
            ]

        nli = self._get_nli_model()
        claims: list[Claim] = []

        for claim_text in claim_texts:
            # Build premise-hypothesis pairs: (source_chunk, claim)
            pairs = [(chunk, claim_text) for chunk in source_chunks]

            try:
                # CrossEncoder.predict returns logits of shape (n, 3)
                # Labels: 0=contradiction, 1=neutral, 2=entailment
                scores = nli.predict(pairs, apply_softmax=True)  # type: ignore[arg-type]
            except Exception:
                logger.exception("NLI prediction failed for claim: %s", claim_text[:80])
                claims.append(Claim(text=claim_text, label="ambiguous", confidence=0.0))
                continue

            # Find the source chunk with the highest entailment score
            best_idx = -1
            best_entailment = 0.0

            for idx, score_row in enumerate(scores):
                entailment = float(score_row[2]) if len(score_row) > 2 else float(score_row[-1])
                if entailment > best_entailment:
                    best_entailment = entailment
                    best_idx = idx

            # Label the claim
            if best_entailment >= _ENTAILMENT_THRESHOLD:
                label: Literal["grounded", "ungrounded", "ambiguous"] = "grounded"
            elif best_entailment <= _CONTRADICTION_THRESHOLD:
                label = "ungrounded"
            else:
                label = "ambiguous"

            claims.append(
                Claim(
                    text=claim_text,
                    label=label,
                    confidence=best_entailment,
                    source_index=best_idx if best_idx >= 0 else None,
                )
            )

        return claims
