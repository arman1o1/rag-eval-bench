"""Evaluation metrics — RAGAS wrappers, hallucination rate, citation accuracy, latency.

Provides composable metric functions and a MetricsReport dataclass that
aggregates all evaluation results for a single pipeline run.
"""

from __future__ import annotations

import logging
import re
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class MetricsReport:
    """Aggregated metrics from a single evaluation run."""

    # RAGAS metrics (0–1 scale)
    faithfulness: float = 0.0
    answer_relevancy: float = 0.0
    context_precision: float = 0.0
    context_recall: float = 0.0

    # Custom metrics
    hallucination_rate: float = 0.0
    citation_accuracy: float = 0.0

    # Latency stats (seconds)
    avg_retrieval_latency: float = 0.0
    avg_generation_latency: float = 0.0
    avg_total_latency: float = 0.0

    # Per-query breakdown
    per_query: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary for JSON output."""
        return {
            "faithfulness": self.faithfulness,
            "answer_relevancy": self.answer_relevancy,
            "context_precision": self.context_precision,
            "context_recall": self.context_recall,
            "hallucination_rate": self.hallucination_rate,
            "citation_accuracy": self.citation_accuracy,
            "avg_retrieval_latency": self.avg_retrieval_latency,
            "avg_generation_latency": self.avg_generation_latency,
            "avg_total_latency": self.avg_total_latency,
            "num_queries": len(self.per_query),
        }


def compute_ragas_metrics(eval_dataset: Any) -> dict[str, float]:
    """Run RAGAS evaluation on a HuggingFace Dataset.

    Args:
        eval_dataset: HuggingFace Dataset with columns:
            question, answer, contexts, ground_truth.

    Returns:
        Dictionary of metric_name → score.
    """
    try:
        from ragas import evaluate
        from ragas.metrics import (
            answer_relevancy,
            context_precision,
            context_recall,
            faithfulness,
        )
    except ImportError as exc:
        logger.error("RAGAS not installed: %s", exc)
        return {
            "faithfulness": 0.0,
            "answer_relevancy": 0.0,
            "context_precision": 0.0,
            "context_recall": 0.0,
        }

    metrics = [faithfulness, answer_relevancy, context_precision, context_recall]

    try:
        logger.info("Running RAGAS evaluation on %d samples...", len(eval_dataset))
        result = evaluate(eval_dataset, metrics=metrics)
        scores = {
            "faithfulness": float(result.get("faithfulness", 0.0)),
            "answer_relevancy": float(result.get("answer_relevancy", 0.0)),
            "context_precision": float(result.get("context_precision", 0.0)),
            "context_recall": float(result.get("context_recall", 0.0)),
        }
        logger.info("RAGAS scores: %s", scores)
        return scores
    except Exception:
        logger.exception("RAGAS evaluation failed")
        return {
            "faithfulness": 0.0,
            "answer_relevancy": 0.0,
            "context_precision": 0.0,
            "context_recall": 0.0,
        }


def compute_hallucination_rate(
    answers: list[str], contexts: list[list[str]]
) -> float:
    """Compute hallucination rate across Q&A pairs.

    Uses HallucinationDetector from the generation layer. Falls back to a
    simple heuristic (sentence overlap) if the detector is unavailable.

    Args:
        answers: List of generated answers.
        contexts: List of context lists (one per answer).

    Returns:
        Fraction of claims detected as hallucinated (0.0–1.0).
    """
    if not answers:
        return 0.0

    try:
        from src.generation.hallucination import HallucinationDetector

        detector = HallucinationDetector()
        total_claims = 0
        hallucinated_claims = 0

        for answer, ctx_list in zip(answers, contexts):
            context_text = "\n\n".join(ctx_list)
            result = detector.detect(answer=answer, context=context_text)
            total_claims += result.total_claims
            hallucinated_claims += result.hallucinated_claims

        if total_claims == 0:
            return 0.0
        rate = hallucinated_claims / total_claims
        logger.info(
            "Hallucination rate: %.3f (%d/%d claims)",
            rate, hallucinated_claims, total_claims,
        )
        return rate

    except (ImportError, AttributeError):
        logger.warning("HallucinationDetector unavailable, using heuristic fallback")
        return _heuristic_hallucination_rate(answers, contexts)


def _heuristic_hallucination_rate(
    answers: list[str], contexts: list[list[str]]
) -> float:
    """Simple heuristic: fraction of answer sentences not overlapping context."""
    total_sentences = 0
    ungrounded_sentences = 0

    for answer, ctx_list in zip(answers, contexts):
        context_lower = " ".join(ctx_list).lower()
        sentences = [s.strip() for s in re.split(r"[.!?]+", answer) if s.strip()]
        for sentence in sentences:
            total_sentences += 1
            # Check if key words (>50%) appear in context
            words = sentence.lower().split()
            if not words:
                continue
            overlap = sum(1 for w in words if w in context_lower)
            if overlap / len(words) < 0.5:
                ungrounded_sentences += 1

    return ungrounded_sentences / total_sentences if total_sentences > 0 else 0.0


def compute_citation_accuracy(
    answers: list[str], contexts: list[list[str]]
) -> float:
    """Verify [N] citation markers in answers map to actual source chunks.

    Looks for patterns like [1], [2], etc. and checks if the cited index
    corresponds to a valid context chunk.

    Args:
        answers: List of generated answers containing [N] citations.
        contexts: List of context lists (one per answer).

    Returns:
        Fraction of citations that map to valid chunks (0.0–1.0).
    """
    if not answers:
        return 0.0

    total_citations = 0
    valid_citations = 0
    citation_pattern = re.compile(r"\[(\d+)\]")

    for answer, ctx_list in zip(answers, contexts):
        matches = citation_pattern.findall(answer)
        for match in matches:
            total_citations += 1
            idx = int(match) - 1  # citations are 1-indexed
            if 0 <= idx < len(ctx_list):
                valid_citations += 1

    if total_citations == 0:
        logger.info("No citations found in answers")
        return 1.0  # No citations to be wrong about

    accuracy = valid_citations / total_citations
    logger.info(
        "Citation accuracy: %.3f (%d/%d valid)",
        accuracy, valid_citations, total_citations,
    )
    return accuracy


def compute_latency(func: Callable[..., Any], *args: Any, **kwargs: Any) -> tuple[Any, float]:
    """Time a function call and return (result, elapsed_seconds).

    Args:
        func: Callable to time.
        *args: Positional arguments for the callable.
        **kwargs: Keyword arguments for the callable.

    Returns:
        Tuple of (function_result, elapsed_time_in_seconds).
    """
    start = time.perf_counter()
    result = func(*args, **kwargs)
    elapsed = time.perf_counter() - start
    logger.debug("Latency for %s: %.4f s", func.__name__, elapsed)
    return result, elapsed
